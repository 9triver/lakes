"""Scene-adaptive water evidence from local five-band multispectral imagery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import reproject, transform_bounds

from lake_workbench.automatic_labels.spectral_methods import (
    bounded_otsu_threshold,
    sigmoid,
    waterdetect_cluster,
)
from lake_workbench.imagery.validity import valid_pixel_mask


BACKGROUND_LABEL = 0
WATER_LABEL = 1
IGNORE_LABEL = 255
DEFAULT_WATER_THRESHOLD = 0.62
DEFAULT_SPECTRAL_MAX_DIMENSION = 1024
STABLE_LABEL_MAX_DIMENSION = 2048
SPECTRAL_ALGORITHM_VERSION = "spectral_consensus_v3"


@dataclass(frozen=True)
class ExternalQualityMasks:
    """Quality masks aligned to the spectral classification grid."""

    invalid: np.ndarray
    cloud: np.ndarray
    snow: np.ndarray
    shadow: np.ndarray
    sources: tuple[str, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpectralClassification:
    water_score: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    quality_valid: np.ndarray
    methods: dict[str, np.ndarray]
    indexes: dict[str, np.ndarray]
    bands: dict[str, int]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SpectralEvidence(SpectralClassification):
    transform: Affine
    crs: Any

    @property
    def shape(self) -> tuple[int, int]:
        return self.water_score.shape


def spectral_band_indexes(src: Any) -> dict[str, int]:
    """Resolve the five Sentinel-like bands used by the local products."""
    descriptions = [str(value or "").strip().lower() for value in (src.descriptions or ())]

    def find(*tokens: str) -> int | None:
        for index, description in enumerate(descriptions, start=1):
            if any(token in description for token in tokens):
                return index
        return None

    result = {
        "blue": find("rhot_492", "rhos_492", "b02", "blue"),
        "green": find("rhot_560", "rhos_560", "b03", "green"),
        "red": find("rhot_665", "rhos_665", "b04", "red"),
        "nir": find("rhot_833", "rhos_833", "b08", "nir"),
        "swir1": find("rhot_1614", "rhos_1614", "b11", "swir1", "swir 1"),
    }
    if all(value is not None for value in result.values()):
        return {key: int(value) for key, value in result.items() if value is not None}
    if src.count >= 5:
        return {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5}
    raise ValueError(
        "Automatic spectral labeling needs B02/blue, B03/green, B04/red, "
        "B08/NIR and B11/SWIR1 bands"
    )


def spectral_processing_level(src: Any, path: Path | None = None) -> str:
    """Infer whether values are top-of-atmosphere (L1C) or surface (L2A)."""
    descriptions = " ".join(str(value or "").lower() for value in (src.descriptions or ()))
    tags = " ".join(f"{key}={value}" for key, value in src.tags().items()).lower()
    name = str(path or getattr(src, "name", "")).lower()
    combined = f"{descriptions} {tags} {name}"
    if "rhos_" in combined or "msil2a" in combined or "level-2a" in combined:
        return "L2A"
    if "rhot_" in combined or "msil1c" in combined or "level-1c" in combined:
        return "L1C"
    return "unknown"


def normalize_reflectance(
    image: np.ndarray,
    valid: np.ndarray,
    *,
    scales: tuple[float, ...] | list[float] | None = None,
    offsets: tuple[float, ...] | list[float] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Normalize stored values to reflectance while retaining valid negatives."""
    reflectance = image.astype(np.float32, copy=True)
    band_count = reflectance.shape[0]
    scale_values = list(scales or ())
    offset_values = list(offsets or ())
    metadata_applied = (
        len(scale_values) == band_count
        and len(offset_values) == band_count
        and any(
            not np.isclose(scale, 1.0) or not np.isclose(offset, 0.0)
            for scale, offset in zip(scale_values, offset_values, strict=True)
        )
    )
    if metadata_applied:
        for index, (scale, offset) in enumerate(
            zip(scale_values, offset_values, strict=True)
        ):
            reflectance[index] = reflectance[index] * float(scale) + float(offset)

    selected = reflectance[:, valid]
    finite_positive = selected[np.isfinite(selected) & (selected > 0)]
    divisor = 1.0
    if finite_positive.size and float(np.percentile(finite_positive, 99)) > 2.0:
        # Local Sentinel products use the conventional 10000 reflectance scale.
        divisor = 10000.0
        reflectance /= divisor
    return reflectance, divisor, metadata_applied


def classify_spectral_array(
    image: np.ndarray,
    valid: np.ndarray,
    bands: dict[str, int],
    *,
    processing_level: str = "unknown",
    scales: tuple[float, ...] | list[float] | None = None,
    offsets: tuple[float, ...] | list[float] | None = None,
    threshold: float = DEFAULT_WATER_THRESHOLD,
    random_seed: int = 42,
    cluster_sample_size: int = 2500,
    external_quality: ExternalQualityMasks | None = None,
) -> SpectralClassification:
    """Combine adaptive clustering, bounded Otsu, and five-band DSWx tests.

    ``water_score`` is an ensemble score, not a calibrated probability. It is
    only used with the conservative three-state label rule below.
    """
    reflectance, divisor, metadata_applied = normalize_reflectance(
        image, valid, scales=scales, offsets=offsets
    )
    blue = reflectance[bands["blue"] - 1]
    green = reflectance[bands["green"] - 1]
    red = reflectance[bands["red"] - 1]
    nir = reflectance[bands["nir"] - 1]
    swir1 = reflectance[bands["swir1"] - 1]

    ndwi = _normalized_difference(green, nir)
    mndwi = _normalized_difference(green, swir1)
    ndvi = _normalized_difference(nir, red)
    indexes = {"ndwi": ndwi, "mndwi": mndwi, "ndvi": ndvi}

    required = np.stack((blue, green, red, nir, swir1))
    physical = np.all((required >= -0.2) & (required <= 1.6), axis=0)
    visible_mean = (blue + green + red) / 3.0
    visible_range = np.maximum.reduce((blue, green, red)) - np.minimum.reduce(
        (blue, green, red)
    )
    whiteness = visible_range / np.maximum(np.abs(visible_mean), 1e-4)
    likely_cloud = (
        (visible_mean > 0.25)
        & (nir > 0.20)
        & (swir1 > 0.15)
        & (whiteness < 0.8)
    )
    likely_snow = (mndwi > 0.40) & (green > 0.20) & (nir > 0.15)
    external_invalid = (
        external_quality.invalid
        if external_quality is not None
        else np.zeros(valid.shape, dtype=bool)
    )
    external_cloud = (
        external_quality.cloud
        if external_quality is not None
        else np.zeros(valid.shape, dtype=bool)
    )
    external_snow = (
        external_quality.snow
        if external_quality is not None
        else np.zeros(valid.shape, dtype=bool)
    )
    external_shadow = (
        external_quality.shadow
        if external_quality is not None
        else np.zeros(valid.shape, dtype=bool)
    )
    for name, mask in (
        ("invalid", external_invalid),
        ("cloud", external_cloud),
        ("snow", external_snow),
        ("shadow", external_shadow),
    ):
        if mask.shape != valid.shape:
            raise ValueError(f"external quality mask {name} has the wrong shape")
    likely_shadow = (
        (visible_mean < 0.12)
        & (nir < 0.15)
        & (swir1 < 0.10)
        & (ndwi < -0.05)
        & (mndwi < -0.05)
    )
    shadow_mask = external_shadow | likely_shadow
    quality_valid = (
        valid
        & physical
        & ~likely_cloud
        & ~likely_snow
        & ~external_invalid
        & ~external_cloud
        & ~external_snow
        & ~shadow_mask
    )

    cluster = waterdetect_cluster(
        mndwi,
        ndwi,
        nir,
        swir1,
        quality_valid,
        random_seed=random_seed,
        max_sample_size=cluster_sample_size,
    )
    otsu_threshold = bounded_otsu_threshold(mndwi[quality_valid])
    otsu_score = sigmoid((mndwi - otsu_threshold) / 0.055)
    otsu_vote = (
        quality_valid
        & (mndwi >= otsu_threshold)
        & (ndwi > -0.15)
        & (nir < 0.30)
        & (swir1 < 0.20)
    )

    dswx_test_1 = mndwi > 0.124
    dswx_test_2 = (green + red) > (nir + swir1)
    dswx_test_4 = (
        (mndwi > -0.44) & (swir1 < 0.09) & (nir < 0.15) & (ndvi < 0.7)
    )
    dswx_count = (
        dswx_test_1.astype(np.uint8)
        + dswx_test_2.astype(np.uint8)
        + dswx_test_4.astype(np.uint8)
    )
    dswx_vote = quality_valid & (dswx_count == 3)
    dswx_score = dswx_count.astype(np.float32) / 3.0

    methods = {
        "waterdetect_cluster": quality_valid & cluster.vote,
        "bounded_mndwi_otsu": otsu_vote,
        "dswx_five_band_subset": dswx_vote,
    }
    water_score = np.clip(
        0.45 * cluster.water_score + 0.30 * otsu_score + 0.25 * dswx_score,
        0.0,
        1.0,
    ).astype(np.float32)
    water_score[~quality_valid] = 0.0
    labels = spectral_labels(
        water_score, methods, valid, quality_valid, threshold=threshold
    )

    valid_pixels = int(np.count_nonzero(valid))
    quality_pixels = int(np.count_nonzero(quality_valid))
    diagnostics: dict[str, Any] = {
        "algorithm": SPECTRAL_ALGORITHM_VERSION,
        "processing_level": processing_level,
        "reflectance_divisor": divisor,
        "raster_scale_offset_applied": metadata_applied,
        "ndwi_mean": _valid_mean(ndwi, quality_valid),
        "mndwi_mean": _valid_mean(mndwi, quality_valid),
        "ndvi_mean": _valid_mean(ndvi, quality_valid),
        "otsu_mndwi_threshold": otsu_threshold,
        "cluster_status": cluster.status,
        "cluster_best_k": cluster.best_k,
        "cluster_calinski_harabasz": cluster.score,
        "cluster_sample_pixels": cluster.sample_size,
        "cluster_water_mndwi": cluster.water_cluster_mndwi,
        "cloud_pixels": int(np.count_nonzero(valid & (likely_cloud | external_cloud))),
        "snow_pixels": int(np.count_nonzero(valid & (likely_snow | external_snow))),
        "shadow_pixels": int(np.count_nonzero(valid & shadow_mask)),
        "quality_ignored_pixels": valid_pixels - quality_pixels,
        "waterdetect_vote_ratio": _valid_mean(methods["waterdetect_cluster"], quality_valid),
        "otsu_vote_ratio": _valid_mean(otsu_vote, quality_valid),
        "dswx_subset_vote_ratio": _valid_mean(dswx_vote, quality_valid),
        "spectral_water_score_mean": _valid_mean(water_score, quality_valid),
        "score_is_calibrated_probability": False,
        "quality_mask_sources": list(external_quality.sources) if external_quality else [],
        "quality_mask_errors": list(external_quality.errors) if external_quality else [],
    }
    return SpectralClassification(
        water_score=water_score,
        labels=labels,
        valid=valid,
        quality_valid=quality_valid,
        methods=methods,
        indexes=indexes,
        bands=bands,
        diagnostics=diagnostics,
    )


def spectral_labels(
    water_score: np.ndarray,
    methods: dict[str, np.ndarray],
    valid: np.ndarray,
    quality_valid: np.ndarray,
    *,
    threshold: float = DEFAULT_WATER_THRESHOLD,
) -> np.ndarray:
    """Return 0 background, 1 confident water, and 255 uncertain/invalid."""
    bounded_threshold = max(0.5, min(0.95, float(threshold)))
    agreement = np.zeros(water_score.shape, dtype=np.uint8)
    for vote in methods.values():
        agreement += vote.astype(np.uint8)
    labels = np.full(water_score.shape, IGNORE_LABEL, dtype=np.uint8)
    confident_background = (
        quality_valid
        & (agreement == 0)
        & (water_score <= min(0.40, 1.0 - bounded_threshold + 0.05))
    )
    confident_water = (
        quality_valid & (agreement >= 2) & (water_score >= bounded_threshold)
    )
    labels[confident_background] = BACKGROUND_LABEL
    labels[confident_water] = WATER_LABEL
    labels[~valid] = IGNORE_LABEL
    return labels


def read_spectral_evidence(
    path: Path,
    bounds_wgs84: tuple[float, float, float, float],
    *,
    max_dimension: int = DEFAULT_SPECTRAL_MAX_DIMENSION,
    threshold: float = DEFAULT_WATER_THRESHOLD,
    quality_path: Path | None = None,
) -> SpectralEvidence:
    """Read the requested view and calculate scene-adaptive water evidence."""
    with rasterio.open(path) as src:
        if not src.crs:
            raise ValueError(f"Raster has no CRS: {path}")
        source_bounds = transform_bounds(
            "EPSG:4326", src.crs, *bounds_wgs84, densify_pts=21
        )
        left = max(source_bounds[0], src.bounds.left)
        bottom = max(source_bounds[1], src.bounds.bottom)
        right = min(source_bounds[2], src.bounds.right)
        top = min(source_bounds[3], src.bounds.top)
        if right <= left or top <= bottom:
            raise ValueError("Current map view does not overlap the selected imagery")
        window = (
            from_bounds(left, bottom, right, top, transform=src.transform)
            .round_offsets()
            .round_lengths()
        )
        read_height = max(1, int(window.height))
        read_width = max(1, int(window.width))
        scale = min(1.0, max_dimension / max(read_height, read_width))
        height = max(1, int(round(read_height * scale)))
        width = max(1, int(round(read_width * scale)))
        source_bands = spectral_band_indexes(src)
        band_names = ("blue", "green", "red", "nir", "swir1")
        indexes = [source_bands[name] for name in band_names]
        image = src.read(
            indexes,
            window=window,
            out_shape=(len(indexes), height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        masks = src.read_masks(
            indexes,
            window=window,
            out_shape=(len(indexes), height, width),
            resampling=Resampling.bilinear,
        )
        nodata = tuple(src.nodatavals[index - 1] for index in indexes)
        # Require a fully valid source neighborhood at interpolated edges.
        masks = np.where(masks >= 254, 255, 0).astype(np.uint8)
        valid = valid_pixel_mask(image, nodata, masks)
        bands = {name: index + 1 for index, name in enumerate(band_names)}
        processing_level = spectral_processing_level(src, path)
        transform = window_transform(window, src.transform) * Affine.scale(
            read_width / width, read_height / height
        )
        crs = src.crs
        scales = tuple(src.scales[index - 1] for index in indexes)
        offsets = tuple(src.offsets[index - 1] for index in indexes)

    external_quality = read_external_quality_masks(
        path,
        quality_path,
        crs,
        transform,
        (height, width),
    )

    classification = classify_spectral_array(
        image,
        valid,
        bands,
        processing_level=processing_level,
        scales=scales,
        offsets=offsets,
        threshold=threshold,
        external_quality=external_quality,
    )
    return SpectralEvidence(
        water_score=classification.water_score,
        labels=classification.labels,
        valid=classification.valid,
        quality_valid=classification.quality_valid,
        methods=classification.methods,
        indexes=classification.indexes,
        bands=classification.bands,
        diagnostics=classification.diagnostics,
        transform=transform,
        crs=crs,
    )


def read_external_quality_masks(
    image_path: Path,
    quality_path: Path | None,
    target_crs: Any,
    target_transform: Affine,
    target_shape: tuple[int, int],
) -> ExternalQualityMasks | None:
    """Read optional Sentinel quality layers and align them to the image grid.

    Downloaded SAFE products normally provide SCL, while some products or
    sidecar exports provide cloud/snow probabilities or QA60. Local IMG/TIFF
    products often have none of these files, so the caller can continue with
    the spectral quality heuristics in that case.
    """
    paths = _quality_layer_paths(image_path, quality_path)
    if not paths:
        return None
    invalid = np.zeros(target_shape, dtype=bool)
    cloud = np.zeros(target_shape, dtype=bool)
    snow = np.zeros(target_shape, dtype=bool)
    shadow = np.zeros(target_shape, dtype=bool)
    sources: list[str] = []
    errors: list[str] = []
    for kind, path in paths.items():
        try:
            values, covered = _reproject_quality_layer(
                path, target_crs, target_transform, target_shape
            )
        except (OSError, ValueError, rasterio.errors.RasterioIOError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if kind == "scl":
            classes = np.rint(values).astype(np.int16)
            cloud |= covered & np.isin(classes, (8, 9, 10))
            snow |= covered & (classes == 11)
            shadow |= covered & np.isin(classes, (2, 3))
            invalid |= covered & np.isin(classes, (0, 1, 2, 3, 7, 8, 9, 10, 11))
            sources.append(f"SCL:{path.name}")
        elif kind == "cloud_probability":
            cloud_pixels = covered & (values >= 40.0)
            cloud |= cloud_pixels
            invalid |= cloud_pixels
            sources.append(f"CLDPRB:{path.name}")
        elif kind == "snow_probability":
            snow_pixels = covered & (values >= 40.0)
            snow |= snow_pixels
            invalid |= snow_pixels
            sources.append(f"SNWPRB:{path.name}")
        elif kind == "qa60":
            qa = np.rint(values).astype(np.uint16)
            cloud_pixels = covered & (
                ((qa & (1 << 10)) != 0) | ((qa & (1 << 11)) != 0)
            )
            cloud |= cloud_pixels
            invalid |= cloud_pixels
            sources.append(f"QA60:{path.name}")
    return ExternalQualityMasks(
        invalid=invalid,
        cloud=cloud,
        snow=snow,
        shadow=shadow,
        sources=tuple(sources),
        errors=tuple(errors),
    )


def _quality_layer_paths(image_path: Path, quality_path: Path | None) -> dict[str, Path]:
    root = image_path.parent
    if quality_path:
        root = quality_path if quality_path.is_dir() else quality_path.parent
    roots = [root]
    candidates: dict[str, list[Path]] = {
        "scl": [],
        "cloud_probability": [],
        "snow_probability": [],
        "qa60": [],
    }
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [item for item in root.rglob("*") if item.is_file()]
        for item in files:
            if item in seen:
                continue
            seen.add(item)
            name = item.name.lower()
            if "scl" in name:
                candidates["scl"].append(item)
            elif "cldprb" in name or "cloudprob" in name:
                candidates["cloud_probability"].append(item)
            elif "snwprb" in name or "snowprob" in name:
                candidates["snow_probability"].append(item)
            elif "qa60" in name:
                candidates["qa60"].append(item)
    selected: dict[str, Path] = {}
    for kind, items in candidates.items():
        if items:
            # Prefer the highest-resolution named layer when multiple SAFE
            # companions are present, then keep selection deterministic.
            selected[kind] = sorted(
                items,
                key=lambda item: ("10m" not in item.name.lower(), len(item.parts), str(item)),
            )[0]
    return selected


def _reproject_quality_layer(
    path: Path,
    target_crs: Any,
    target_transform: Affine,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        sentinel = np.nan
        values = np.full(target_shape, sentinel, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=values,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=sentinel,
            resampling=Resampling.nearest,
        )
    return values, np.isfinite(values)


def _normalized_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = first + second
    return np.divide(
        first - second,
        denominator,
        out=np.zeros_like(first, dtype=np.float32),
        where=np.abs(denominator) > 1e-6,
    )


def _valid_mean(values: np.ndarray, valid: np.ndarray) -> float:
    return float(values[valid].mean()) if np.any(valid) else 0.0
