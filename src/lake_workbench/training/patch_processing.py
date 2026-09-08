"""Pure raster operations used by logical and materialized training patches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pyproj import Transformer
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.utils import display_path


DEFAULT_PATCH_SIZE = 512


@dataclass(frozen=True)
class RasterLabelOverlay:
    """One three-state label mask aligned to its target imagery grid."""

    labels: np.ndarray
    apply: np.ndarray


def resize_patch(
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    valid_patch: np.ndarray,
    output_size: int,
) -> dict:
    """Resize already-windowed Patch arrays and calculate their statistics."""
    if image_patch.shape[-2:] != (output_size, output_size):
        bilinear = Image.Resampling.BILINEAR
        nearest = Image.Resampling.NEAREST
        dtype = image_patch.dtype
        image_patch = np.stack(
            [
                np.asarray(Image.fromarray(band.astype("float32"), mode="F").resize((output_size, output_size), bilinear))
                for band in image_patch
            ]
        ).round().astype(dtype)
        mask_patch = np.asarray(Image.fromarray(mask_patch).resize((output_size, output_size), nearest))
        valid_patch = np.asarray(
            Image.fromarray(valid_patch.astype("uint8")).resize((output_size, output_size), nearest)
        ).astype(bool)
    mask_patch = np.where(valid_patch, mask_patch, 255).astype("uint8")
    valid_pixels = int(valid_patch.sum())
    water_pixels = int(np.count_nonzero(mask_patch == 1))
    return {
        "image": image_patch,
        "mask": mask_patch,
        "valid": valid_patch,
        "valid_pixels": valid_pixels,
        "valid_ratio": valid_pixels / float(output_size**2),
        "water_pixels": water_pixels,
        "water_ratio_valid": water_pixels / valid_pixels if valid_pixels else 0.0,
        "ignore_pixels": int(np.count_nonzero(mask_patch == 255)),
    }


def read_patch_window(
    src: Any,
    row_off: int,
    col_off: int,
    patch_size: int,
    label_geometries: list[tuple[Any, int]],
    scope_geometry: Any | None = None,
    raster_label_overlays: list[RasterLabelOverlay] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one fixed-size raster window and rasterize its label and scope."""
    window = Window(col_off, row_off, patch_size, patch_size)
    image = src.read(window=window, boundless=True, fill_value=0)
    masks = src.read_masks(window=window, boundless=True)
    valid = valid_pixel_mask(image, src.nodatavals, masks)
    transform = window_transform(window, src.transform)
    label = rasterize_geometries(
        label_geometries,
        (patch_size, patch_size),
        transform,
    )
    for overlay in raster_label_overlays or []:
        overlay_labels, overlay_apply = _raster_label_window(
            overlay, row_off, col_off, patch_size
        )
        label[overlay_apply] = overlay_labels[overlay_apply]
    if scope_geometry is not None:
        scope = rasterize_geometries(
            [(scope_geometry, 1)],
            (patch_size, patch_size),
            transform,
        ).astype(bool)
        valid &= scope
    target = np.where(valid, label, 255).astype("uint8")
    return image, target, valid


def eligible_actual_patch(actual: dict, patch_id: str, config: dict) -> bool:
    if actual["valid_ratio"] < float(config["min_valid_ratio"]):
        return False
    if actual["water_pixels"] >= int(config["min_water_pixels"]):
        return True
    ratio = float(config["negative_ratio"])
    token = f"{config['id']}:{int(config.get('negative_seed', 42))}:{patch_id}".encode("utf-8")
    sample = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)
    return sample < ratio


def grid_cell_bounds(
    transform: Any,
    row_off: int,
    col_off: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> tuple[float, float, float, float]:
    corners = [
        transform * (col_off, row_off),
        transform * (col_off + patch_size, row_off),
        transform * (col_off + patch_size, row_off + patch_size),
        transform * (col_off, row_off + patch_size),
    ]
    xs, ys = zip(*corners)
    return (min(xs), min(ys), max(xs), max(ys))


def image_fingerprint(path: Path, src: Any) -> str:
    payload = {
        "path": display_path(path),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "width": src.width,
        "height": src.height,
        "count": src.count,
        "dtype": list(src.dtypes),
        "crs": str(src.crs or ""),
        "transform": tuple(src.transform),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def label_geometries(payload: dict, crs: Any) -> list[tuple[Any, int]]:
    """Transform 0/1/255 label features from WGS84 into a raster CRS once."""
    features = payload.get("features") if payload.get("type") == "FeatureCollection" else [payload]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True) if crs else None
    geometries = []
    for feature in features or []:
        if not feature.get("geometry"):
            continue
        geometry = make_valid(shape(feature["geometry"]))
        if geometry.is_empty:
            continue
        if transformer and str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geometry = shapely_transform(transformer.transform, geometry)
        properties = feature.get("properties") or {}
        label_class = str(properties.get("label_class") or "").strip().lower()
        raw_value = properties.get("label_value")
        if raw_value is None:
            value = {"background": 0, "water": 1, "ignore": 255}.get(
                label_class, 1
            )
        else:
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid label_value: {raw_value}") from exc
        if value not in {0, 1, 255}:
            raise ValueError(f"label_value must be 0, 1 or 255: {value}")
        geometries.append((geometry, value))
    return geometries


def raster_label_overlays(
    payload: dict,
    label_path: Path,
    target: Any,
) -> list[RasterLabelOverlay]:
    """Load sample raster-label sidecars and align each one to the image grid."""
    properties = payload.get("properties") or {}
    sources = properties.get("raster_label_sources") or []
    if not isinstance(sources, list):
        raise ValueError("raster_label_sources must be a list")
    overlays = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid raster label source metadata")
        source_path = _raster_label_path(label_path, source)
        with np.load(source_path, allow_pickle=False) as arrays:
            labels = np.asarray(arrays["labels"], dtype=np.uint8)
            valid = np.asarray(arrays["valid"], dtype=bool)
            transform_values = np.asarray(arrays["transform"], dtype=np.float64)
            crs = str(np.asarray(arrays["crs"]).item())
        if labels.ndim != 2 or labels.shape != valid.shape:
            raise ValueError(f"invalid raster label dimensions: {source_path}")
        if transform_values.size != 6 or not crs:
            raise ValueError(f"invalid raster label grid: {source_path}")
        values = set(int(value) for value in np.unique(labels))
        if not values.issubset({0, 1, 255}):
            raise ValueError(f"invalid raster label values: {source_path}")
        overlays.append(
            _align_raster_label(
                labels,
                valid & (labels != 0),
                Affine(*transform_values.tolist()),
                crs,
                target,
            )
        )
    return overlays


def training_label_fingerprint(label_path: Path, payload: dict | None = None) -> str:
    """Fingerprint a label snapshot together with all raster sidecars."""
    if payload is None:
        serialized = label_path.read_bytes()
        payload = json.loads(serialized.decode("utf-8"))
    else:
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    digest = hashlib.sha256(serialized)
    sources = (payload.get("properties") or {}).get("raster_label_sources") or []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid raster label source metadata")
        source_path = _raster_label_path(label_path, source)
        digest.update(source_path.name.encode("utf-8"))
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def _raster_label_path(label_path: Path, source: dict) -> Path:
    path_text = str(source.get("path") or "").strip()
    if not path_text:
        raise ValueError("raster label source has no path")
    path = Path(path_text)
    if not path.is_absolute():
        path = label_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"raster label not found: {path}")
    return path


def _align_raster_label(
    labels: np.ndarray,
    apply: np.ndarray,
    transform: Affine,
    crs: str,
    target: Any,
) -> RasterLabelOverlay:
    target_shape = (target.height, target.width)
    if (
        labels.shape == target_shape
        and str(target.crs or "") == crs
        and transform.almost_equals(target.transform)
    ):
        return RasterLabelOverlay(labels=labels, apply=apply)
    if not target.crs:
        raise ValueError("target imagery has no CRS for raster label alignment")
    aligned_labels = np.zeros(target_shape, dtype=np.uint8)
    aligned_apply = np.zeros(target_shape, dtype=np.uint8)
    reproject(
        source=labels,
        destination=aligned_labels,
        src_transform=transform,
        src_crs=crs,
        dst_transform=target.transform,
        dst_crs=target.crs,
        resampling=Resampling.nearest,
    )
    reproject(
        source=apply.astype(np.uint8),
        destination=aligned_apply,
        src_transform=transform,
        src_crs=crs,
        dst_transform=target.transform,
        dst_crs=target.crs,
        resampling=Resampling.nearest,
    )
    return RasterLabelOverlay(labels=aligned_labels, apply=aligned_apply.astype(bool))


def _raster_label_window(
    overlay: RasterLabelOverlay,
    row_off: int,
    col_off: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((patch_size, patch_size), dtype=np.uint8)
    apply = np.zeros((patch_size, patch_size), dtype=bool)
    source_row_stop = min(row_off + patch_size, overlay.labels.shape[0])
    source_col_stop = min(col_off + patch_size, overlay.labels.shape[1])
    if source_row_stop <= row_off or source_col_stop <= col_off:
        return labels, apply
    height = source_row_stop - row_off
    width = source_col_stop - col_off
    labels[:height, :width] = overlay.labels[
        row_off:source_row_stop, col_off:source_col_stop
    ]
    apply[:height, :width] = overlay.apply[
        row_off:source_row_stop, col_off:source_col_stop
    ]
    return labels, apply


def rasterize_geometries(
    geometries: list[tuple[Any, int]],
    out_shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    return (
        rasterize(
            geometries,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        if geometries
        else np.zeros(out_shape, dtype="uint8")
    )


def write_preview(path: Path, image: np.ndarray, mask: np.ndarray, valid: np.ndarray, overlay: bool = True) -> None:
    preview_image(image, mask, valid, overlay=overlay).save(path)


def preview_image(image: np.ndarray, mask: np.ndarray, valid: np.ndarray, overlay: bool = True) -> Image.Image:
    indexes = [2, 1, 0] if image.shape[0] >= 3 else [0, 0, 0]
    rgb = np.stack([image[index] for index in indexes], axis=-1).astype("float32")
    output = np.zeros_like(rgb, dtype="uint8")
    for index in range(3):
        values = rgb[..., index]
        selected = values[valid & np.isfinite(values)]
        low, high = np.percentile(selected, [2, 98]) if selected.size else (0, 1)
        if high <= low:
            high = low + 1
        output[..., index] = np.clip((values - low) * 255 / (high - low), 0, 255).astype("uint8")
    output[~valid] = [34, 34, 34]
    if overlay:
        water = mask == 1
        output[water] = (
            output[water].astype("uint16") * 25 // 100
            + np.array([255, 32, 128], dtype="uint16") * 75 // 100
        ).astype("uint8")
    return Image.fromarray(output)
