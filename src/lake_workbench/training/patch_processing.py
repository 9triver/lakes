"""Pure raster operations used by logical and materialized training patches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from lake_workbench.utils import display_path


DEFAULT_PATCH_SIZE = 512


def derive_patch(image: np.ndarray, target: np.ndarray, valid: np.ndarray, row: dict, output_size: int) -> dict:
    row_off, col_off = int(row["row_off"]), int(row["col_off"])
    logical_size = int(row.get("logical_size") or row.get("window_width") or DEFAULT_PATCH_SIZE)
    image_patch = padded_crop(image, row_off, col_off, fill=0, patch_size=logical_size)
    mask_patch = padded_crop(target, row_off, col_off, fill=255, patch_size=logical_size)
    valid_patch = padded_crop(valid, row_off, col_off, fill=False, patch_size=logical_size)
    if output_size != logical_size:
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


def eligible_actual_patch(actual: dict, patch_id: str, config: dict) -> bool:
    if actual["valid_ratio"] < float(config["min_valid_ratio"]):
        return False
    if actual["water_pixels"] >= int(config["min_water_pixels"]):
        return True
    ratio = float(config["negative_ratio"])
    token = f"{config['id']}:{int(config.get('negative_seed', 42))}:{patch_id}".encode("utf-8")
    sample = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)
    return sample < ratio


def padded_crop(
    array: np.ndarray,
    row_off: int,
    col_off: int,
    fill: Any,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> np.ndarray:
    output_shape = (*array.shape[:-2], patch_size, patch_size)
    output = np.full(output_shape, fill, dtype=array.dtype)
    height = min(patch_size, max(0, array.shape[-2] - row_off))
    width = min(patch_size, max(0, array.shape[-1] - col_off))
    if height and width:
        output[..., :height, :width] = array[..., row_off : row_off + height, col_off : col_off + width]
    return output


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


def rasterize_label(path: Path, src: Any) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return rasterize_label_payload(payload, src)


def rasterize_label_payload(payload: dict, src: Any) -> np.ndarray:
    features = payload.get("features") if payload.get("type") == "FeatureCollection" else [payload]
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) if src.crs else None
    geometries = []
    for feature in features or []:
        if not feature.get("geometry"):
            continue
        geometry = make_valid(shape(feature["geometry"]))
        if geometry.is_empty:
            continue
        if transformer and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geometry = shapely_transform(transformer.transform, geometry)
        geometries.append((geometry, 1))
    return (
        rasterize(
            geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )
        if geometries
        else np.zeros((src.height, src.width), dtype="uint8")
    )


def rasterize_source_variants(variants: list[dict], src: Any) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) if src.crs else None
    geometries = []
    seen = set()
    for variant in variants:
        for feature in variant.get("snapshot", {}).get("features", []):
            geometry_payload = feature.get("geometry")
            if not geometry_payload:
                continue
            token = json.dumps(geometry_payload, sort_keys=True, separators=(",", ":"))
            if token in seen:
                continue
            seen.add(token)
            geometry = make_valid(shape(geometry_payload))
            if geometry.is_empty:
                continue
            if transformer and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                geometry = shapely_transform(transformer.transform, geometry)
            geometries.append((geometry, 1))
    return (
        rasterize(
            geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )
        if geometries
        else np.zeros((src.height, src.width), dtype="uint8")
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
