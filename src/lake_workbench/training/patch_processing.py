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
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.utils import display_path


DEFAULT_PATCH_SIZE = 512


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
    """Transform label features from WGS84 into a raster CRS once."""
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
        geometries.append((geometry, 1))
    return geometries


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
