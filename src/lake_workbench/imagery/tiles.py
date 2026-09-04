"""XYZ tile rendering for local and downloaded raster imagery."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, transform_bounds

from lake_workbench.geo import boxes_intersect
from lake_workbench.imagery.display import display_band_indexes, to_display_rgb
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.paths import PROJECT_ROOT


WEB_MERCATOR_LIMIT = 20037508.342789244


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def image_cache_key(site, size: int, padding: float, tci_rows: list[dict]) -> str:
    payload = {
        "site_id": site.site_id,
        "bbox": [round(value, 8) for value in site.bbox],
        "size": size,
        "padding": round(padding, 4),
        "products": [
            {
                "tile": row["tile"],
                "date": row["date"],
                "product": row["product"],
                "path": display_path(row["tci_path"]),
                "mtime": row["tci_path"].stat().st_mtime,
            }
            for row in sorted(tci_rows, key=lambda item: item["tile"])
        ],
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:24]


def tile_cache_key(site, z: int, x: int, y: int, padding: float, tci_rows: list[dict]) -> str:
    payload = {
        "render_version": 2,
        "site_id": site.site_id,
        "z": int(z),
        "x": int(x),
        "y": int(y),
        "padding": round(padding, 4),
        "products": [
            {
                "tile": row["tile"],
                "product": row["product"],
                "path": display_path(row["tci_path"]),
                "mtime": row["tci_path"].stat().st_mtime,
            }
            for row in sorted(tci_rows, key=lambda item: item["tile"])
        ],
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:28]


def mosaic_source_meta(tci_rows: list[dict]) -> dict:
    rows = sorted(tci_rows, key=lambda row: row["tile"])
    return {
        "tiles": [row["tile"] for row in rows],
        "dates": sorted({str(row["date"]) for row in rows}),
        "products": [str(row["product"]) for row in rows],
        "tci_path": [display_path(row["tci_path"]) for row in rows],
    }


def rows_bounds(tci_rows: list[dict]) -> tuple[float, float, float, float]:
    bounds = []
    for row in tci_rows:
        with rasterio.open(row["tci_path"]) as src:
            bounds.append(transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def xyz_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    tiles = 2**int(z)
    size = 2 * WEB_MERCATOR_LIMIT / tiles
    left = -WEB_MERCATOR_LIMIT + int(x) * size
    right = left + size
    top = WEB_MERCATOR_LIMIT - int(y) * size
    bottom = top - size
    return (left, bottom, right, top)


def render_tci_xyz_tile(
    tci_rows: list[dict],
    bounds_3857: tuple[float, float, float, float],
    tile_size: int = 256,
) -> bytes:
    output = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
    filled = np.zeros((tile_size, tile_size), dtype=bool)
    dst_transform = transform_from_bounds(*bounds_3857, tile_size, tile_size)
    ordered_rows = sorted(tci_rows, key=lambda row: float(row.get("valid_ratio", 0) or 0), reverse=True)
    for row in ordered_rows:
        with rasterio.open(row["tci_path"]) as src:
            raster_bounds_3857 = transform_bounds(src.crs, "EPSG:3857", *src.bounds, densify_pts=21)
            if not boxes_intersect(bounds_3857, raster_bounds_3857):
                continue
            indexes = display_band_indexes(src, row)
            raw = np.zeros((3, tile_size, tile_size), dtype=np.float32)
            reproject(
                source=rasterio.band(src, indexes),
                destination=raw,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
            masks = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            for mask_index, source_index in enumerate(indexes):
                source_mask = src.read_masks(indexes=[source_index])[0]
                reproject(
                    source=source_mask,
                    destination=masks[mask_index],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:3857",
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
            nodata = tuple(src.nodatavals[index - 1] for index in indexes)
            data = to_display_rgb(raw)
        valid = valid_pixel_mask(raw, nodata, masks) & ~filled
        if np.any(valid):
            output[:, valid] = data[:, valid]
            filled |= valid
        if np.all(filled):
            break
    rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.moveaxis(output, 0, -1)
    rgba[:, :, 3] = np.where(filled, 255, 0).astype(np.uint8)
    image = Image.fromarray(rgba, "RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
