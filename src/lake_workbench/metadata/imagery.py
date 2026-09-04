"""Scan local imagery and derive valid footprints for site metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import transform_geom as rasterio_transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from lake_workbench.imagery.formats import local_imagery_paths_from_roots, local_label_path
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.metadata.geometry import metric_geometry, polygonal_geometry
from lake_workbench.metadata.sources import image_date, image_tile_name, spatial_intersections
from lake_workbench.regions.config import RegionConfig
from lake_workbench.utils import display_region_path


def inspect_imagery_asset(
    region: RegionConfig,
    site_id: str,
    path: Path,
    sentinel_index,
    footprint_size: int,
    footprint_cache: dict[tuple, tuple[Any, float]] | None = None,
    tile_cache: dict[tuple, str] | None = None,
    label_dirs: list[Path] | None = None,
) -> dict:
    with rasterio.open(path) as src:
        signature = raster_signature(src, footprint_size)
        if footprint_cache is not None and signature in footprint_cache:
            geometry, valid_ratio = footprint_cache[signature]
        else:
            geometry, valid_ratio = valid_footprint(src, footprint_size)
            if footprint_cache is not None:
                footprint_cache[signature] = (geometry, valid_ratio)
        tile_hint = re.search(r"_T([0-9A-Z]{5})_", path.name)
        tile_key = (signature, tile_hint.group(1) if tile_hint else None)
        if tile_cache is not None and tile_key in tile_cache:
            tile = tile_cache[tile_key]
        else:
            tile = image_tile_name(path.name, geometry, sentinel_index)
            if tile_cache is not None:
                tile_cache[tile_key] = tile
        resolution_x = abs(float(src.transform.a))
        resolution_y = abs(float(src.transform.e))
        label = local_label_path(path, label_dirs)
        return {
            "asset_id": f"{site_id}_{path.stem}",
            "site_id": site_id,
            "filename": path.name,
            "product_name": path.stem,
            "acquisition_date": image_date(path.name) or None,
            "mgrs_tile": tile or None,
            "path": display_region_path(path, region),
            "width": int(src.width),
            "height": int(src.height),
            "band_count": int(src.count),
            "crs": str(src.crs or ""),
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "nodata": float(src.nodata) if src.nodata is not None else None,
            "storage_format": path.suffix.lower().lstrip("."),
            "label_path": display_region_path(label, region) if label.exists() else "",
            "valid_ratio": valid_ratio,
            "is_default": 0,
            "geometry": geometry,
        }


def scan_site_imagery(region: RegionConfig, site_dir: Path, sentinel_index, footprint_size: int):
    site_id = f"{region.key}_{site_dir.name}"
    paths = local_imagery_paths_from_roots([site_dir])
    assets = []
    footprint_cache: dict[tuple, tuple[Any, float]] = {}
    tile_cache: dict[tuple, str] = {}
    for path in paths:
        try:
            if path.stat().st_size == 0:
                print(f"[{region.key}] skip empty imagery {site_id}/{path.name}")
                continue
            assets.append(
                inspect_imagery_asset(
                    region,
                    site_id,
                    path,
                    sentinel_index,
                    footprint_size,
                    footprint_cache,
                    tile_cache,
                    [site_dir],
                )
            )
        except (OSError, rasterio.errors.RasterioIOError, ValueError) as exc:
            print(f"[{region.key}] skip unreadable imagery {site_id}/{path.name}: {exc}")
    return site_dir, site_id, assets


def raster_signature(src, footprint_size: int) -> tuple:
    """Identify rasters likely to share the same valid-data footprint."""
    return (
        int(src.width),
        int(src.height),
        tuple(src.transform),
        str(src.crs or ""),
        float(src.nodata) if src.nodata is not None else None,
        int(src.count),
        int(footprint_size),
    )


def valid_footprint(src, max_size: int) -> tuple[Any, float]:
    scale = max(src.width / max_size, src.height / max_size, 1.0)
    width = max(1, int(round(src.width / scale)))
    height = max(1, int(round(src.height / scale)))
    count = src.count
    indexes = list(range(1, count + 1))
    data = src.read(indexes, out_shape=(count, height, width), resampling=Resampling.nearest)
    masks = src.read_masks(indexes, out_shape=(count, height, width), resampling=Resampling.nearest)
    valid = valid_pixel_mask(data, src.nodatavals, masks)
    valid_ratio = float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0
    transform = src.transform * Affine.scale(src.width / width, src.height / height)
    geometries = [
        shape(geometry)
        for geometry, value in shapes(valid.astype("uint8"), mask=valid, transform=transform)
        if int(value) == 1
    ]
    if not geometries:
        raise ValueError(f"imagery has no valid pixels: {src.name}")
    geometry = polygonal_geometry(unary_union(geometries))
    pixel_size = max(abs(float(transform.a)), abs(float(transform.e)))
    geometry = polygonal_geometry(geometry.simplify(pixel_size * 1.5, preserve_topology=True))
    if src.crs and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        geometry = shape(rasterio_transform_geom(src.crs, "EPSG:4326", mapping(geometry), precision=7))
    return polygonal_geometry(geometry), valid_ratio


def sentinel_tiles_for_geometry(geometry, sentinel_index) -> list[str]:
    if sentinel_index is None or sentinel_index.empty:
        return []
    candidates = spatial_intersections(sentinel_index, geometry).copy()
    if candidates.empty:
        return []
    site_metric = metric_geometry(geometry)
    candidates["overlap"] = [
        metric_geometry(tile).intersection(site_metric).area
        for tile in candidates.geometry
    ]
    candidates = candidates[candidates["overlap"] > 0].sort_values(["overlap", "Name"], ascending=[False, True])
    return candidates["Name"].astype(str).tolist()
