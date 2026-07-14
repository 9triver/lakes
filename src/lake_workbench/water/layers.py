"""ESA WorldCover and JRC surface-water polygon layers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.validation import make_valid

from lake_workbench.geo import smooth_water_geometry, transform_geom
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import RegionConfig


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def esa_polygon_cache_path(region: RegionConfig, site_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", site_id)
    return region.esa_polygon_dir / safe_id / "esa_water.geojson"


def read_esa_polygon_cache(region: RegionConfig, site_id: str) -> dict | None:
    path = esa_polygon_cache_path(region, site_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload.setdefault("properties", {})
    payload["properties"]["cached"] = True
    payload["properties"]["cache_path"] = display_path(path)
    return payload


def write_esa_polygon_cache(region: RegionConfig, site_id: str, layer: dict) -> Path:
    path = esa_polygon_cache_path(region, site_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
    return path



def build_esa_smoothed_layer(region: RegionConfig, site: Any) -> dict | None:
    try:
        geom = site.geometry.buffer(max(site.bbox[2] - site.bbox[0], site.bbox[3] - site.bbox[1]) * 0.15)
        paths = [region.esa_water_mask] if region.esa_water_mask.exists() else esa_worldcover_tile_paths(region, geom)
        geoms = []
        for path in paths:
            geoms.extend(water_polygons_from_raster(path, geom, site.geometry, lambda arr: arr == 80 if path != region.esa_water_mask else arr == 1))
    except Exception:
        return None
    if not geoms:
        return {
            "source": "ESA WorldCover 2021 water mask, smoothed",
            "geometry": None,
            "properties": {
                "water_id": f"ESA_{site.site_id}",
                "empty": True,
                "pre_generated": False,
            },
        }
    source = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
    smoothed = smooth_esa_geometry(source, site.area_km2)
    return {
        "source": "ESA WorldCover 2021 water mask, smoothed",
        "geometry": mapping(smoothed),
        "properties": {
            "water_id": f"ESA_{site.site_id}",
            "pre_generated": False,
        },
    }


def jrc_polygon_cache_path(region: RegionConfig, site_id: str, threshold: int) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", site_id)
    return region.jrc_polygon_dir / safe_id / f"jrc_occurrence_ge{threshold}.geojson"


def read_jrc_polygon_cache(region: RegionConfig, site_id: str, threshold: int) -> dict | None:
    path = jrc_polygon_cache_path(region, site_id, threshold)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload.setdefault("properties", {})
    payload["properties"]["cached"] = True
    payload["properties"]["cache_path"] = display_path(path)
    return payload


def write_jrc_polygon_cache(region: RegionConfig, site_id: str, threshold: int, layer: dict) -> Path:
    path = jrc_polygon_cache_path(region, site_id, threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
    return path


def available_jrc_thresholds(region: RegionConfig, site_id: str) -> list[int]:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", site_id)
    folder = region.jrc_polygon_dir / safe_id
    if not folder.exists():
        return []
    thresholds = []
    for path in folder.glob("jrc_occurrence_ge*.geojson"):
        match = re.search(r"ge(\d+)\.geojson$", path.name)
        if match:
            thresholds.append(int(match.group(1)))
    return sorted(thresholds)


def build_jrc_occurrence_layer(region: RegionConfig, site: Any, threshold: int) -> dict | None:
    threshold = max(1, min(100, int(threshold)))
    try:
        geom = site.geometry.buffer(max(site.bbox[2] - site.bbox[0], site.bbox[3] - site.bbox[1]) * 0.2)
        paths = [region.jrc_occurrence] if region.jrc_occurrence.exists() else jrc_occurrence_tile_paths(region, geom)
        geoms = []
        for path in paths:
            geoms.extend(water_polygons_from_raster(path, geom, site.geometry, lambda arr: (arr >= threshold) & (arr <= 100)))
    except Exception:
        return None
    if not geoms:
        return {
            "source": "JRC GSW occurrence 2021",
            "geometry": None,
            "properties": {
                "water_id": f"JRC_{site.site_id}_{threshold}",
                "threshold": threshold,
                "empty": True,
            },
        }
    source = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
    smoothed = smooth_jrc_geometry(source, site.area_km2)
    return {
        "source": "JRC GSW occurrence 2021",
        "geometry": mapping(smoothed),
        "properties": {
            "water_id": f"JRC_{site.site_id}_{threshold}",
            "threshold": threshold,
            "pre_generated": False,
        },
    }


def esa_worldcover_tile_paths(region: RegionConfig, geom) -> list[Path]:
    paths = sorted(region.esa_worldcover_dir.glob("ESA_WorldCover_10m_2021_v200_*_Map.tif"))
    return intersecting_raster_paths(paths, geom)


def jrc_occurrence_tile_paths(region: RegionConfig, geom) -> list[Path]:
    paths = sorted(region.jrc_gsw_dir.glob("occurrence_*v1_4_2021.tif"))
    return intersecting_raster_paths(paths, geom)


def intersecting_raster_paths(paths: list[Path], geom) -> list[Path]:
    selected = []
    for path in paths:
        try:
            with rasterio.open(path) as src:
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        except Exception:
            continue
        if box(*bounds).intersects(geom):
            selected.append(path)
    return selected


def water_polygons_from_raster(path: Path, clip_geom, target_geom, water_mask_fn) -> list:
    with rasterio.open(path) as src:
        clipped, transform = mask(src, [mapping(clip_geom)], crop=True, filled=True)
    arr = clipped[0]
    water = water_mask_fn(arr)
    geoms = []
    for geom_json, value in shapes(water.astype("uint8"), mask=water, transform=transform):
        if int(value) != 1:
            continue
        poly = shape(geom_json)
        if poly.is_empty:
            continue
        if not poly.intersects(target_geom):
            continue
        geoms.append(poly)
    return geoms



def smooth_jrc_geometry(geom, lake_area_km2: float):
    if lake_area_km2 > 250:
        metric = transform_geom(geom, "EPSG:4326", "EPSG:3857")
        metric = make_valid(metric)
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        min_area_m2 = 100_000
        kept = [part for part in parts if part.area >= min_area_m2]
        if not kept:
            kept = parts
        simplified = flatten_polygons(make_valid(part.simplify(30, preserve_topology=True)) for part in kept)
        result = simplified[0] if len(simplified) == 1 else MultiPolygon(simplified)
        return transform_geom(result, "EPSG:3857", "EPSG:4326")
    return smooth_water_geometry(geom)


def smooth_esa_geometry(geom, lake_area_km2: float):
    if lake_area_km2 > 250:
        metric = transform_geom(geom, "EPSG:4326", "EPSG:3857")
        metric = make_valid(metric)
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        min_area_m2 = 100_000
        kept = [part for part in parts if part.area >= min_area_m2]
        if not kept:
            kept = parts
        simplified = flatten_polygons(make_valid(part.simplify(30, preserve_topology=True)) for part in kept)
        result = simplified[0] if len(simplified) == 1 else MultiPolygon(simplified)
        return transform_geom(result, "EPSG:3857", "EPSG:4326")
    return smooth_water_geometry(geom)


def flatten_polygons(geoms) -> list[Polygon]:
    polygons: list[Polygon] = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(part for part in geom.geoms if not part.is_empty)
    return polygons
