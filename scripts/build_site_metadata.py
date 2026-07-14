#!/usr/bin/env python3
"""Build observation-site metadata from local imagery directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom as rasterio_transform_geom
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lake_workbench.geo import transform_geom  # noqa: E402
from lake_workbench.regions.config import RegionConfig, load_region_configs  # noqa: E402
from lake_workbench.water.layers import (  # noqa: E402
    build_esa_smoothed_layer,
    build_jrc_occurrence_layer,
    write_esa_polygon_cache,
    write_jrc_polygon_cache,
)

from site_metadata_sources import (  # noqa: E402
    clean_text,
    display_path,
    image_date,
    image_tile_name,
    load_external_hydrolakes,
    load_external_osm_water,
    load_sentinel_tile_index,
)


REGIONS, DEFAULT_REGION_KEY = load_region_configs()
DATE_RE = re.compile(r"(19\d{2}|20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")


@dataclass
class BuildResult:
    sites: gpd.GeoDataFrame
    cores: gpd.GeoDataFrame
    imagery: gpd.GeoDataFrame
    labels: gpd.GeoDataFrame
    external: gpd.GeoDataFrame
    product_rows: list[dict]
    default_imagery: dict[str, str]
    esa_cache: list[tuple[str, dict]]
    jrc_cache: list[tuple[str, int, dict]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--footprint-size", type=int, default=256)
    parser.add_argument("--jrc-threshold", type=int, default=75)
    parser.add_argument("--skip-raster-water", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = REGIONS[args.region]
    output_dir = args.output_dir.resolve() if args.output_dir else region.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_site_metadata(
        region,
        footprint_size=max(64, min(1024, args.footprint_size)),
        jrc_threshold=max(1, min(100, args.jrc_threshold)),
        include_raster_water=not args.skip_raster_water,
    )
    write_result(region, output_dir, result)


def build_site_metadata(
    region: RegionConfig,
    footprint_size: int = 256,
    jrc_threshold: int = 75,
    include_raster_water: bool = True,
) -> BuildResult:
    if region.source_img_root is None or not region.source_img_root.exists():
        raise FileNotFoundError(f"local imagery root not found for {region.key}: {region.source_img_root}")
    sentinel_index = load_sentinel_tile_index(region.sentinel_tile_index_paths)
    site_dirs = sorted(path for path in region.source_img_root.iterdir() if path.is_dir())
    if not site_dirs:
        raise RuntimeError(f"no site directories under {region.source_img_root}")

    site_work: list[dict] = []
    imagery_rows: list[dict] = []
    product_rows: list[dict] = []
    default_imagery: dict[str, str] = {}
    for index, site_dir in enumerate(site_dirs, start=1):
        site_id = f"{region.key}_{site_dir.name}"
        paths = sorted(site_dir.glob("*.img"))
        if not paths:
            continue
        assets = [inspect_imagery_asset(site_id, path, sentinel_index, footprint_size) for path in paths]
        assets.sort(key=lambda item: (item["acquisition_date"], item["filename"]))
        default_asset = assets[-1]
        default_asset["is_default"] = 1
        union = polygonal_geometry(unary_union([item["geometry"] for item in assets]))
        core = majority_coverage_geometry([item["geometry"] for item in assets], union)
        tiles = sentinel_tiles_for_geometry(union, sentinel_index)
        if not tiles:
            tiles = sorted({item["mgrs_tile"] for item in assets if item["mgrs_tile"]})
        for asset in assets:
            imagery_rows.append(asset)
            product_rows.append(asset_product_row(site_id, site_dir, asset))
        if default_asset["mgrs_tile"]:
            default_imagery[f"{site_id}:{default_asset['mgrs_tile']}"] = default_asset["asset_id"]
        site_work.append(
            {
                "site_id": site_id,
                "local_directory_id": site_dir.name,
                "source_path": display_path(site_dir),
                "geometry": union,
                "core_geometry": core,
                "imagery": assets,
                "sentinel_tiles": tiles,
            }
        )
        print(f"[{region.key}] imagery {index}/{len(site_dirs)} {site_id}: {len(assets)} assets")

    labels, label_asset_counts, label_feature_counts = load_local_label_features(site_work)
    total_bounds = unary_union([item["geometry"] for item in site_work]).bounds
    osm = load_external_osm_water(region.osm_water, total_bounds)
    hydro = load_external_hydrolakes(region.hydrolakes, total_bounds)
    external_rows: list[dict] = []
    site_candidate_counts: dict[str, Counter] = {}
    suggested_names: dict[str, str] = {}
    for item in site_work:
        site_id = item["site_id"]
        candidates = build_vector_candidates(site_id, item["geometry"], osm, hydro)
        mark_suggested_candidate(candidates)
        external_rows.extend(candidates)
        site_candidate_counts[site_id] = Counter(row["source"] for row in candidates)
        suggested_names[site_id] = suggested_site_name(candidates)

    esa_cache: list[tuple[str, dict]] = []
    jrc_cache: list[tuple[str, int, dict]] = []
    if include_raster_water:
        for index, item in enumerate(site_work, start=1):
            site_id = item["site_id"]
            site_like = site_object(site_id, item["geometry"])
            esa = build_esa_smoothed_layer(region, site_like)
            if esa is not None:
                esa_cache.append((site_id, esa))
                candidate = raster_candidate(site_id, item["geometry"], "esa", esa)
                if candidate is not None:
                    external_rows.append(candidate)
                    site_candidate_counts[site_id]["esa"] += 1
            jrc = build_jrc_occurrence_layer(region, site_like, jrc_threshold)
            if jrc is not None:
                jrc_cache.append((site_id, jrc_threshold, jrc))
                candidate = raster_candidate(site_id, item["geometry"], "jrc", jrc, threshold=jrc_threshold)
                if candidate is not None:
                    external_rows.append(candidate)
                    site_candidate_counts[site_id]["jrc"] += 1
            print(f"[{region.key}] raster water {index}/{len(site_work)} {site_id}")

    site_rows = []
    core_rows = []
    for item in site_work:
        site_id = item["site_id"]
        assets = item["imagery"]
        dates = [asset["acquisition_date"] for asset in assets if asset["acquisition_date"]]
        bounds = item["geometry"].bounds
        suggested = suggested_names.get(site_id, "")
        display_name = site_display_name(item["local_directory_id"], suggested)
        counts = site_candidate_counts.get(site_id, Counter())
        site_rows.append(
            {
                "site_id": site_id,
                "region": region.key,
                "local_directory_id": item["local_directory_id"],
                "display_name": display_name,
                "suggested_name": suggested or None,
                "identity_source": "local_imagery",
                "source_path": item["source_path"],
                "image_count": len(assets),
                "label_asset_count": label_asset_counts.get(site_id, 0),
                "label_feature_count": label_feature_counts.get(site_id, 0),
                "external_feature_count": sum(counts.values()),
                "osm_feature_count": counts.get("osm", 0),
                "hydrolakes_feature_count": counts.get("hydrolakes", 0),
                "esa_feature_count": counts.get("esa", 0),
                "jrc_feature_count": counts.get("jrc", 0),
                "first_acquisition_date": min(dates) if dates else None,
                "last_acquisition_date": max(dates) if dates else None,
                "sentinel_tiles": ",".join(item["sentinel_tiles"]),
                "coverage_area_km2": geometry_area_km2(item["geometry"]),
                "core_area_km2": geometry_area_km2(item["core_geometry"]),
                "core_coverage_fraction": 0.8,
                "bbox_west": bounds[0],
                "bbox_south": bounds[1],
                "bbox_east": bounds[2],
                "bbox_north": bounds[3],
                "center_lon": (bounds[0] + bounds[2]) / 2,
                "center_lat": (bounds[1] + bounds[3]) / 2,
                "geometry": item["geometry"],
            }
        )
        core_rows.append(
            {
                "site_id": site_id,
                "core_area_km2": geometry_area_km2(item["core_geometry"]),
                "geometry": item["core_geometry"],
            }
        )

    return BuildResult(
        sites=geo_frame(site_rows),
        cores=geo_frame(core_rows),
        imagery=geo_frame(imagery_rows),
        labels=geo_frame(labels),
        external=geo_frame(external_rows),
        product_rows=product_rows,
        default_imagery=default_imagery,
        esa_cache=esa_cache,
        jrc_cache=jrc_cache,
    )


def inspect_imagery_asset(site_id: str, path: Path, sentinel_index, footprint_size: int) -> dict:
    with rasterio.open(path) as src:
        geometry, valid_ratio = valid_footprint(src, footprint_size)
        tile = image_tile_name(path.name, geometry, sentinel_index)
        resolution_x = abs(float(src.transform.a))
        resolution_y = abs(float(src.transform.e))
        return {
            "asset_id": f"{site_id}_{path.stem}",
            "site_id": site_id,
            "filename": path.name,
            "product_name": path.stem,
            "acquisition_date": image_date(path.name) or None,
            "mgrs_tile": tile or None,
            "path": display_path(path),
            "width": int(src.width),
            "height": int(src.height),
            "band_count": int(src.count),
            "crs": str(src.crs or ""),
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "nodata": float(src.nodata) if src.nodata is not None else None,
            "valid_ratio": valid_ratio,
            "is_default": 0,
            "geometry": geometry,
        }


def valid_footprint(src, max_size: int) -> tuple[Any, float]:
    scale = max(src.width / max_size, src.height / max_size, 1.0)
    width = max(1, int(round(src.width / scale)))
    height = max(1, int(round(src.height / scale)))
    count = min(src.count, 3)
    data = src.read(
        list(range(1, count + 1)),
        out_shape=(count, height, width),
        resampling=Resampling.nearest,
    )
    valid = np.any(data != 0, axis=0)
    if src.nodata is not None:
        valid &= np.any(data != src.nodata, axis=0)
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
    candidates = sentinel_index[sentinel_index.geometry.intersects(geometry)].copy()
    if candidates.empty:
        return []
    site_m = transform_geom(geometry, "EPSG:4326", "EPSG:3857")
    candidates["overlap"] = [
        transform_geom(tile, "EPSG:4326", "EPSG:3857").intersection(site_m).area
        for tile in candidates.geometry
    ]
    candidates = candidates[candidates["overlap"] > 0].sort_values(["overlap", "Name"], ascending=[False, True])
    return candidates["Name"].astype(str).tolist()


def majority_coverage_geometry(geometries: list[Any], union, fraction: float = 0.8, size: int = 256):
    if len(geometries) == 1:
        return geometries[0]
    bounds = union.bounds
    width = max(bounds[2] - bounds[0], 1e-9)
    height = max(bounds[3] - bounds[1], 1e-9)
    if width >= height:
        columns = size
        rows = max(1, int(round(size * height / width)))
    else:
        rows = size
        columns = max(1, int(round(size * width / height)))
    transform = from_bounds(*bounds, columns, rows)
    counts = np.zeros((rows, columns), dtype="uint16")
    for geometry in geometries:
        counts += rasterize(
            [(mapping(geometry), 1)],
            out_shape=(rows, columns),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    threshold = max(1, int(np.ceil(len(geometries) * fraction)))
    core_mask = counts >= threshold
    parts = [
        shape(geometry)
        for geometry, value in shapes(core_mask.astype("uint8"), mask=core_mask, transform=transform)
        if int(value) == 1
    ]
    return polygonal_geometry(unary_union(parts).intersection(union)) if parts else union


def load_local_label_features(site_work: list[dict]) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    rows: list[dict] = []
    asset_counts: dict[str, int] = {}
    feature_counts: dict[str, int] = {}
    for site in site_work:
        site_id = site["site_id"]
        directory = PROJECT_ROOT / site["source_path"]
        paths = sorted(directory.glob("*.shp"))
        asset_counts[site_id] = len(paths)
        for path in paths:
            label_path = display_path(path)
            label_asset_id = hashlib.sha1(label_path.encode("utf-8")).hexdigest()[:16]
            date = date_from_text(path.stem)
            try:
                frame = pyogrio.read_dataframe(path)
            except Exception as exc:  # noqa: BLE001 - preserve the rest of a batch.
                print(f"warning: failed local label {path}: {exc}")
                continue
            if frame.crs is not None:
                frame = frame.to_crs("EPSG:4326")
            for feature_index, row in frame.iterrows():
                geometry = polygonal_geometry(row.geometry)
                if geometry.is_empty:
                    continue
                properties = {
                    key: json_value(row.get(key))
                    for key in frame.columns
                    if key != "geometry"
                }
                rows.append(
                    {
                        "label_feature_id": f"{label_asset_id}_{feature_index}",
                        "label_asset_id": label_asset_id,
                        "site_id": site_id,
                        "source": "local_shapefile",
                        "source_path": label_path,
                        "source_filename": path.name,
                        "acquisition_date": date or None,
                        "source_feature_id": str(feature_index),
                        "area_km2": geometry_area_km2(geometry),
                        "properties_json": json.dumps(properties, ensure_ascii=False, sort_keys=True),
                        "geometry": geometry,
                    }
                )
        feature_counts[site_id] = sum(row["site_id"] == site_id for row in rows)
    return rows, asset_counts, feature_counts


def build_vector_candidates(site_id: str, site_geometry, osm, hydro) -> list[dict]:
    candidates: list[dict] = []
    for source, frame in (("osm", osm), ("hydrolakes", hydro)):
        if frame.empty:
            continue
        selected = frame[frame.geometry.intersects(site_geometry)]
        for row in selected.itertuples():
            geometry = polygonal_geometry(row.geometry)
            overlap = polygonal_geometry(geometry.intersection(site_geometry))
            if overlap.is_empty:
                continue
            values = row._asdict()
            source_id = clean_text(values.get("source_feature_id")) or str(row.Index)
            name = candidate_name(values)
            feature_area = geometry_area_km2(geometry)
            overlap_area = geometry_area_km2(overlap)
            site_area = geometry_area_km2(site_geometry)
            properties = {
                key: json_value(value)
                for key, value in values.items()
                if key not in {"Index", "geometry"}
            }
            candidates.append(
                {
                    "candidate_id": f"{site_id}_{source}_{source_id}",
                    "site_id": site_id,
                    "source": source,
                    "source_feature_id": source_id,
                    "name": name or None,
                    "water_type": clean_text(values.get("water_type")),
                    "area_km2": feature_area,
                    "intersection_area_km2": overlap_area,
                    "site_coverage_ratio": overlap_area / site_area if site_area else 0.0,
                    "feature_coverage_ratio": overlap_area / feature_area if feature_area else 0.0,
                    "is_suggested_primary": 0,
                    "properties_json": json.dumps(properties, ensure_ascii=False, sort_keys=True),
                    "geometry": geometry,
                }
            )
    return candidates


def mark_suggested_candidate(candidates: list[dict]) -> None:
    if not candidates:
        return
    candidates.sort(
        key=lambda row: (
            row["intersection_area_km2"],
            row["feature_coverage_ratio"],
            1 if row["source"] == "osm" else 0,
        ),
        reverse=True,
    )
    candidates[0]["is_suggested_primary"] = 1


def suggested_site_name(candidates: list[dict]) -> str:
    selected = next((row for row in candidates if row.get("is_suggested_primary") and row.get("name")), None)
    if not selected:
        return ""
    # A large imagery footprint often touches unrelated named ponds. Only expose
    # a name hint when the overlap is spatially meaningful for this site.
    if selected["intersection_area_km2"] < 0.1 or selected["site_coverage_ratio"] < 0.001:
        return ""
    return str(selected["name"])


def site_display_name(local_directory_id: str, suggested_name: str = "") -> str:
    base = f"区域 {local_directory_id}"
    return f"{base}（{suggested_name}附近）" if suggested_name else base


def candidate_name(values: dict) -> str:
    for key in ("name_zh", "name", "name_en", "display_name"):
        value = clean_text(values.get(key))
        if value and not value.isdigit() and value != clean_text(values.get("source_feature_id")):
            return value
    return ""


def site_object(site_id: str, geometry) -> Any:
    bounds = geometry.bounds
    return SimpleNamespace(
        site_id=site_id,
        geometry=geometry,
        bbox=bounds,
        area_km2=geometry_area_km2(geometry),
    )


def raster_candidate(site_id: str, site_geometry, source: str, layer: dict, threshold: int | None = None) -> dict | None:
    geometry_json = layer.get("geometry")
    if not geometry_json:
        return None
    geometry = polygonal_geometry(shape(geometry_json))
    if geometry.is_empty:
        return None
    overlap = polygonal_geometry(geometry.intersection(site_geometry))
    if overlap.is_empty:
        return None
    properties = dict(layer.get("properties") or {})
    source_id = f"{source}_{site_id}" + (f"_{threshold}" if threshold is not None else "")
    area = geometry_area_km2(geometry)
    overlap_area = geometry_area_km2(overlap)
    site_area = geometry_area_km2(site_geometry)
    return {
        "candidate_id": source_id,
        "site_id": site_id,
        "source": source,
        "source_feature_id": source_id,
        "name": None,
        "water_type": "water",
        "area_km2": area,
        "intersection_area_km2": overlap_area,
        "site_coverage_ratio": overlap_area / site_area if site_area else 0.0,
        "feature_coverage_ratio": overlap_area / area if area else 0.0,
        "is_suggested_primary": 0,
        "properties_json": json.dumps(properties, ensure_ascii=False, sort_keys=True),
        "geometry": geometry,
    }


def asset_product_row(site_id: str, site_dir: Path, asset: dict) -> dict:
    return {
        "site_id": site_id,
        "product_id": asset["asset_id"],
        "product_name": asset["asset_id"],
        "tile": asset["mgrs_tile"] or "",
        "date": asset["acquisition_date"] or "",
        "cloud_cover": "",
        "product_type": "MSIL1C_IMG",
        "source": "local_img",
        "safe_path": display_path(site_dir),
        "tci_path": asset["path"],
        "download_status": "downloaded",
        "downloaded_at": "",
        "valid_ratio": asset["valid_ratio"],
    }


def write_result(region: RegionConfig, output_dir: Path, result: BuildResult) -> None:
    gpkg_path = output_dir / "site_metadata.gpkg"
    csv_path = output_dir / "site_metadata.csv"
    tmp_path = output_dir / ".site_metadata.building.gpkg"
    if tmp_path.exists():
        tmp_path.unlink()
    layers = [
        ("sites", result.sites),
        ("site_coverage_core", result.cores),
        ("imagery_assets", result.imagery),
        ("local_label_features", result.labels),
        ("external_water_features", result.external),
    ]
    for layer_name, frame in layers:
        pyogrio.write_dataframe(frame, tmp_path, layer=layer_name, driver="GPKG", promote_to_multi=True)
    validate_written_layers(tmp_path, layers)
    tmp_path.replace(gpkg_path)
    result.sites.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    products_path = output_dir / "sentinel_products.csv"
    generated = pd.DataFrame(result.product_rows)
    if products_path.exists():
        existing = pd.read_csv(products_path)
        preserved = existing[existing.get("source", "").fillna("") != "local_img"].copy()
        if not preserved.empty:
            generated = pd.concat([generated, preserved], ignore_index=True, sort=False)
    generated.to_csv(products_path, index=False)

    active_path = output_dir / "active_imagery.json"
    existing_active = {}
    if active_path.exists():
        try:
            payload = json.loads(active_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing_active = {str(key): str(value) for key, value in payload.items()}
        except json.JSONDecodeError:
            pass
    active_path.write_text(
        json.dumps({**result.default_imagery, **existing_active}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for site_id, layer in result.esa_cache:
        write_esa_polygon_cache(region, site_id, layer)
    for site_id, threshold, layer in result.jrc_cache:
        write_jrc_polygon_cache(region, site_id, threshold, layer)
    normalize_site_identity_files(output_dir, result.sites)

    print(f"wrote {gpkg_path}")
    for layer_name, frame in layers:
        print(f"  {layer_name}: {len(frame)}")
    print(f"wrote {csv_path}")
    print(f"wrote {products_path}")
    print(f"wrote {active_path}")


def normalize_site_identity_files(output_dir: Path, sites: gpd.GeoDataFrame) -> None:
    names = dict(zip(sites["site_id"], sites["display_name"]))
    paths = [output_dir / "training_samples.csv"]
    paths.extend((output_dir / "training_patches").glob("*/manifest.csv"))
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "site_id" not in frame:
            raise ValueError(f"missing site_id in {path}")
        if "site_name" not in frame:
            frame["site_name"] = frame["site_id"].map(names).fillna("")
        else:
            generated_names = frame["site_id"].map(names).fillna("")
            frame["site_name"] = frame["site_name"].where(frame["site_name"] != "", generated_names)
        frame.to_csv(path, index=False)


def validate_written_layers(path: Path, layers: list[tuple[str, gpd.GeoDataFrame]]) -> None:
    actual = {name for name, _geometry_type in pyogrio.list_layers(path)}
    expected = {name for name, _frame in layers}
    if actual != expected:
        raise RuntimeError(f"site metadata layers differ: expected={expected}, actual={actual}")
    for layer_name, frame in layers:
        info = pyogrio.read_info(path, layer=layer_name)
        if info["features"] != len(frame):
            raise RuntimeError(f"wrong feature count for {layer_name}: {info['features']} != {len(frame)}")


def geo_frame(rows: list[dict]) -> gpd.GeoDataFrame:
    if not rows:
        raise RuntimeError("metadata layer unexpectedly has no features")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def polygonal_geometry(geometry) -> Any:
    if geometry is None or geometry.is_empty:
        return MultiPolygon()
    geometry = make_valid(geometry)
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, MultiPolygon):
                polygons.extend(part.geoms)
        return MultiPolygon([part for part in polygons if not part.is_empty])
    return MultiPolygon()


def geometry_area_km2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(transform_geom(geometry, "EPSG:4326", "EPSG:3857").area / 1_000_000)


def date_from_text(value: str) -> str:
    match = DATE_RE.search(value)
    return "-".join(match.groups()) if match else ""


def json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
