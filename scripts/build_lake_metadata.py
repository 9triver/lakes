#!/usr/bin/env python3
"""Build a unified lake metadata dataset for one supported region."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lakes_browser.region_config import RegionConfig, load_region_configs  # noqa: E402

TAG_RE = re.compile(r'"([^"]+)"=>"([^"]*)"')
REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--data-dir", type=Path, default=None, help="override configured raw data directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="override configured processed directory")
    parser.add_argument("--min-area-km2", type=float, default=0.01)
    parser.add_argument("--source-img-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = REGIONS[args.region]
    data_dir = args.data_dir.resolve() if args.data_dir else region.data_dir
    output_dir = args.output_dir.resolve() if args.output_dir else region.processed_dir
    if args.data_dir or args.output_dir:
        region = replace(region, data_dir=data_dir, processed_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = InputPaths(region)
    print(f"data_dir={data_dir}")
    print(f"region={args.region}")
    if region.metadata_source == "img_bounds":
        sentinel_tile_index = load_sentinel_tile_index(paths.sentinel_tile_index_paths)
        print("building lakes from image bounds")
        source_img_root = args.source_img_root.resolve() if args.source_img_root else region.source_img_root
        if source_img_root is None:
            raise ValueError(f"Region {region.key} requires source_img_root for img_bounds metadata")
        lakes, image_rows, active_imagery = load_img_bound_lakes(
            source_img_root,
            paths,
            args.min_area_km2,
            args.region,
            region.name,
            region.uid_prefix,
            sentinel_tile_index,
        )
        print(f"img_bound_lakes={len(lakes)}")
    else:
        print("loading OSM water polygons")
        lakes = load_osm_lakes(paths.osm_water, args.min_area_km2, args.region, region.name, region.uid_prefix)
        image_rows = []
        active_imagery = {}
        print(f"osm_lakes={len(lakes)}")
        sentinel_tile_index = load_sentinel_tile_index(paths.sentinel_tile_index_paths)

    if region.metadata_source == "img_bounds":
        print("keeping Sentinel-2 grid tiles from local image bounds")
    elif sentinel_tile_index is None:
        print("warning: Sentinel-2 tile grid missing; sentinel_tiles will be empty")
        lakes = attach_empty_sentinel_metadata(lakes)
    else:
        print(f"sentinel_grid_tiles={len(sentinel_tile_index)}")
        print("matching Sentinel-2 grid tiles")
        lakes = attach_sentinel_tile_metadata(lakes, sentinel_tile_index)

    if paths.tci_index.exists():
        print("loading optional Sentinel-2 TCI index")
        tci_index, tile_footprints = load_tci_index(paths.tci_index, data_dir)
        print(f"tci_tiles={len(tci_index)}")
        print("matching downloaded Sentinel-2 TCI")
        lakes = attach_tci_metadata(lakes, tci_index, tile_footprints)
    else:
        print("optional Sentinel-2 TCI index missing; downloaded imagery metadata will be empty")

    print("matching HydroLAKES")
    lakes = attach_hydrolakes_metadata(lakes, paths.hydrolakes)

    lakes = finalize_columns(lakes)
    gpkg_path = region.lake_metadata
    csv_path = region.lake_metadata_csv

    if gpkg_path.exists():
        gpkg_path.unlink()
    pyogrio.write_dataframe(lakes, gpkg_path, layer="lake_metadata", driver="GPKG")
    lakes.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    if image_rows:
        products_path = region.user_sentinel_index
        active_path = region.active_imagery
        products_path.parent.mkdir(parents=True, exist_ok=True)
        products = merge_generated_product_rows(products_path, image_rows)
        products.to_csv(products_path, index=False)
        active_payload = merge_active_imagery(active_path, active_imagery)
        active_path.write_text(json.dumps(active_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {products_path}")
        print(f"wrote {active_path}")

    print(f"wrote {gpkg_path}")
    print(f"wrote {csv_path}")
    print_summary(lakes)


class InputPaths:
    def __init__(self, region: RegionConfig) -> None:
        self.osm_water = region.osm_water
        self.tci_index = region.tci_index
        self.sentinel_tile_index_paths = region.sentinel_tile_index_paths
        self.hydrolakes = region.hydrolakes


def merge_generated_product_rows(products_path: Path, generated_rows: list[dict]) -> pd.DataFrame:
    generated = pd.DataFrame(generated_rows)
    if not products_path.exists():
        return generated
    existing = pd.read_csv(products_path)
    if "source" not in existing.columns:
        return generated
    preserved = existing[existing["source"].fillna("") != "local_img"].copy()
    if preserved.empty:
        return generated
    return pd.concat([generated, preserved], ignore_index=True, sort=False)


def merge_active_imagery(active_path: Path, generated_active: dict[str, str]) -> dict[str, str]:
    if not active_path.exists():
        return generated_active
    try:
        existing = json.loads(active_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return generated_active
    if not isinstance(existing, dict):
        return generated_active
    return {**generated_active, **{str(key): str(value) for key, value in existing.items()}}


def load_img_bound_lakes(
    source_root: Path,
    paths: InputPaths,
    min_area_km2: float,
    region_key: str,
    province_name: str,
    uid_prefix: str,
    sentinel_tile_index: gpd.GeoDataFrame | None,
) -> tuple[gpd.GeoDataFrame, list[dict], dict[str, str]]:
    if not source_root.exists():
        raise FileNotFoundError(f"source image root not found: {source_root}")
    image_sets = []
    for lake_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        img_paths = sorted(lake_dir.glob("*.img"))
        if not img_paths:
            continue
        image_records = read_image_records(lake_dir.name, img_paths, sentinel_tile_index)
        image_sets.append((lake_dir, image_records))
    if not image_sets:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326"), [], {}
    total_search_geom = unary_union([record["geometry"] for _, records in image_sets for record in records])
    external_bbox = total_search_geom.bounds
    osm_water = load_external_osm_water(paths.osm_water, external_bbox)
    hydro_water = load_external_hydrolakes(paths.hydrolakes, external_bbox)
    records = []
    product_rows = []
    active_imagery = {}
    for lake_dir, image_records in image_sets:
        search_geom = unary_union([record["geometry"] for record in image_records])
        match = match_largest_external_water(search_geom, osm_water, hydro_water, min_area_km2)
        geom = match["geometry"] if match else search_geom
        geom = geom if geom.is_valid else geom.buffer(0)
        if geom.is_empty:
            continue
        bounds = geom.bounds
        lake_uid = f"{uid_prefix}_{lake_dir.name}"
        area_km2 = geometry_area_km2(geom)
        display_name = match.get("display_name") if match else None
        name = match.get("name") if match else None
        source_primary = match.get("source") if match else "img_bounds"
        source_feature_id = match.get("source_feature_id") if match else lake_dir.name
        water_type = match.get("water_type") if match else "unknown"
        best_image = max(
            image_records,
            key=lambda item: (item.get("date") or "", item["path"].name),
        )
        tiles = sorted({record["tile"] for record in image_records if record.get("tile")})
        for record in image_records:
            product_name = f"{lake_uid}_{record['path'].stem}"
            row = {
                "lake_id": lake_uid,
                "product_id": product_name,
                "product_name": product_name,
                "tile": record.get("tile") or "",
                "date": record.get("date") or "",
                "cloud_cover": "",
                "product_type": "MSIL1C_IMG",
                "source": "local_img",
                "safe_path": display_path(lake_dir),
                "tci_path": display_path(record["path"]),
                "download_status": "downloaded",
                "downloaded_at": "",
                "valid_ratio": record.get("valid_ratio", 1.0),
            }
            product_rows.append(row)
            if record["path"] == best_image["path"] and row["tile"]:
                active_imagery[f"{lake_uid}:{row['tile']}"] = product_name
        records.append(
            {
                "lake_uid": lake_uid,
                "display_name": display_name or lake_uid,
                "name": name,
                "name_zh": name,
                "name_en": None,
                "source_primary": source_primary,
                "source_feature_id": str(source_feature_id),
                "osm_id_text": match.get("osm_id_text") if match else None,
                "osm_way_id_text": match.get("osm_way_id_text") if match else None,
                "wikidata": match.get("wikidata") if match else None,
                "wikipedia": match.get("wikipedia") if match else None,
                "osm_code": match.get("osm_code") if match else None,
                "nsdi_code": None,
                "zhb_code": None,
                "water_tag": match.get("water_tag") if match else None,
                "water_type": water_type,
                "area_km2": area_km2,
                "bbox_west": bounds[0],
                "bbox_south": bounds[1],
                "bbox_east": bounds[2],
                "bbox_north": bounds[3],
                "center_lon": (bounds[0] + bounds[2]) / 2,
                "center_lat": (bounds[1] + bounds[3]) / 2,
                "province": province_name,
                "region": region_key,
                "city": None,
                "county": None,
                "admin_source": None,
                "sentinel_tiles": ",".join(tiles),
                "best_tci_tile": best_image.get("tile") or (tiles[0] if tiles else None),
                "best_tci_date": best_image.get("date"),
                "best_tci_product": f"{lake_uid}_{best_image['path'].stem}",
                "best_tci_valid_ratio": best_image.get("valid_ratio", 1.0),
                "best_tci_path": display_path(best_image["path"]),
                "has_tci": 1,
                "has_osm_polygon": 1 if match and match.get("source") == "osm" else 0,
                "has_hydrolakes_polygon": 1 if match and match.get("source") == "hydrolakes" else 0,
                "has_esa_polygon": 0,
                "natural": match.get("natural") if match else None,
                "landuse": match.get("landuse") if match else None,
                "man_made": match.get("man_made") if match else None,
                "type": match.get("type") if match else None,
                "other_tags": match.get("other_tags") if match else "",
                "geometry": geom,
            }
        )
    if not records:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326"), product_rows, active_imagery
    lakes = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return lakes.sort_values("area_km2", ascending=False).reset_index(drop=True), product_rows, active_imagery


def read_image_records(lake_dir_name: str, img_paths: list[Path], sentinel_tile_index: gpd.GeoDataFrame | None) -> list[dict]:
    records = []
    for path in img_paths:
        with rasterio.open(path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            geom = box(*bounds)
            tile = image_tile_name(path.name, geom, sentinel_tile_index)
            records.append(
                {
                    "lake_dir": lake_dir_name,
                    "path": path,
                    "date": image_date(path.name),
                    "tile": tile,
                    "geometry": geom,
                    "valid_ratio": image_valid_ratio(path),
                }
            )
    return records


def image_tile_name(name: str, geom, sentinel_tile_index: gpd.GeoDataFrame | None) -> str:
    match = re.search(r"_T([0-9A-Z]{5})_", name)
    if match:
        return match.group(1)
    if sentinel_tile_index is None or sentinel_tile_index.empty:
        return ""
    candidates = sentinel_tile_index[sentinel_tile_index.geometry.intersects(geom)].copy()
    if candidates.empty:
        return ""
    candidates["overlap"] = [geometry_area_km2(tile_geom.intersection(geom)) for tile_geom in candidates.geometry]
    candidates = candidates[candidates["overlap"] > 0].sort_values(["overlap", "Name"], ascending=[False, True])
    return str(candidates.iloc[0]["Name"]) if not candidates.empty else ""


def image_date(name: str) -> str:
    match = re.search(r"MSIL\d[AC]?_(\d{8})", name)
    if not match:
        return ""
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def image_valid_ratio(path: Path) -> float:
    import numpy as np
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        scale = max(src.width / 1024, src.height / 1024, 1)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        indexes = list(range(1, min(src.count, 3) + 1))
        data = src.read(indexes, out_shape=(len(indexes), out_height, out_width), resampling=Resampling.nearest)
    valid = np.any(data != 0, axis=0)
    return float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0


def load_external_osm_water(osm_path: Path, bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    if not osm_path.exists():
        return empty_external_water()
    columns = [
        "osm_id",
        "osm_way_id",
        "name",
        "type",
        "natural",
        "landuse",
        "man_made",
        "other_tags",
    ]
    data = pyogrio.read_dataframe(osm_path, layer="osm_water_polygons", columns=columns, bbox=bbox).to_crs("EPSG:4326")
    data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
    if data.empty:
        return empty_external_water()
    data["other_tags"] = data["other_tags"].fillna("")
    data = data[
        ~data["other_tags"].str.contains('"water"=>"river"', regex=False)
        & ~data["other_tags"].str.contains('"water"=>"canal"', regex=False)
    ].copy()
    if data.empty:
        return empty_external_water()
    tag_records = data["other_tags"].map(parse_other_tags)
    data["source"] = "osm"
    data["area_km2"] = [geometry_area_km2(geom) for geom in data.geometry]
    data["osm_id_text"] = data["osm_id"].map(clean_id)
    data["osm_way_id_text"] = data["osm_way_id"].map(clean_id)
    data["source_feature_id"] = [
        first_present(row.osm_id_text, row.osm_way_id_text, f"osm_{i + 1}")
        for i, row in data.reset_index(drop=True).iterrows()
    ]
    data["name"] = data["name"].map(clean_text)
    data["name_zh"] = [clean_text(tags.get("name:zh")) or row.name for tags, row in zip(tag_records, data.itertuples())]
    data["name_en"] = [clean_text(tags.get("name:en")) for tags in tag_records]
    data["display_name"] = [
        first_present(row.name_zh, row.name, row.name_en, row.source_feature_id)
        for row in data.itertuples()
    ]
    data["water_tag"] = [clean_text(tags.get("water")) for tags in tag_records]
    data["wikidata"] = [clean_text(tags.get("wikidata")) for tags in tag_records]
    data["wikipedia"] = [clean_text(tags.get("wikipedia")) for tags in tag_records]
    data["osm_code"] = [clean_text(tags.get("code")) for tags in tag_records]
    data["water_type"] = [infer_water_type(row, tags) for tags, row in zip(tag_records, data.itertuples())]
    return data


def load_external_hydrolakes(hydrolakes_path: Path, bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    if not hydrolakes_path.exists():
        return empty_external_water()
    columns = ["Hylak_id", "Lake_name", "Country", "Lake_area"]
    data = pyogrio.read_dataframe(hydrolakes_path, columns=columns, bbox=bbox).to_crs("EPSG:4326")
    data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
    if data.empty:
        return empty_external_water()
    data["source"] = "hydrolakes"
    data["source_feature_id"] = data["Hylak_id"].map(clean_id)
    data["display_name"] = data["Lake_name"].map(clean_text)
    data["name"] = data["display_name"]
    data["name_zh"] = None
    data["name_en"] = data["display_name"]
    data["water_type"] = "lake"
    data["osm_id_text"] = None
    data["osm_way_id_text"] = None
    data["wikidata"] = None
    data["wikipedia"] = None
    data["osm_code"] = None
    data["water_tag"] = "lake"
    data["natural"] = "water"
    data["landuse"] = None
    data["man_made"] = None
    data["type"] = None
    data["other_tags"] = ""
    return data


def empty_external_water() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")


def match_largest_external_water(search_geom, osm_water: gpd.GeoDataFrame, hydro_water: gpd.GeoDataFrame, min_area_km2: float) -> dict | None:
    osm_match = match_largest_water_from_source(search_geom, osm_water, 0.0)
    if osm_match is not None:
        return osm_match
    return match_largest_water_from_source(search_geom, hydro_water, min_area_km2)


def match_largest_water_from_source(search_geom, data: gpd.GeoDataFrame, min_overlap_km2: float) -> dict | None:
    matches = []
    if data.empty:
        return None
    candidates = data[data.geometry.intersects(search_geom)].copy()
    if candidates.empty:
        return None
    for row in candidates.itertuples():
        overlap_geom = row.geometry.intersection(search_geom)
        overlap_km2 = geometry_area_km2(overlap_geom)
        if overlap_km2 <= min_overlap_km2:
            continue
        total_km2 = geometry_area_km2(row.geometry)
        matches.append(
            {
                **{key: getattr(row, key) for key in row._fields if key not in {"Index", "geometry"}},
                "geometry": row.geometry,
                "overlap_km2": overlap_km2,
                "area_km2": total_km2,
            }
        )
    if not matches:
        return None
    matches.sort(key=lambda item: (item["overlap_km2"], item["area_km2"]), reverse=True)
    return matches[0]


def geometry_area_km2(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    gdf = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857")
    return float(gdf.area.iloc[0] / 1_000_000)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_osm_lakes(
    osm_path: Path,
    min_area_km2: float,
    region_key: str,
    province_name: str,
    uid_prefix: str,
) -> gpd.GeoDataFrame:
    columns = [
        "osm_id",
        "osm_way_id",
        "name",
        "type",
        "natural",
        "landuse",
        "man_made",
        "other_tags",
    ]
    lakes = pyogrio.read_dataframe(osm_path, layer="osm_water_polygons", columns=columns).to_crs("EPSG:4326")
    lakes = lakes[~lakes.geometry.is_empty & lakes.geometry.notna()].copy()
    lakes["other_tags"] = lakes["other_tags"].fillna("")
    lakes = lakes[
        ~lakes["other_tags"].str.contains('"water"=>"river"', regex=False)
        & ~lakes["other_tags"].str.contains('"water"=>"canal"', regex=False)
    ].copy()

    metric = lakes.to_crs("EPSG:3857")
    lakes["area_km2"] = metric.geometry.area / 1_000_000
    lakes = lakes[lakes["area_km2"] >= min_area_km2].copy()
    lakes = lakes.sort_values("area_km2", ascending=False).reset_index(drop=True)

    tag_records = lakes["other_tags"].map(parse_other_tags)
    lakes["osm_id_text"] = lakes["osm_id"].map(clean_id)
    lakes["osm_way_id_text"] = lakes["osm_way_id"].map(clean_id)
    lakes["source_feature_id"] = [
        first_present(row.osm_id_text, row.osm_way_id_text, f"feature_{i + 1}")
        for i, row in lakes.reset_index(drop=True).iterrows()
    ]
    lakes["lake_uid"] = [
        make_lake_uid(uid_prefix, row.osm_id_text, row.osm_way_id_text, row.source_feature_id)
        for _, row in lakes.iterrows()
    ]
    lakes["source_primary"] = "osm"
    lakes["name"] = lakes["name"].map(clean_text)
    lakes["name_zh"] = [clean_text(tags.get("name:zh")) or row.name for tags, row in zip(tag_records, lakes.itertuples())]
    lakes["name_en"] = [clean_text(tags.get("name:en")) for tags in tag_records]
    lakes["display_name"] = [
        first_present(row.name_zh, row.name, row.name_en, row.lake_uid)
        for row in lakes.itertuples()
    ]
    lakes["water_tag"] = [clean_text(tags.get("water")) for tags in tag_records]
    lakes["wikidata"] = [clean_text(tags.get("wikidata")) for tags in tag_records]
    lakes["wikipedia"] = [clean_text(tags.get("wikipedia")) for tags in tag_records]
    lakes["osm_code"] = [clean_text(tags.get("code")) for tags in tag_records]
    lakes["nsdi_code"] = [clean_text(tags.get("nsdi_code")) for tags in tag_records]
    lakes["zhb_code"] = [clean_text(tags.get("zhb_code")) for tags in tag_records]
    lakes["water_type"] = [infer_water_type(row, tags) for tags, row in zip(tag_records, lakes.itertuples())]

    bounds = lakes.geometry.bounds
    lakes["bbox_west"] = bounds["minx"]
    lakes["bbox_south"] = bounds["miny"]
    lakes["bbox_east"] = bounds["maxx"]
    lakes["bbox_north"] = bounds["maxy"]
    lakes["center_lon"] = (lakes["bbox_west"] + lakes["bbox_east"]) / 2
    lakes["center_lat"] = (lakes["bbox_south"] + lakes["bbox_north"]) / 2

    lakes["province"] = province_name
    lakes["region"] = region_key
    lakes["city"] = None
    lakes["county"] = None
    lakes["admin_source"] = None
    lakes["has_osm_polygon"] = 1
    lakes["has_hydrolakes_polygon"] = 0
    lakes["has_esa_polygon"] = 0
    return lakes


def load_tci_index(index_path: Path, data_dir: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    tci = pd.read_csv(index_path)
    records = []
    for row in tci.to_dict("records"):
        path = resolve_tci_path(row["tci_path"], data_dir)
        if not path.exists():
            continue
        try:
            with rasterio.open(path) as src:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                xs = [src.bounds.left, src.bounds.right, src.bounds.right, src.bounds.left]
                ys = [src.bounds.bottom, src.bounds.bottom, src.bounds.top, src.bounds.top]
                lons, lats = transformer.transform(xs, ys)
                footprint = box(min(lons), min(lats), max(lons), max(lats))
        except Exception as exc:  # noqa: BLE001 - keep batch generation moving.
            print(f"warning: failed to read TCI footprint for {path}: {exc}")
            continue
        records.append(
            {
                "tile": str(row["tile"]).upper(),
                "tci_date": str(row["date"]),
                "tci_source": clean_text(row.get("source")),
                "tci_valid_ratio": float(row.get("valid_ratio", 0) or 0),
                "tci_product": clean_text(row.get("product")),
                "tci_path": str(path),
                "geometry": footprint,
            }
        )
    footprints = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if not records:
        return pd.DataFrame(columns=["tile", "tci_date", "tci_source", "tci_valid_ratio", "tci_product", "tci_path"]), footprints
    return pd.DataFrame(records).drop(columns=["geometry"]), footprints


def load_sentinel_tile_index(paths: list[Path]) -> gpd.GeoDataFrame | None:
    index_path = next((path for path in paths if path.exists()), None)
    if index_path is None:
        return None
    tiles = pyogrio.read_dataframe(index_path, columns=["Name"]).to_crs("EPSG:4326")
    tiles = tiles[tiles.geometry.notna() & ~tiles.geometry.is_empty].copy()
    tiles["Name"] = tiles["Name"].astype(str).str.upper().str.removeprefix("T")
    return tiles


def attach_sentinel_tile_metadata(lakes: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tile_names = []
    tile_rows = tiles[["Name", "geometry"]].copy()
    tile_rows_m = tile_rows.to_crs("EPSG:3857")
    lakes_m = lakes.to_crs("EPSG:3857")
    for lake in lakes.itertuples():
        lake_geom = lake.geometry
        candidates = tile_rows[tile_rows.geometry.intersects(lake_geom)].copy()
        if candidates.empty:
            tile_names.append("")
            continue
        lake_geom_m = lakes_m.geometry.iloc[lake.Index]
        candidates_m = tile_rows_m.loc[candidates.index]
        candidates["overlap"] = candidates_m.geometry.intersection(lake_geom_m).area
        candidates = candidates[candidates["overlap"] > 0].copy()
        if candidates.empty:
            tile_names.append("")
            continue
        candidates = candidates.sort_values(["overlap", "Name"], ascending=[False, True])
        tile_names.append(",".join(candidates["Name"].tolist()))
    lakes["sentinel_tiles"] = tile_names
    initialize_empty_tci_columns(lakes)
    return lakes


def attach_empty_sentinel_metadata(lakes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lakes["sentinel_tiles"] = ""
    initialize_empty_tci_columns(lakes)
    return lakes


def initialize_empty_tci_columns(lakes: gpd.GeoDataFrame) -> None:
    defaults = {
        "best_tci_tile": None,
        "best_tci_date": None,
        "best_tci_product": None,
        "best_tci_valid_ratio": None,
        "best_tci_path": None,
        "has_tci": 0,
    }
    for column, value in defaults.items():
        if column not in lakes.columns:
            lakes[column] = value


def attach_tci_metadata(
    lakes: gpd.GeoDataFrame,
    tci_index: pd.DataFrame,
    footprints: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if tci_index.empty or footprints.empty:
        return lakes
    tci_by_tile = {row.tile: row for row in tci_index.itertuples()}
    footprints_m = footprints.to_crs("EPSG:3857")
    tile_names = []
    best_tiles = []
    best_dates = []
    best_products = []
    best_valid_ratios = []
    best_paths = []
    has_tci = []
    lakes_m = lakes.to_crs("EPSG:3857")

    for lake in lakes.itertuples():
        lake_geom = lake.geometry
        candidates = footprints[footprints.geometry.intersects(lake_geom)].copy()
        if candidates.empty:
            tile_names.append("")
            best_tiles.append(None)
            best_dates.append(None)
            best_products.append(None)
            best_valid_ratios.append(None)
            best_paths.append(None)
            has_tci.append(0)
            continue
        lake_geom_m = lakes_m.geometry.iloc[lake.Index]
        candidates_m = footprints_m.loc[candidates.index]
        candidates["overlap"] = candidates_m.geometry.intersection(lake_geom_m).area
        candidates = candidates[candidates["overlap"] > 0].copy()
        if candidates.empty:
            tile_names.append("")
            best_tiles.append(None)
            best_dates.append(None)
            best_products.append(None)
            best_valid_ratios.append(None)
            best_paths.append(None)
            has_tci.append(0)
            continue
        candidates = candidates.sort_values(["overlap", "tci_valid_ratio"], ascending=False)
        tiles = candidates["tile"].tolist()
        best = tci_by_tile[tiles[0]]
        tile_names.append(",".join(tiles))
        best_tiles.append(best.tile)
        best_dates.append(best.tci_date)
        best_products.append(best.tci_product)
        best_valid_ratios.append(best.tci_valid_ratio)
        best_paths.append(best.tci_path)
        has_tci.append(1)

    lakes["sentinel_tiles"] = tile_names
    lakes["best_tci_tile"] = best_tiles
    lakes["best_tci_date"] = best_dates
    lakes["best_tci_product"] = best_products
    lakes["best_tci_valid_ratio"] = best_valid_ratios
    lakes["best_tci_path"] = best_paths
    lakes["has_tci"] = has_tci
    return lakes


def attach_hydrolakes_metadata(lakes: gpd.GeoDataFrame, hydrolakes_path: Path) -> gpd.GeoDataFrame:
    west, south, east, north = lakes.total_bounds
    pad = 0.5
    columns = ["Hylak_id", "Lake_name", "Country", "Lake_area"]
    candidates = pyogrio.read_dataframe(
        hydrolakes_path,
        bbox=(west - pad, south - pad, east + pad, north + pad),
        columns=columns,
    ).to_crs("EPSG:4326")
    if candidates.empty:
        add_empty_hydrolakes_columns(lakes)
        return lakes

    lakes_m = lakes[["lake_uid", "geometry"]].to_crs("EPSG:3857")
    hydro_m = candidates.to_crs("EPSG:3857").reset_index(drop=True)
    joined = gpd.sjoin(
        lakes_m,
        hydro_m[["Hylak_id", "Lake_name", "Country", "Lake_area", "geometry"]],
        how="left",
        predicate="intersects",
    )

    best_by_uid = {}
    for row in joined.dropna(subset=["index_right"]).itertuples():
        lake_geom = lakes_m.loc[row.Index, "geometry"]
        hydro_geom = hydro_m.loc[int(row.index_right), "geometry"]
        overlap_m2 = lake_geom.intersection(hydro_geom).area
        current = best_by_uid.get(row.lake_uid)
        if current is None or overlap_m2 > current["hydrolakes_overlap_m2"]:
            best_by_uid[row.lake_uid] = {
                "hylak_id": int(row.Hylak_id) if pd.notna(row.Hylak_id) else None,
                "hylak_name": clean_text(row.Lake_name),
                "hylak_country": clean_text(row.Country),
                "hylak_area_km2": float(row.Lake_area) if pd.notna(row.Lake_area) else None,
                "hydrolakes_overlap_m2": float(overlap_m2),
            }

    lakes["hylak_id"] = [best_by_uid.get(uid, {}).get("hylak_id") for uid in lakes["lake_uid"]]
    lakes["hylak_name"] = [best_by_uid.get(uid, {}).get("hylak_name") for uid in lakes["lake_uid"]]
    lakes["hylak_country"] = [best_by_uid.get(uid, {}).get("hylak_country") for uid in lakes["lake_uid"]]
    lakes["hylak_area_km2"] = [best_by_uid.get(uid, {}).get("hylak_area_km2") for uid in lakes["lake_uid"]]
    lakes["hydrolakes_overlap_m2"] = [
        best_by_uid.get(uid, {}).get("hydrolakes_overlap_m2") for uid in lakes["lake_uid"]
    ]
    lakes["has_hydrolakes_polygon"] = lakes["hylak_id"].notna().astype(int)
    return lakes


def add_empty_hydrolakes_columns(lakes: gpd.GeoDataFrame) -> None:
    lakes["hylak_id"] = None
    lakes["hylak_name"] = None
    lakes["hylak_country"] = None
    lakes["hylak_area_km2"] = None
    lakes["hydrolakes_overlap_m2"] = None
    lakes["has_hydrolakes_polygon"] = 0


def finalize_columns(lakes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lakes["polygon_quality"] = [
        infer_polygon_quality(row.area_km2, row.has_hydrolakes_polygon, row.has_tci)
        for row in lakes.itertuples()
    ]
    lakes["metadata_quality"] = [
        infer_metadata_quality(row.display_name, row.hylak_id, row.best_tci_tile)
        for row in lakes.itertuples()
    ]
    lakes["other_tags_json"] = lakes["other_tags"].map(lambda value: json.dumps(parse_other_tags(value), ensure_ascii=False))

    columns = [
        "lake_uid",
        "display_name",
        "name",
        "name_zh",
        "name_en",
        "source_primary",
        "source_feature_id",
        "osm_id_text",
        "osm_way_id_text",
        "hylak_id",
        "hylak_name",
        "hylak_country",
        "hylak_area_km2",
        "hydrolakes_overlap_m2",
        "wikidata",
        "wikipedia",
        "osm_code",
        "nsdi_code",
        "zhb_code",
        "water_tag",
        "water_type",
        "area_km2",
        "bbox_west",
        "bbox_south",
        "bbox_east",
        "bbox_north",
        "center_lon",
        "center_lat",
        "province",
        "city",
        "county",
        "admin_source",
        "sentinel_tiles",
        "best_tci_tile",
        "best_tci_date",
        "best_tci_product",
        "best_tci_valid_ratio",
        "best_tci_path",
        "has_tci",
        "has_osm_polygon",
        "has_hydrolakes_polygon",
        "has_esa_polygon",
        "polygon_quality",
        "metadata_quality",
        "natural",
        "landuse",
        "man_made",
        "type",
        "other_tags_json",
        "geometry",
    ]
    return lakes[columns].copy()


def parse_other_tags(value) -> dict[str, str]:
    if value is None or pd.isna(value):
        return {}
    return dict(TAG_RE.findall(str(value)))


def resolve_tci_path(value, data_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "data_download" and parts[1] == "downloads":
        return data_dir.joinpath(*parts[2:])
    return PROJECT_ROOT / path


def clean_id(value) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    if text.endswith(".0"):
        return text[:-2]
    return text


def clean_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def first_present(*values) -> str | None:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return None


def make_lake_uid(prefix: str, osm_id: str | None, osm_way_id: str | None, fallback: str) -> str:
    osm_id = clean_id(osm_id)
    osm_way_id = clean_id(osm_way_id)
    fallback = clean_id(fallback) or "feature"
    if osm_id:
        return f"{prefix}_osm_{osm_id}"
    if osm_way_id:
        return f"{prefix}_osm_way_{osm_way_id}"
    return f"{prefix}_osm_{fallback}"


def infer_water_type(row, tags: dict[str, str]) -> str:
    name = first_present(row.name, row.name_zh, row.name_en, "") or ""
    water = clean_text(tags.get("water")) or ""
    landuse = clean_text(row.landuse) or ""
    natural = clean_text(row.natural) or ""

    if water == "reservoir" or landuse == "reservoir" or "水库" in name:
        return "reservoir"
    if landuse == "aquaculture" or "鱼塘" in name or "养殖" in name:
        return "aquaculture"
    if natural == "wetland" or water == "wetland" or "湿地" in name:
        return "wetland"
    if water == "pond" or "塘" in name:
        return "pond"
    if water == "lake" or "湖" in name:
        return "lake"
    if row.area_km2 < 0.1 and not first_present(row.name, row.name_zh, row.name_en):
        return "pond_candidate"
    return "unknown"


def infer_polygon_quality(area_km2: float, has_hydrolakes: int, has_tci: int) -> str:
    if has_hydrolakes and has_tci:
        return "high"
    if has_tci or area_km2 >= 1:
        return "medium"
    return "low"


def infer_metadata_quality(display_name: str | None, hylak_id, best_tci_tile: str | None) -> str:
    score = 0
    if display_name:
        score += 1
    if pd.notna(hylak_id):
        score += 1
    if best_tci_tile:
        score += 1
    return ["low", "medium", "high", "high"][score]


def print_summary(lakes: gpd.GeoDataFrame) -> None:
    print(f"rows={len(lakes)}")
    print(f"named={int(lakes['name'].notna().sum())}")
    print(f"has_tci={int(lakes['has_tci'].sum())}")
    print(f"has_hydrolakes={int(lakes['has_hydrolakes_polygon'].sum())}")
    print("water_type_counts=")
    for water_type, count in lakes["water_type"].value_counts().items():
        print(f"  {water_type}: {count}")


if __name__ == "__main__":
    main()
