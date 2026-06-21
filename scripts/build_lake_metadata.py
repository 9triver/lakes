#!/usr/bin/env python3
"""Build the unified Hunan lake metadata dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TAG_RE = re.compile(r'"([^"]+)"=>"([^"]*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-area-km2", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = InputPaths(data_dir)
    print(f"data_dir={data_dir}")
    print("loading OSM water polygons")
    lakes = load_osm_lakes(paths.osm_water, args.min_area_km2)
    print(f"osm_lakes={len(lakes)}")

    sentinel_tile_index = load_sentinel_tile_index(paths.sentinel_tile_index_paths)
    if sentinel_tile_index is None:
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
    gpkg_path = output_dir / "hunan_lake_metadata.gpkg"
    csv_path = output_dir / "hunan_lake_metadata.csv"

    if gpkg_path.exists():
        gpkg_path.unlink()
    pyogrio.write_dataframe(lakes, gpkg_path, layer="lake_metadata", driver="GPKG")
    lakes.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    print(f"wrote {gpkg_path}")
    print(f"wrote {csv_path}")
    print_summary(lakes)


class InputPaths:
    def __init__(self, data_dir: Path) -> None:
        self.osm_water = data_dir / "hunan_osm_water" / "hunan_water_raw.gpkg"
        self.tci_index = data_dir / "hunan_single_tiles" / "hunan_selected_best_coverage_tci_valid.csv"
        self.sentinel_tile_index_paths = [
            data_dir / "sentinel_2_tiles" / "sentinel_2_index.geojson",
            data_dir / "sentinel_2_tiles" / "sentinel_2_index_shapefile.shp",
        ]
        self.hydrolakes = data_dir / "hydrolakes" / "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp"


def load_osm_lakes(osm_path: Path, min_area_km2: float) -> gpd.GeoDataFrame:
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
        make_lake_uid(row.osm_id_text, row.osm_way_id_text, row.source_feature_id)
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

    lakes["province"] = "湖南省"
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
    lakes["best_tci_tile"] = None
    lakes["best_tci_date"] = None
    lakes["best_tci_product"] = None
    lakes["best_tci_valid_ratio"] = None
    lakes["best_tci_path"] = None
    lakes["has_tci"] = 0
    return lakes


def attach_empty_sentinel_metadata(lakes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lakes["sentinel_tiles"] = ""
    lakes["best_tci_tile"] = None
    lakes["best_tci_date"] = None
    lakes["best_tci_product"] = None
    lakes["best_tci_valid_ratio"] = None
    lakes["best_tci_path"] = None
    lakes["has_tci"] = 0
    return lakes


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


def make_lake_uid(osm_id: str | None, osm_way_id: str | None, fallback: str) -> str:
    osm_id = clean_id(osm_id)
    osm_way_id = clean_id(osm_way_id)
    fallback = clean_id(fallback) or "feature"
    if osm_id:
        return f"hn_osm_{osm_id}"
    if osm_way_id:
        return f"hn_osm_way_{osm_way_id}"
    return f"hn_osm_{fallback}"


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
