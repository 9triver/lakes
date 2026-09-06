"""Source readers shared by the observation-site metadata builder."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from lake_workbench.metadata.geometry import geometry_areas_km2, metric_geometry
from lake_workbench.paths import PROJECT_ROOT


TAG_RE = re.compile(r'"([^"]+)"=>"([^"]*)"')


def spatial_intersections(frame: gpd.GeoDataFrame, geometry) -> gpd.GeoDataFrame:
    """Return intersecting features using the frame's spatial index."""
    if frame.empty:
        return frame
    try:
        indexes = frame.sindex.query(geometry, predicate="intersects")
    except (AttributeError, ImportError, NotImplementedError):
        # Keep the builder usable with older GeoPandas installations.
        return frame[frame.geometry.intersects(geometry)]
    return frame.iloc[indexes]


def image_tile_name(name: str, geometry, sentinel_tile_index: gpd.GeoDataFrame | None) -> str:
    match = re.search(r"_T([0-9A-Z]{5})_", name)
    if match:
        return match.group(1)
    if sentinel_tile_index is None or sentinel_tile_index.empty:
        return ""
    candidates = spatial_intersections(sentinel_tile_index, geometry).copy()
    if candidates.empty:
        return ""
    geometry_m = metric_geometry(geometry)
    candidates["overlap"] = [metric_geometry(tile).intersection(geometry_m).area / 1_000_000 for tile in candidates.geometry]
    candidates = candidates[candidates["overlap"] > 0].sort_values(["overlap", "Name"], ascending=[False, True])
    return str(candidates.iloc[0]["Name"]) if not candidates.empty else ""


def image_date(name: str) -> str:
    # Local imagery also contains Sentinel-1 and mosaic products whose
    # prefixes differ from MSIL1C/MSIL2A, but all use an 8-digit date.
    match = re.search(r"(?:^|[_-])(\d{8})(?:[_.-]|$)", name)
    if not match:
        return ""
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def load_external_osm_water(osm_path: Path, bbox: tuple[float, float, float, float] | None = None) -> gpd.GeoDataFrame:
    if not osm_path.exists():
        return empty_external_water()
    columns = ["osm_id", "osm_way_id", "name", "type", "natural", "landuse", "man_made", "other_tags"]
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
    data["area_km2"] = geometry_areas_km2(data.geometry)
    data["osm_id_text"] = data["osm_id"].map(clean_id)
    data["osm_way_id_text"] = data["osm_way_id"].map(clean_id)
    data["source_feature_id"] = [
        first_present(row.osm_id_text, row.osm_way_id_text, f"osm_{index + 1}")
        for index, row in data.reset_index(drop=True).iterrows()
    ]
    data["name"] = data["name"].map(clean_text)
    data["name_zh"] = [clean_text(tags.get("name:zh")) or row.name for tags, row in zip(tag_records, data.itertuples())]
    data["name_en"] = [clean_text(tags.get("name:en")) for tags in tag_records]
    data["display_name"] = [first_present(row.name_zh, row.name, row.name_en, row.source_feature_id) for row in data.itertuples()]
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
    data["area_km2"] = geometry_areas_km2(data.geometry)
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


def load_sentinel_tile_index(paths: list[Path]) -> gpd.GeoDataFrame | None:
    index_path = next((path for path in paths if path.exists()), None)
    if index_path is None:
        return None
    tiles = pyogrio.read_dataframe(index_path, columns=["Name"]).to_crs("EPSG:4326")
    tiles = tiles[tiles.geometry.notna() & ~tiles.geometry.is_empty].copy()
    tiles["Name"] = tiles["Name"].astype(str).str.upper().str.removeprefix("T")
    return tiles


def empty_external_water() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_other_tags(value) -> dict[str, str]:
    if value is None or pd.isna(value):
        return {}
    return dict(TAG_RE.findall(str(value)))


def clean_id(value) -> str | None:
    text = clean_text(value)
    if text and text.endswith(".0"):
        return text[:-2]
    return text


def clean_text(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def first_present(*values) -> str | None:
    return next((text for value in values if (text := clean_text(value))), None)


def infer_water_type(row, tags: dict[str, str]) -> str:
    water = clean_text(tags.get("water"))
    if water in {"reservoir", "pond", "lake", "lagoon", "basin", "oxbow"}:
        return water
    if clean_text(row.landuse) == "reservoir":
        return "reservoir"
    if clean_text(row.man_made) in {"reservoir_covered", "wastewater_plant"}:
        return "reservoir"
    return "lake"
