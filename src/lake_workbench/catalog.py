#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regional lake metadata, imagery inventory, and Sentinel product catalog."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pandas as pd
import pyogrio
from shapely.geometry import box, mapping

from lake_workbench.models.validation import ModelValidationMixin
from lake_workbench.local_labels import LocalLabelCatalogMixin
from lake_workbench.imagery.inventory import ImageryInventoryMixin
from lake_workbench.imagery.rendering import ImageryRenderingMixin
from lake_workbench.regions.config import RegionConfig
from lake_workbench.sentinel.catalog import SentinelCatalogMixin
from lake_workbench.training.catalog import TrainingCatalogMixin
from lake_workbench.water.annotations import WaterAnnotationsMixin
from lake_workbench.utils import (
    area_in_bucket,
    clean_optional,
    count_values,
    display_path,
    first_present,
    jsonable as _jsonable,
    legacy_lake_keys,
    metadata_tiles,
    parse_float,
)


@dataclass
class LakeRecord:
    lake_id: str
    object_id: str
    name: str | None
    source: str
    area_km2: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    properties: dict
    geometry: object


class LakeCatalog(
    ImageryInventoryMixin,
    ImageryRenderingMixin,
    LocalLabelCatalogMixin,
    SentinelCatalogMixin,
    ModelValidationMixin,
    TrainingCatalogMixin,
    WaterAnnotationsMixin,
):
    def __init__(self, region: RegionConfig, default_region_key: str) -> None:
        self.region = region
        self.default_region_key = default_region_key
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self.base_tci_by_tile = self._load_tci_index()
        self.tci_by_tile = dict(self.base_tci_by_tile)
        self.user_tci_rows = self._load_user_tci_rows()
        self.active_imagery = self._load_active_imagery()
        self._rebuild_effective_tci()
        self.tci_footprints = self._load_tci_footprints()
        self.sentinel_tile_index = self._load_sentinel_tile_index()
        self._valid_ratio_cache: dict[str, float] = {}
        self.lakes = self._load_lakes()
        self._lake_lookup = self._build_lake_lookup()
        self._summary_cache = {lake.object_id: self._build_summary(lake) for lake in self.lakes}
        self._detail_cache: dict[str, dict] = {}

    def _load_lakes(self) -> list[LakeRecord]:
        if self.region.lake_metadata.exists():
            return self._load_lakes_from_metadata()
        if not self.region.osm_water.exists():
            self.load_error = (
                f"Region data is not prepared: missing {display_path(self.region.lake_metadata)} "
                f"or {display_path(self.region.osm_water)}"
            )
            return []
        return self._load_lakes_from_osm()

    def _load_lakes_from_metadata(self) -> list[LakeRecord]:
        data = pyogrio.read_dataframe(self.region.lake_metadata, layer="lake_metadata").to_crs("EPSG:4326")
        records: list[LakeRecord] = []
        for _, row in data.sort_values("area_km2", ascending=False).reset_index(drop=True).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            object_id = str(row.get("lake_uid"))
            lake_id = first_present(row.get("source_feature_id"), row.get("osm_id_text"), row.get("osm_way_id_text"), object_id)
            name = clean_optional(row.get("display_name")) or clean_optional(row.get("name"))
            bbox = (
                float(row.get("bbox_west")),
                float(row.get("bbox_south")),
                float(row.get("bbox_east")),
                float(row.get("bbox_north")),
            )
            attrs = {
                key: _jsonable(row.get(key))
                for key in data.columns
                if key != "geometry"
            }
            records.append(
                LakeRecord(
                    lake_id=str(lake_id),
                    object_id=object_id,
                    name=name,
                    source=clean_optional(row.get("source_primary")) or "metadata",
                    area_km2=float(row.get("area_km2")),
                    bbox=bbox,
                    center=(float(row.get("center_lon")), float(row.get("center_lat"))),
                    properties=attrs,
                    geometry=geom,
                )
            )
        return records

    def _load_lakes_from_osm(self) -> list[LakeRecord]:
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
        data = pyogrio.read_dataframe(self.region.osm_water, layer="osm_water_polygons", columns=columns)
        data = data.to_crs("EPSG:4326")
        metric = data.to_crs("EPSG:3857")
        data["area_km2"] = metric.geometry.area / 1_000_000
        records: list[LakeRecord] = []
        for index, row in data.sort_values("area_km2", ascending=False).reset_index(drop=True).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            other_tags = str(row.get("other_tags") or "")
            # Keep lakes/reservoir-like water objects as the primary workbench list.
            if '"water"=>"river"' in other_tags or '"water"=>"canal"' in other_tags:
                continue
            if float(row["area_km2"]) < 0.01:
                continue
            osm_id = first_present(row.get("osm_id"), row.get("osm_way_id"), f"feature_{index + 1}")
            object_id = f"osm_{osm_id}"
            name = row.get("name")
            if pd.isna(name):
                name = None
            xmin, ymin, xmax, ymax = geom.bounds
            attrs = {
                key: _jsonable(row.get(key))
                for key in ["osm_id", "osm_way_id", "name", "type", "natural", "landuse", "man_made", "other_tags"]
            }
            records.append(
                LakeRecord(
                    lake_id=str(osm_id),
                    object_id=object_id,
                    name=name,
                    source="osm",
                    area_km2=float(row["area_km2"]),
                    bbox=(xmin, ymin, xmax, ymax),
                    center=((xmin + xmax) / 2, (ymin + ymax) / 2),
                    properties=attrs,
                    geometry=geom,
                )
            )
        return records

    def _build_lake_lookup(self) -> dict[str, LakeRecord]:
        lookup = {}
        for lake in self.lakes:
            keys = [
                lake.object_id,
                lake.lake_id,
                lake.properties.get("lake_uid"),
                lake.properties.get("source_feature_id"),
                lake.properties.get("osm_id_text"),
                lake.properties.get("osm_way_id_text"),
            ]
            osm_id = clean_optional(lake.properties.get("osm_id_text"))
            osm_way_id = clean_optional(lake.properties.get("osm_way_id_text"))
            if osm_id:
                keys.append(f"osm_{osm_id}")
            if osm_way_id:
                keys.append(f"osm_{osm_way_id}")
            keys.extend(legacy_lake_keys(lake.object_id))
            for key in keys:
                text = clean_optional(key)
                if text:
                    lookup[text] = lake
        return lookup

    def list_lakes(
        self,
        query: str = "",
        limit: int = 200,
        offset: int = 0,
        filters: dict | None = None,
    ) -> dict:
        query = query.strip().lower()
        filtered = self.lakes
        if query:
            filtered = [
                lake
                for lake in filtered
                if query in lake.lake_id.lower()
                or query in lake.object_id.lower()
                or query in str(lake.name or "").lower()
                or query in str(lake.properties.get("display_name", "")).lower()
                or query in str(lake.properties.get("name_zh", "")).lower()
                or query in str(lake.properties.get("name_en", "")).lower()
                or query in str(lake.properties.get("hylak_id", "")).lower()
                or query in str(lake.properties.get("osm_way_id", "")).lower()
                or query in str(lake.properties.get("osm_way_id_text", "")).lower()
            ]
        filtered = self._apply_filters(filtered, filters or {})
        page = filtered[offset : offset + limit]
        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "facets": self._facets(filtered),
            "items": [self._summary(lake) for lake in page],
        }

    def _apply_filters(self, lakes: list[LakeRecord], filters: dict) -> list[LakeRecord]:
        result = lakes
        for key in ["water_type", "province", "city", "county", "polygon_quality", "metadata_quality"]:
            value = clean_optional(filters.get(key))
            if value and value != "all":
                result = [lake for lake in result if clean_optional(lake.properties.get(key)) == value]

        has_tci = clean_optional(filters.get("has_tci"))
        if has_tci in {"true", "false"}:
            expected = has_tci == "true"
            result = [lake for lake in result if bool(lake.properties.get("has_tci", self._has_tci(lake))) is expected]

        has_name = clean_optional(filters.get("has_name"))
        if has_name in {"true", "false"}:
            expected = has_name == "true"
            result = [lake for lake in result if self._has_real_name(lake) is expected]

        area_bucket = clean_optional(filters.get("area_bucket"))
        if area_bucket and area_bucket != "all":
            result = [lake for lake in result if area_in_bucket(lake.area_km2, area_bucket)]

        min_area = parse_float(filters.get("min_area"))
        max_area = parse_float(filters.get("max_area"))
        if min_area is not None:
            result = [lake for lake in result if lake.area_km2 >= min_area]
        if max_area is not None:
            result = [lake for lake in result if lake.area_km2 <= max_area]
        return result

    def _facets(self, lakes: list[LakeRecord]) -> dict:
        return {
            "water_type": count_values(lake.properties.get("water_type") for lake in lakes),
            "province": count_values(lake.properties.get("province") for lake in lakes),
            "city": count_values(lake.properties.get("city") for lake in lakes),
            "county": count_values(lake.properties.get("county") for lake in lakes),
            "polygon_quality": count_values(lake.properties.get("polygon_quality") for lake in lakes),
            "metadata_quality": count_values(lake.properties.get("metadata_quality") for lake in lakes),
        }

    def get_lake(self, lake_key: str) -> LakeRecord | None:
        return self._lake_lookup.get(lake_key)

    def get_lake_detail(self, lake: LakeRecord) -> dict:
        if lake.object_id in self._detail_cache:
            return self._detail_cache[lake.object_id]
        attrs = dict(lake.properties)
        hydrolakes = self._match_hydrolakes(lake)
        osm_layer = self._osm_layer(lake)
        detail = {
            **self._summary(lake),
            "properties": attrs,
            "layers": {
                "osm": osm_layer,
                "hydrolakes": hydrolakes,
                "esa": None,
                "jrc": None,
            },
            "geometry": mapping(lake.geometry),
        }
        self._detail_cache[lake.object_id] = detail
        return detail


    def _summary(self, lake: LakeRecord) -> dict:
        return dict(self._summary_cache.get(lake.object_id) or self._build_summary(lake))

    def _build_summary(self, lake: LakeRecord) -> dict:
        tiles = metadata_tiles(lake.properties.get("sentinel_tiles"))
        if not tiles and lake.properties.get("best_tci_tile"):
            tiles = [str(lake.properties.get("best_tci_tile")).upper().removeprefix("T")]
        return {
            "lake_id": lake.lake_id,
            "object_id": lake.object_id,
            "shape_id": lake.object_id,
            "name": lake.name,
            "display_name": lake.properties.get("display_name") or lake.name,
            "source": lake.source,
            "water_type": lake.properties.get("water_type"),
            "area_km2": lake.area_km2,
            "center": lake.center,
            "bbox": lake.bbox,
            "province": lake.properties.get("province"),
            "city": lake.properties.get("city"),
            "county": lake.properties.get("county"),
            "hylak_id": lake.properties.get("hylak_id"),
            "best_tci_tile": lake.properties.get("best_tci_tile"),
            "best_tci_date": lake.properties.get("best_tci_date"),
            "best_tci_valid_ratio": lake.properties.get("best_tci_valid_ratio"),
            "polygon_quality": lake.properties.get("polygon_quality"),
            "metadata_quality": lake.properties.get("metadata_quality"),
            "tiles": tiles,
            "has_tci": bool(lake.properties.get("has_tci") or lake.properties.get("best_tci_tile")),
        }

    def _has_tci(self, lake: LakeRecord) -> bool:
        return any(item["geometry"].intersects(box(*lake.bbox)) for item in self.tci_footprints)

    def _has_real_name(self, lake: LakeRecord) -> bool:
        return any(
            clean_optional(lake.properties.get(key))
            for key in ["name", "name_zh", "name_en"]
        )
