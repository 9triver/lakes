"""Lake-level orchestration for external and contextual water annotations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyogrio
from shapely.geometry import mapping
from shapely.validation import make_valid

from lake_workbench.geo import geometry_coverage_ratio, lake_aoi_geometry, transform_geom
from lake_workbench.utils import clean_optional, jsonable, parse_float, truthy_flag
from lake_workbench.water.layers import (
    available_jrc_thresholds,
    build_esa_smoothed_layer,
    build_jrc_occurrence_layer,
    read_esa_polygon_cache,
    read_jrc_polygon_cache,
)


class WaterAnnotationsMixin:
    """External water-layer operations that require a lake catalog."""

    region: Any

    def _osm_layer(self, lake: Any) -> dict | None:
        if clean_optional(lake.properties.get("source_primary")) != "osm":
            return None
        if not truthy_flag(lake.properties.get("has_osm_polygon"), default=True):
            return None
        return {
            "source": "OSM",
            "geometry": mapping(lake.geometry),
            "properties": dict(lake.properties),
        }

    def context_water_for_lake(
        self,
        lake: Any,
        padding: float = 0.8,
        min_area_km2: float = 1.0,
        limit: int = 500,
    ) -> dict:
        aoi = lake_aoi_geometry(lake, padding=padding)
        return {
            "lake_id": lake.object_id,
            "aoi_bounds": list(aoi.bounds),
            "min_area_km2": min_area_km2,
            "limit": limit,
            "sources": {
                "osm": self._context_osm_water(lake, aoi, min_area_km2=min_area_km2, limit=limit),
                "hydrolakes": self._context_hydrolakes_water(lake, aoi, min_area_km2=min_area_km2, limit=limit),
            },
        }

    def _context_osm_water(self, lake: Any, aoi: Any, min_area_km2: float, limit: int) -> dict:
        if not self.region.osm_water.exists():
            return {"type": "FeatureCollection", "features": []}
        xmin, ymin, xmax, ymax = aoi.bounds
        columns = ["osm_id", "osm_way_id", "name", "type", "natural", "landuse", "man_made", "other_tags"]
        try:
            candidates = pyogrio.read_dataframe(
                self.region.osm_water,
                layer="osm_water_polygons",
                columns=columns,
                bbox=(xmin, ymin, xmax, ymax),
            ).to_crs("EPSG:4326")
        except Exception:
            return {"type": "FeatureCollection", "features": []}
        if candidates.empty:
            return {"type": "FeatureCollection", "features": []}
        candidates["area_km2"] = candidates.to_crs("EPSG:3857").geometry.area / 1_000_000
        target_ids = {
            clean_optional(lake.properties.get("osm_id_text")),
            clean_optional(lake.properties.get("osm_way_id_text")),
        }
        target_ids.discard(None)
        items = []
        for _, row in candidates.sort_values("area_km2", ascending=False).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.intersects(aoi):
                continue
            other_tags = str(row.get("other_tags") or "")
            if '"water"=>"river"' in other_tags or '"water"=>"canal"' in other_tags:
                continue
            area_km2 = parse_float(row.get("area_km2")) or 0.0
            if area_km2 < min_area_km2:
                continue
            osm_id = clean_optional(row.get("osm_id"))
            osm_way_id = clean_optional(row.get("osm_way_id"))
            if (osm_id and osm_id in target_ids) or (osm_way_id and osm_way_id in target_ids):
                continue
            if geometry_coverage_ratio(lake.geometry, geom) > 0.9 and geometry_coverage_ratio(geom, lake.geometry) > 0.9:
                continue
            clipped = make_valid(geom.intersection(aoi))
            if clipped.is_empty:
                continue
            items.append(
                {
                    "area_km2": area_km2,
                    "feature": {
                        "type": "Feature",
                        "geometry": mapping(clipped),
                        "properties": {
                            "source": "OSM",
                            "osm_id": osm_id,
                            "osm_way_id": osm_way_id,
                            "name": clean_optional(row.get("name")) or "",
                            "type": clean_optional(row.get("type")),
                            "natural": clean_optional(row.get("natural")),
                            "landuse": clean_optional(row.get("landuse")),
                            "man_made": clean_optional(row.get("man_made")),
                            "area_km2": area_km2,
                        },
                    },
                }
            )
        items.sort(key=lambda item: item["area_km2"], reverse=True)
        return {"type": "FeatureCollection", "features": [item["feature"] for item in items[:limit]]}

    def _context_hydrolakes_water(self, lake: Any, aoi: Any, min_area_km2: float, limit: int) -> dict:
        xmin, ymin, xmax, ymax = aoi.bounds
        try:
            candidates = pyogrio.read_dataframe(
                self.region.hydrolakes,
                bbox=(xmin, ymin, xmax, ymax),
                columns=["Hylak_id", "Lake_name", "Country", "Lake_area"],
            ).to_crs("EPSG:4326")
        except Exception:
            candidates = pd.DataFrame()
        if candidates.empty:
            return {"type": "FeatureCollection", "features": []}
        target_hylak = clean_optional(lake.properties.get("hylak_id"))
        items = []
        for _, row in candidates.iterrows():
            hylak_id = clean_optional(row.get("Hylak_id"))
            if target_hylak and hylak_id == target_hylak:
                continue
            area_km2 = parse_float(row.get("Lake_area")) or 0.0
            geom = row.geometry
            if area_km2 < min_area_km2 or geom is None or geom.is_empty or not geom.intersects(aoi):
                continue
            clipped = make_valid(geom.intersection(aoi))
            if clipped.is_empty:
                continue
            items.append(
                {
                    "area_km2": area_km2,
                    "feature": {
                        "type": "Feature",
                        "geometry": mapping(clipped),
                        "properties": {
                            "source": "HydroLAKES",
                            "hylak_id": jsonable(row.get("Hylak_id")),
                            "name": jsonable(row.get("Lake_name")) or "",
                            "country": jsonable(row.get("Country")),
                            "area_km2": jsonable(row.get("Lake_area")),
                        },
                    },
                }
            )
        items.sort(key=lambda item: item["area_km2"], reverse=True)
        return {"type": "FeatureCollection", "features": [item["feature"] for item in items[:limit]]}

    def _match_hydrolakes(self, lake: Any) -> dict | None:
        xmin, ymin, xmax, ymax = lake.bbox
        pad_x = max((xmax - xmin) * 0.2, 0.01)
        pad_y = max((ymax - ymin) * 0.2, 0.01)
        try:
            candidates = pyogrio.read_dataframe(
                self.region.hydrolakes,
                bbox=(xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y),
                columns=["Hylak_id", "Lake_name", "Country", "Lake_area"],
            ).to_crs("EPSG:4326")
        except Exception:
            return None
        if candidates.empty:
            return None
        lake_m = transform_geom(lake.geometry, "EPSG:4326", "EPSG:3857")
        best = None
        best_area = 0.0
        for _, row in candidates.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.intersects(lake.geometry):
                continue
            overlap = transform_geom(geom, "EPSG:4326", "EPSG:3857").intersection(lake_m).area
            if overlap > best_area:
                best_area = overlap
                best = row
        if best is None:
            return None
        return {
            "source": "HydroLAKES",
            "geometry": mapping(best.geometry),
            "properties": {
                "Hylak_id": jsonable(best.get("Hylak_id")),
                "Lake_name": jsonable(best.get("Lake_name")),
                "Country": jsonable(best.get("Country")),
                "Lake_area": jsonable(best.get("Lake_area")),
                "overlap_m2": best_area,
            },
        }

    def _esa_smoothed_layer(self, lake: Any) -> dict | None:
        cached = read_esa_polygon_cache(self.region, lake.object_id)
        if cached is not None:
            return cached
        if lake.area_km2 > 250:
            return {
                "source": "ESA WorldCover 2021 water mask, smoothed",
                "geometry": None,
                "properties": {
                    "water_id": f"ESA_{lake.object_id}",
                    "skipped": True,
                    "reason": "object too large for on-demand ESA polygonization; pre-generate this lake first",
                },
            }
        return build_esa_smoothed_layer(self.region, lake)

    def _jrc_occurrence_layer(self, lake: Any, threshold: int = 75) -> dict | None:
        threshold = max(1, min(100, int(threshold)))
        cached = read_jrc_polygon_cache(self.region, lake.object_id, threshold)
        if cached is not None:
            return cached
        if lake.area_km2 > 250:
            return {
                "source": "JRC GSW occurrence 2021",
                "geometry": None,
                "properties": {
                    "water_id": f"JRC_{lake.object_id}_{threshold}",
                    "threshold": threshold,
                    "skipped": True,
                    "reason": "object too large for on-demand JRC polygonization; pre-generate this threshold first",
                    "available_thresholds": available_jrc_thresholds(self.region, lake.object_id),
                },
            }
        return build_jrc_occurrence_layer(self.region, lake, threshold)
