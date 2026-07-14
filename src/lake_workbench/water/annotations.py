"""Site-level orchestration for external and contextual water annotations."""

from __future__ import annotations

from typing import Any

from shapely.geometry import mapping

from lake_workbench.geo import site_aoi_geometry
from lake_workbench.utils import clean_optional, jsonable
from lake_workbench.water.providers import annotation_provider, annotation_status, candidate_properties


class WaterAnnotationsMixin:
    """Annotation operations that require an observation-site catalog."""

    region: Any

    def annotation_for_site(self, site: Any, source: str, options: dict | None = None) -> dict:
        provider = annotation_provider(source)
        loaded = provider.load(self, site, options or {})
        return {
            "site_id": site.site_id,
            "source": provider.source,
            "status": annotation_status(loaded.annotation),
            "parameters": loaded.parameters,
            "annotation": loaded.annotation,
        }

    def context_water_for_site(
        self,
        site: Any,
        padding: float = 0.8,
        min_area_km2: float = 1.0,
        limit: int = 500,
    ) -> dict:
        aoi = site_aoi_geometry(site, padding=padding)
        return {
            "site_id": site.site_id,
            "aoi_bounds": list(aoi.bounds),
            "min_area_km2": min_area_km2,
            "limit": limit,
            "sources": {
                "osm": self._candidate_collection(site, "osm", "OSM", min_area_km2, limit),
                "hydrolakes": self._candidate_collection(site, "hydrolakes", "HydroLAKES", min_area_km2, limit),
            },
        }

    def _candidate_collection(
        self,
        site: Any,
        source: str,
        source_label: str,
        min_area_km2: float,
        limit: int,
    ) -> dict:
        candidates = self.water_candidates_for_site(site, source)
        if candidates is None or candidates.empty:
            return {"type": "FeatureCollection", "features": []}
        primary = self.suggested_water_candidate(site, source)
        primary_id = clean_optional(primary.get("candidate_id")) if primary is not None else None
        candidates = candidates[
            candidates["area_km2"].fillna(0).astype(float) >= float(min_area_km2)
        ].sort_values("intersection_area_km2", ascending=False)
        features = []
        for _, candidate in candidates.head(max(1, int(limit))).iterrows():
            if primary_id and clean_optional(candidate.get("candidate_id")) == primary_id:
                continue
            properties = candidate_properties(candidate)
            properties.update(
                {
                    "candidate_id": jsonable(candidate.get("candidate_id")),
                    "site_id": jsonable(candidate.get("site_id")),
                    "source": source_label,
                    "area_km2": jsonable(candidate.get("area_km2")),
                    "intersection_area_km2": jsonable(candidate.get("intersection_area_km2")),
                }
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(candidate.geometry),
                    "properties": properties,
                }
            )
        return {"type": "FeatureCollection", "features": features}
