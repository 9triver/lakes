"""Annotation providers for vector candidates and raster-derived water layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from shapely.geometry import mapping

from lake_workbench.utils import clean_optional, jsonable, parse_int_or_default
from lake_workbench.water.layers import (
    available_jrc_thresholds,
    build_esa_smoothed_layer,
    build_jrc_occurrence_layer,
    read_esa_polygon_cache,
    read_jrc_polygon_cache,
)


@dataclass(frozen=True)
class AnnotationLoadResult:
    annotation: dict | None
    parameters: dict


class AnnotationProvider(Protocol):
    source: str

    def load(self, catalog: Any, site: Any, options: dict) -> AnnotationLoadResult: ...


@dataclass(frozen=True)
class VectorCandidateProvider:
    source: str
    source_label: str

    def load(self, catalog: Any, site: Any, options: dict) -> AnnotationLoadResult:
        candidate = catalog.suggested_water_candidate(site, self.source)
        return AnnotationLoadResult(candidate_layer(candidate, self.source_label), {})


class LocalLabelProvider:
    source = "local"

    def load(self, catalog: Any, site: Any, options: dict) -> AnnotationLoadResult:
        label_id = clean_optional(options.get("label_id"))
        if not label_id:
            raise ValueError("label_id is required for local annotations")
        parameters = {"label_id": label_id}
        try:
            payload = catalog.local_label_geojson(site, label_id)
        except FileNotFoundError:
            return AnnotationLoadResult(None, parameters)
        annotation = dict(payload.get("geojson") or {})
        annotation["properties"] = {
            **(annotation.get("properties") or {}),
            "source": "Local label",
            "label": payload.get("label") or {},
        }
        return AnnotationLoadResult(annotation, parameters)


class CachedRasterProvider:
    source: str
    source_label: str
    max_on_demand_area_km2 = 250

    def load(self, catalog: Any, site: Any, options: dict) -> AnnotationLoadResult:
        parameters = self.parameters(options)
        cached = self.read_cache(catalog.region, site, parameters)
        if cached is not None:
            return AnnotationLoadResult(cached, parameters)
        if site.area_km2 > self.max_on_demand_area_km2:
            return AnnotationLoadResult(self.skipped_layer(catalog.region, site, parameters), parameters)
        return AnnotationLoadResult(self.build(catalog.region, site, parameters), parameters)

    def parameters(self, options: dict) -> dict:
        return {}

    def read_cache(self, region: Any, site: Any, parameters: dict) -> dict | None:
        raise NotImplementedError

    def build(self, region: Any, site: Any, parameters: dict) -> dict | None:
        raise NotImplementedError

    def skipped_layer(self, region: Any, site: Any, parameters: dict) -> dict:
        raise NotImplementedError


class EsaAnnotationProvider(CachedRasterProvider):
    source = "esa"
    source_label = "ESA WorldCover 2021 water mask, smoothed"

    def read_cache(self, region: Any, site: Any, parameters: dict) -> dict | None:
        return read_esa_polygon_cache(region, site.site_id)

    def build(self, region: Any, site: Any, parameters: dict) -> dict | None:
        return build_esa_smoothed_layer(region, site)

    def skipped_layer(self, region: Any, site: Any, parameters: dict) -> dict:
        return {
            "source": self.source_label,
            "geometry": None,
            "properties": {
                "water_id": f"ESA_{site.site_id}",
                "skipped": True,
                "reason": "object too large for on-demand ESA polygonization; pre-generate this site first",
            },
        }


class JrcAnnotationProvider(CachedRasterProvider):
    source = "jrc"
    source_label = "JRC GSW occurrence 2021"

    def parameters(self, options: dict) -> dict:
        threshold = max(1, min(100, parse_int_or_default(options.get("threshold"), 75)))
        return {"threshold": threshold}

    def read_cache(self, region: Any, site: Any, parameters: dict) -> dict | None:
        return read_jrc_polygon_cache(region, site.site_id, parameters["threshold"])

    def build(self, region: Any, site: Any, parameters: dict) -> dict | None:
        return build_jrc_occurrence_layer(region, site, parameters["threshold"])

    def skipped_layer(self, region: Any, site: Any, parameters: dict) -> dict:
        threshold = parameters["threshold"]
        return {
            "source": self.source_label,
            "geometry": None,
            "properties": {
                "water_id": f"JRC_{site.site_id}_{threshold}",
                "threshold": threshold,
                "skipped": True,
                "reason": "object too large for on-demand JRC polygonization; pre-generate this threshold first",
                "available_thresholds": available_jrc_thresholds(region, site.site_id),
            },
        }


ANNOTATION_PROVIDERS: dict[str, AnnotationProvider] = {
    provider.source: provider
    for provider in (
        VectorCandidateProvider("osm", "OSM"),
        VectorCandidateProvider("hydrolakes", "HydroLAKES"),
        LocalLabelProvider(),
        EsaAnnotationProvider(),
        JrcAnnotationProvider(),
    )
}


def annotation_provider(source: str) -> AnnotationProvider:
    key = str(source or "").strip().lower()
    provider = ANNOTATION_PROVIDERS.get(key)
    if provider is None:
        raise ValueError(f"unknown annotation source: {source}")
    return provider


def annotation_status(annotation: dict | None) -> str:
    if annotation is None:
        return "missing"
    properties = annotation.get("properties") or {}
    if properties.get("skipped"):
        return "skipped"
    if annotation.get("type") == "FeatureCollection":
        return "available" if annotation.get("features") else "empty"
    if properties.get("empty") or not annotation.get("geometry"):
        return "empty"
    return "available"


def candidate_layer(candidate: Any, source_label: str) -> dict | None:
    if candidate is None:
        return None
    properties = candidate_properties(candidate)
    properties.update(
        {
            "candidate_id": jsonable(candidate.get("candidate_id")),
            "site_id": jsonable(candidate.get("site_id")),
            "source": source_label,
            "intersection_area_km2": jsonable(candidate.get("intersection_area_km2")),
            "site_coverage_ratio": jsonable(candidate.get("site_coverage_ratio")),
        }
    )
    return {
        "source": source_label,
        "geometry": mapping(candidate.geometry),
        "properties": properties,
    }


def candidate_properties(candidate: Any) -> dict:
    value = candidate.get("properties_json")
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
