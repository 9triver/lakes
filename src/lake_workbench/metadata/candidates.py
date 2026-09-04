"""Match observation sites with external vector water candidates."""

from __future__ import annotations

import json

from lake_workbench.metadata.geometry import geometry_area_km2, json_value, numeric_area, polygonal_geometry
from lake_workbench.metadata.sources import clean_text, spatial_intersections


def build_vector_candidates(site_id: str, site_geometry, osm, hydrolakes) -> list[dict]:
    candidates: list[dict] = []
    for source, frame in (("osm", osm), ("hydrolakes", hydrolakes)):
        if frame.empty:
            continue
        selected = spatial_intersections(frame, site_geometry)
        site_area = geometry_area_km2(site_geometry)
        for row in selected.itertuples():
            geometry = polygonal_geometry(row.geometry)
            overlap = polygonal_geometry(geometry.intersection(site_geometry))
            if overlap.is_empty:
                continue
            values = row._asdict()
            source_id = clean_text(values.get("source_feature_id")) or str(row.Index)
            name = candidate_name(values)
            feature_area = numeric_area(values.get("area_km2")) or geometry_area_km2(geometry)
            overlap_area = geometry_area_km2(overlap)
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
