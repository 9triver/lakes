"""Generate one inspectable water annotation from one evidence source."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.validation import make_valid

from lake_workbench.automatic_labels.basemap import (
    aligned_osm_rgb,
    osm_water_mask,
)
from lake_workbench.automatic_labels.spectral import (
    BACKGROUND_LABEL,
    DEFAULT_WATER_THRESHOLD,
    IGNORE_LABEL,
    STABLE_LABEL_MAX_DIMENSION,
    SPECTRAL_ALGORITHM_VERSION,
    WATER_LABEL,
    read_spectral_evidence,
)
from lake_workbench.training.identity import normalized_view_extent
from lake_workbench.utils import (
    clean_optional,
    display_path,
    resolve_data_path,
)


SPECTRAL_WATER = "spectral_water"
SPECTRAL_OSM_CONSENSUS = "spectral_osm_consensus"
DERIVED_LABEL_SOURCES = (
    SPECTRAL_WATER,
    SPECTRAL_OSM_CONSENSUS,
)
SPECTRAL_LABEL_SOURCES = (SPECTRAL_WATER, SPECTRAL_OSM_CONSENSUS)


def generate_derived_label(
    catalog: Any,
    site: Any,
    source: str,
    payload: dict,
    output_dir: Path,
) -> dict:
    """Generate and persist one source-specific water polygon layer."""
    if source not in DERIVED_LABEL_SOURCES:
        raise ValueError(f"Unknown generated annotation source: {source}")
    requested_extent = payload.get("extent") or list(site.bbox)
    view_state = {
        "selected_imagery_asset_id": clean_optional(payload.get("asset_id")) or "",
        "selected_product": clean_optional(payload.get("product")) or "",
        "map": {"extent": requested_extent},
    }
    extent = normalized_view_extent(view_state, digits=7)
    if not extent:
        raise ValueError("A valid WGS84 extent is required")
    extent = _clip_extent_to_site(extent, site.bbox)
    if not extent:
        raise ValueError("The requested annotation extent does not overlap the site")
    bounds = (extent[0], extent[1], extent[2], extent[3])
    imagery = catalog.selected_training_imagery(site, view_state)
    if len(imagery) != 1:
        raise ValueError("Generated annotations require one selected imagery asset")
    image_path = resolve_data_path(imagery[0].get("tci_path", ""), catalog.region)
    if not image_path.exists():
        raise FileNotFoundError(f"Selected imagery does not exist: {image_path}")

    # Generated labels are persisted and later reused for training. Keep their
    # raster grid stable instead of making it depend on the browser viewport.
    max_dimension = _bounded_int(
        payload.get("max_dimension"), STABLE_LABEL_MAX_DIMENSION, 256, 4096
    )
    threshold = _bounded_float(payload.get("threshold"), DEFAULT_WATER_THRESHOLD)
    if source in SPECTRAL_LABEL_SOURCES:
        threshold = max(0.5, threshold)
    safe_path_text = clean_optional(imagery[0].get("safe_path"))
    quality_path = (
        resolve_data_path(safe_path_text, catalog.region)
        if safe_path_text
        else None
    )
    spectral = read_spectral_evidence(
        image_path,
        bounds,
        max_dimension=max_dimension,
        threshold=threshold,
        quality_path=quality_path if quality_path and quality_path.exists() else None,
    )
    details: dict[str, Any]
    if source == SPECTRAL_WATER:
        water_score = spectral.water_score
        valid = spectral.valid
        labels = spectral.labels
        details = {
            "provider": SPECTRAL_ALGORITHM_VERSION,
            "kind": "spectral_consensus",
            "status": "ready",
            "score_name": "water_score",
            "score_is_calibrated_probability": False,
            "processing_mode": "stable_label_grid",
            "processing_max_dimension": max_dimension,
            "bands": spectral.bands,
            **spectral.diagnostics,
        }
    elif source == SPECTRAL_OSM_CONSENSUS:
        osm = aligned_osm_rgb(
            bounds,
            _bounded_int(payload.get("osm_zoom"), 14, 8, 19),
            spectral.crs,
            spectral.transform,
            spectral.shape,
            catalog.region.cache_dir,
        )
        osm_water = osm_water_mask(osm.rgb, osm.valid)
        valid = spectral.valid & osm.valid
        spectral_water = spectral.labels == WATER_LABEL
        spectral_water_valid = valid & spectral_water
        consensus_seed = spectral_water_valid & osm_water
        connected_water = _seeded_spectral_water(
            spectral_water_valid,
            consensus_seed,
        )
        labels = np.full(spectral.shape, IGNORE_LABEL, dtype=np.uint8)
        labels[valid & (spectral.labels == BACKGROUND_LABEL)] = BACKGROUND_LABEL
        labels[connected_water] = WATER_LABEL
        water_score = np.where(connected_water, spectral.water_score, 0.0).astype(
            np.float32
        )
        details = {
            "provider": "osm_blue_water_consensus",
            "kind": "spectral_osm_consensus",
            "status": "ready",
            "score_name": "spectral_water_score",
            "score_is_calibrated_probability": False,
            "processing_mode": "spectral_water_connected_to_osm_seed",
            "connectivity": 8,
            "processing_max_dimension": max_dimension,
            "bands": spectral.bands,
            "spectral_algorithm": SPECTRAL_ALGORITHM_VERSION,
            "spectral_water_pixels": int(np.count_nonzero(spectral_water_valid)),
            "osm_water_pixels": int(np.count_nonzero(valid & osm_water)),
            "consensus_seed_pixels": int(np.count_nonzero(consensus_seed)),
            "consensus_water_pixels": int(np.count_nonzero(connected_water)),
            "spectral_only_pixels": int(
                np.count_nonzero(spectral_water_valid & ~osm_water)
            ),
            "spectral_only_promoted_pixels": int(
                np.count_nonzero(connected_water & ~osm_water)
            ),
            "spectral_only_ignored_pixels": int(
                np.count_nonzero(spectral_water_valid & ~connected_water)
            ),
            "osm": {
                "provider": osm.provider,
                "zoom": osm.zoom,
                "tile_count": osm.tile_count,
                "tile_url": osm.tile_url,
                "water_mask": "standard_osm_blue_palette_v1",
                "attribution": "© OpenStreetMap contributors",
            },
            **spectral.diagnostics,
        }
    water = valid & (labels == WATER_LABEL)
    ignore = valid & (labels == IGNORE_LABEL)
    background = valid & (labels == BACKGROUND_LABEL)
    water_features, water_polygon_count = _label_features(
        water,
        spectral.transform,
        spectral.crs,
        source,
        label_class="water",
        label_value=WATER_LABEL,
    )
    ignore_features, ignore_polygon_count = _label_features(
        ignore,
        spectral.transform,
        spectral.crs,
        source,
        label_class="ignore",
        label_value=IGNORE_LABEL,
    )
    features = water_features + ignore_features
    valid_pixels = int(np.count_nonzero(valid))
    water_pixels = int(np.count_nonzero(water))
    background_pixels = int(np.count_nonzero(background))
    ignore_pixels = int(np.count_nonzero(ignore))
    stats = {
        "pixels": int(water.size),
        "valid_pixels": valid_pixels,
        "water_pixels": water_pixels,
        "background_pixels": background_pixels,
        "ignore_pixels": ignore_pixels,
        "water_ratio": water_pixels / max(valid_pixels, 1),
        "confident_ratio": (water_pixels + background_pixels)
        / max(valid_pixels, 1),
        "polygon_count": water_polygon_count,
        "ignore_polygon_count": ignore_polygon_count,
        "mean_water_score": float(water_score[valid].mean())
        if valid_pixels
        else 0.0,
        "max_water_score": float(water_score[valid].max())
        if valid_pixels
        else 0.0,
    }
    identity = {
        "source": source,
        "site_id": site.site_id,
        "image_path": display_path(image_path),
        "image_size": image_path.stat().st_size,
        "image_mtime_ns": image_path.stat().st_mtime_ns,
        "extent": extent,
        "max_dimension": max_dimension,
        "threshold": threshold,
        "provider": details.get("provider"),
        "algorithm": details.get("algorithm") or details.get("provider"),
        "zoom": details.get("osm", {}).get("zoom"),
    }
    label_id = "derived_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    properties = {
        "source": source,
        "label_id": label_id,
        "site_id": site.site_id,
        "region": catalog.region.key,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "imagery": imagery[0],
        "extent": extent,
        "grid": {
            "width": spectral.shape[1],
            "height": spectral.shape[0],
            "crs": str(spectral.crs),
        },
        "threshold": threshold,
        "details": details,
        "stats": stats,
    }
    label = {
        "type": "FeatureCollection",
        "properties": properties,
        "features": features,
    }
    source_dir = output_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{label_id}.geojson"
    path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    return {
        "label_id": label_id,
        "source": source,
        "label": label,
        "stats": stats,
        "details": details,
    }


def _seeded_spectral_water(
    spectral_water: np.ndarray,
    consensus_seed: np.ndarray,
) -> np.ndarray:
    """Promote spectral-water components that contain an OSM consensus seed."""
    if spectral_water.shape != consensus_seed.shape:
        raise ValueError("spectral water and consensus seed masks must have the same shape")
    if not np.any(consensus_seed):
        return np.zeros(spectral_water.shape, dtype=bool)
    components = np.empty(spectral_water.shape, dtype=np.int32)
    ndimage.label(
        spectral_water,
        structure=np.ones((3, 3), dtype=np.uint8),
        output=components,
    )
    seeded_components = np.unique(components[consensus_seed])
    return spectral_water & np.isin(components, seeded_components)


def _label_features(
    selected: np.ndarray,
    transform: Any,
    crs: Any,
    source: str,
    *,
    label_class: str,
    label_value: int,
) -> tuple[list[dict], int]:
    geometries = []
    pixel_area = max(abs(float(transform.a * transform.e)), 1e-12)
    for geometry_payload, value in shapes(
        selected.astype("uint8"), mask=selected, transform=transform
    ):
        if int(value) != 1:
            continue
        geometry = make_valid(shape(geometry_payload))
        if geometry.is_empty or geometry.area < pixel_area * 4:
            continue
        geometries.append(geometry)
    if not geometries:
        return [], 0
    merged = make_valid(unary_union(geometries))
    tolerance = max(abs(float(transform.a)), abs(float(transform.e))) * 0.35
    merged = make_valid(merged.simplify(tolerance, preserve_topology=True))
    parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
    parts.sort(key=lambda part: part.area, reverse=True)
    merged_wgs84 = _to_wgs84(merged, crs)
    return [
        {
            "type": "Feature",
            "geometry": mapping(merged_wgs84),
            "properties": {
                "source": source,
                "label_class": label_class,
                "label_value": label_value,
                "part_count": len(parts),
            },
        }
    ], len(parts)


def _to_wgs84(geometry: Any, crs: Any) -> Any:
    if str(crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        return geometry
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return make_valid(shapely_transform(transformer.transform, geometry))


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _clip_extent_to_site(
    extent: list[float], site_bbox: tuple[float, float, float, float]
) -> list[float]:
    """Keep generated labels inside the selected observation site.

    OpenLayers can report its initial world view before the asynchronous site
    fit completes. Letting that view drive scene-adaptive statistics makes the
    same image produce a different classification for the wrong area.
    """
    west = max(float(extent[0]), float(site_bbox[0]))
    south = max(float(extent[1]), float(site_bbox[1]))
    east = min(float(extent[2]), float(site_bbox[2]))
    north = min(float(extent[3]), float(site_bbox[3]))
    if east <= west or north <= south:
        return []
    return [west, south, east, north]
