"""Raster-to-vector model prediction operations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import transform_bounds, transform_geom as rasterio_transform_geom
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from lake_workbench.geo import boxes_intersect, transform_geom
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.models.runtime import predict_array
from lake_workbench.paths import PROJECT_ROOT


MODEL_VALIDATION_MAX_DIM = max(256, int(os.environ.get("LAKES_MODEL_VALIDATION_MAX_DIM", "2048")))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def predict_water_geojson(
    path: Path,
    model: Any,
    threshold: float = 0.5,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[dict, dict]:
    threshold = float(threshold)
    with rasterio.open(path) as src:
        if src.count < model.in_channels:
            raise ValueError(f"Raster has {src.count} bands, model needs {model.in_channels}: {display_path(path)}")
        window = None
        if bounds is not None:
            source_bounds = bounds
            if src.crs and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                source_bounds = transform_bounds("EPSG:4326", src.crs, *bounds, densify_pts=21)
            raster_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            if not boxes_intersect(source_bounds, raster_bounds):
                return _empty_prediction_stats(threshold)
            left = max(source_bounds[0], raster_bounds[0])
            bottom = max(source_bounds[1], raster_bounds[1])
            right = min(source_bounds[2], raster_bounds[2])
            top = min(source_bounds[3], raster_bounds[3])
            window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()

        out_shape = None
        transform_scale = Affine.identity()
        read_height = int(window.height) if window is not None else src.height
        read_width = int(window.width) if window is not None else src.width
        if max(read_height, read_width) > MODEL_VALIDATION_MAX_DIM:
            scale = MODEL_VALIDATION_MAX_DIM / max(read_height, read_width)
            out_height = max(1, int(round(read_height * scale)))
            out_width = max(1, int(round(read_width * scale)))
            out_shape = (model.in_channels, out_height, out_width)
            transform_scale = Affine.scale(read_width / out_width, read_height / out_height)
        indexes = list(range(1, model.in_channels + 1))
        read_kwargs = {"indexes": indexes, "window": window, "resampling": Resampling.bilinear}
        if out_shape is not None:
            read_kwargs["out_shape"] = out_shape
        image = src.read(**read_kwargs).astype(np.float32)
        mask_kwargs = {"indexes": indexes, "window": window, "resampling": Resampling.nearest}
        if out_shape is not None:
            mask_kwargs["out_shape"] = (model.in_channels, image.shape[1], image.shape[2])
        masks = src.read_masks(**mask_kwargs)
        if image.shape[1] == 0 or image.shape[2] == 0:
            return _empty_prediction_stats(threshold)
        nodata = tuple(src.nodatavals[index - 1] for index in indexes)
        valid = valid_pixel_mask(image, nodata, masks)
        probability = predict_array(model, image, valid=valid)
        predicted = (probability >= threshold) & valid
        transform = window_transform(window, src.transform) if window is not None else src.transform
        transform = transform * transform_scale
        src_crs = src.crs

    valid_pixels = int(valid.sum())
    predicted_pixels = int(predicted.sum())
    raw_geometries = []
    for geom_json, value in shapes(predicted.astype("uint8"), mask=predicted, transform=transform):
        if int(value) != 1:
            continue
        geometry = shape(geom_json)
        if geometry.is_empty:
            continue
        if src_crs and str(src_crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geometry = shape(rasterio_transform_geom(src_crs, "EPSG:4326", mapping(geometry), precision=7))
        raw_geometries.append(make_valid(geometry))

    metric_geometries = []
    for geometry in raw_geometries:
        try:
            metric = make_valid(transform_geom(geometry, "EPSG:4326", "EPSG:3857"))
        except Exception:
            continue
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        metric_geometries.extend(part for part in parts if not part.is_empty and part.area >= 1000)

    features = []
    area_m2 = 0.0
    if metric_geometries:
        merged = make_valid(unary_union(metric_geometries))
        parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
        parts = [make_valid(part.simplify(5, preserve_topology=True)) for part in parts if not part.is_empty]
        parts.sort(key=lambda part: part.area, reverse=True)
        for index, part in enumerate(parts[:500], start=1):
            if part.is_empty:
                continue
            area_m2 += float(part.area)
            geometry = transform_geom(part, "EPSG:3857", "EPSG:4326")
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geometry),
                    "properties": {
                        "source": "model_prediction",
                        "part": index,
                        "threshold": threshold,
                        "area_m2": round(float(part.area), 2),
                    },
                }
            )

    stats = {
        "threshold": threshold,
        "valid_pixels": valid_pixels,
        "predicted_pixels": predicted_pixels,
        "predicted_ratio": predicted_pixels / max(valid_pixels, 1),
        "polygon_count": len(features),
        "area_km2": area_m2 / 1_000_000,
        "mean_probability": float(probability[valid].mean()) if valid_pixels else 0.0,
        "max_probability": float(probability[valid].max()) if valid_pixels else 0.0,
    }
    return {"type": "FeatureCollection", "features": features}, stats


def _empty_prediction_stats(threshold: float) -> tuple[dict, dict]:
    return {"type": "FeatureCollection", "features": []}, {
        "threshold": threshold,
        "valid_pixels": 0,
        "predicted_pixels": 0,
        "predicted_ratio": 0.0,
        "polygon_count": 0,
        "area_km2": 0.0,
        "mean_probability": 0.0,
        "max_probability": 0.0,
    }
