"""Geometry normalization and area calculations for site metadata."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer


_METRIC_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def geo_frame(rows: list[dict]) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame(
            {"geometry": gpd.GeoSeries([], crs="EPSG:4326")},
            geometry="geometry",
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def polygonal_geometry(geometry) -> Any:
    if geometry is None or geometry.is_empty:
        return MultiPolygon()
    geometry = make_valid(geometry)
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, MultiPolygon):
                polygons.extend(part.geoms)
        return MultiPolygon([part for part in polygons if not part.is_empty])
    return MultiPolygon()


def metric_geometry(geometry):
    return shapely_transform(_METRIC_TRANSFORMER.transform, geometry)


def geometry_area_km2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(metric_geometry(geometry).area / 1_000_000)


def geometry_areas_km2(geometries) -> pd.Series:
    """Calculate many WGS84 geometry areas with one vectorized reprojection."""
    series = gpd.GeoSeries(geometries, crs="EPSG:4326")
    return series.to_crs("EPSG:3857").area / 1_000_000


def majority_coverage_geometry(geometries: list[Any], union, fraction: float = 0.8, size: int = 256):
    """Return the part covered by at least ``fraction`` of the input footprints."""
    if len(geometries) == 1:
        return geometries[0]
    bounds = union.bounds
    width = max(bounds[2] - bounds[0], 1e-9)
    height = max(bounds[3] - bounds[1], 1e-9)
    if width >= height:
        columns = size
        rows = max(1, int(round(size * height / width)))
    else:
        rows = size
        columns = max(1, int(round(size * width / height)))
    transform = from_bounds(*bounds, columns, rows)
    counts = np.zeros((rows, columns), dtype="uint16")
    for geometry in geometries:
        counts += rasterize(
            [(mapping(geometry), 1)],
            out_shape=(rows, columns),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
    threshold = max(1, int(np.ceil(len(geometries) * fraction)))
    core_mask = counts >= threshold
    parts = [
        shape(geometry)
        for geometry, value in shapes(core_mask.astype("uint8"), mask=core_mask, transform=transform)
        if int(value) == 1
    ]
    return polygonal_geometry(unary_union(parts).intersection(union)) if parts else union


def numeric_area(value) -> float:
    try:
        return float(value) if pd.notna(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
