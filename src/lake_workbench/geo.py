"""Shared geometry transformations and smoothing helpers."""

from __future__ import annotations

from typing import Any

from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid


def transform_geom(geom, src_crs: str, dst_crs: str):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def site_aoi_geometry(site: Any, padding: float = 0.8):
    metric = transform_geom(site.geometry, "EPSG:4326", "EPSG:3857")
    minx, miny, maxx, maxy = metric.bounds
    base = max(maxx - minx, maxy - miny)
    buffer_m = max(base * float(padding), 500)
    return transform_geom(metric.buffer(buffer_m), "EPSG:3857", "EPSG:4326")


def geometry_coverage_ratio(target_geom, cover_geom) -> float:
    if target_geom is None or cover_geom is None or target_geom.is_empty or cover_geom.is_empty:
        return 0.0
    try:
        target_m = transform_geom(make_valid(target_geom), "EPSG:4326", "EPSG:3857")
        cover_m = transform_geom(make_valid(cover_geom), "EPSG:4326", "EPSG:3857")
        area = target_m.area
        return float(target_m.intersection(cover_m).area / area) if area > 0 else 0.0
    except Exception:
        return 0.0


def product_geometry(product: dict):
    footprint = product.get("footprint")
    if not footprint:
        return None
    try:
        geom = shape(footprint)
    except Exception:
        return None
    return None if geom.is_empty else geom


def smooth_water_geometry(geom):
    metric = make_valid(transform_geom(geom, "EPSG:4326", "EPSG:3857"))
    opened = metric.buffer(-5, resolution=4, join_style=1).buffer(5, resolution=4, join_style=1)
    if opened.is_empty:
        opened = metric
    closed = opened.buffer(8, resolution=4, join_style=1).buffer(-8, resolution=4, join_style=1)
    if closed.is_empty:
        closed = opened
    simplified = make_valid(closed.simplify(2, preserve_topology=True))

    def smooth_polygon(poly):
        exterior = chaikin_ring(poly.exterior.coords, iterations=2)
        holes = [chaikin_ring(ring.coords, iterations=2) for ring in poly.interiors]
        return make_valid(Polygon(exterior, holes))

    if isinstance(simplified, Polygon):
        parts = [simplified]
    elif isinstance(simplified, MultiPolygon):
        parts = list(simplified.geoms)
    else:
        return transform_geom(simplified, "EPSG:3857", "EPSG:4326")
    smoothed = [out for part in parts if part.area >= 500 if not (out := smooth_polygon(part)).is_empty]
    if not smoothed:
        smoothed = parts
    result = smoothed[0] if len(smoothed) == 1 else MultiPolygon(smoothed)
    return transform_geom(result, "EPSG:3857", "EPSG:4326")


def chaikin_ring(coords, iterations: int = 2):
    points = list(coords)
    if len(points) < 4:
        return points
    if points[0] == points[-1]:
        points = points[:-1]
    for _ in range(iterations):
        smoothed = []
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            smoothed.append((0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2))
            smoothed.append((0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2))
        points = smoothed
    points.append(points[0])
    return points


def padded_bounds(
    bbox_wgs84: tuple[float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox_wgs84
    width = xmax - xmin
    height = ymax - ymin
    pad_x = max(width * padding, 0.005)
    pad_y = max(height * padding, 0.005)
    return (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)


def boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]
