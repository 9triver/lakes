"""Metadata building primitives and external source readers."""

from lake_workbench.metadata.geometry import geometry_area_km2, geometry_areas_km2
from lake_workbench.metadata.sources import (
    clean_id,
    clean_text,
    image_date,
    image_tile_name,
    load_external_hydrolakes,
    load_external_osm_water,
    load_sentinel_tile_index,
    spatial_intersections,
)

__all__ = [
    "clean_id",
    "clean_text",
    "geometry_area_km2",
    "geometry_areas_km2",
    "image_date",
    "image_tile_name",
    "load_external_hydrolakes",
    "load_external_osm_water",
    "load_sentinel_tile_index",
    "spatial_intersections",
]
