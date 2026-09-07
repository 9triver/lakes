"""Region configuration and normalized data paths."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lake_workbench.paths import PROJECT_ROOT


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "regions.toml"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class RegionConfig:
    key: str
    name: str
    data_dir: Path
    processed_dir: Path
    cache_dir: Path
    shared_data_dir: Path
    bounds: tuple[float, float, float, float] | None = None
    geofabrik: str = ""
    external_raster_mode: str = "clip"
    local_imagery_root: Path | None = None
    esa_tiles: tuple[str, ...] = field(default_factory=tuple)
    jrc_tiles: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tci_index(self) -> Path:
        return self.data_dir / "sentinel_products" / "selected_best_coverage_tci_valid.csv"

    @property
    def sentinel_tile_index_paths(self) -> list[Path]:
        return [
            self.shared_data_dir / "sentinel_2_tiles" / "sentinel_2_index.geojson",
            self.shared_data_dir / "sentinel_2_tiles" / "sentinel_2_index_shapefile.shp",
        ]

    @property
    def osm_water(self) -> Path:
        return self.data_dir / "external_water" / "osm" / "water_raw.gpkg"

    @property
    def site_metadata(self) -> Path:
        return self.processed_dir / "site_metadata.gpkg"

    @property
    def hydrolakes(self) -> Path:
        return self.shared_data_dir / "external_water" / "hydrolakes" / "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp"

    @property
    def esa_water_mask(self) -> Path:
        return self.data_dir / "external_water" / "esa_worldcover" / "esa_water_mask.tif"

    @property
    def esa_worldcover_clip(self) -> Path:
        return self.data_dir / "external_water" / "esa_worldcover" / "esa_worldcover_clip.tif"

    @property
    def esa_worldcover_dir(self) -> Path:
        return self.data_dir / "external_water" / "esa_worldcover"

    @property
    def jrc_occurrence(self) -> Path:
        return self.data_dir / "external_water" / "jrc_gsw" / "jrc_occurrence_clip.tif"

    @property
    def jrc_seasonality(self) -> Path:
        return self.data_dir / "external_water" / "jrc_gsw" / "jrc_seasonality_clip.tif"

    @property
    def jrc_gsw_dir(self) -> Path:
        return self.data_dir / "external_water" / "jrc_gsw"

    @property
    def image_cache_dir(self) -> Path:
        return self.cache_dir / "image_clips"

    @property
    def tile_cache_dir(self) -> Path:
        return self.cache_dir / "tiles"

    @property
    def jrc_polygon_dir(self) -> Path:
        return self.processed_dir / "jrc_polygons"

    @property
    def esa_polygon_dir(self) -> Path:
        return self.processed_dir / "esa_polygons"

    @property
    def user_sentinel_index(self) -> Path:
        return self.processed_dir / "sentinel_products.csv"

    @property
    def active_imagery(self) -> Path:
        return self.processed_dir / "active_imagery.json"

    @property
    def training_samples(self) -> Path:
        return self.processed_dir / "training_samples.csv"

    @property
    def training_label_dir(self) -> Path:
        return self.processed_dir / "training_labels"

    @property
    def sentinel_download_dir(self) -> Path:
        return self.data_dir / "sentinel_products" / "products"

def load_region_configs(config_path: Path | None = None) -> tuple[dict[str, RegionConfig], str]:
    path = config_path or project_path(os.environ.get("LAKES_REGIONS_CONFIG", DEFAULT_CONFIG_PATH))
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    shared_data_dir = project_path(payload.get("shared_data_dir", "data/shared"))
    data_root_value = os.environ.get("LAKES_REGION_DATA_ROOT", "").strip()
    data_root = project_path(data_root_value) if data_root_value else None
    regions_payload: dict[str, Any] = payload.get("regions", {})
    regions = {
        key: region_from_mapping(key, value, shared_data_dir, data_root=data_root)
        for key, value in regions_payload.items()
    }
    default_key = os.environ.get("LAKES_DEFAULT_REGION") or payload.get("default") or next(iter(regions), "")
    if default_key not in regions:
        raise ValueError(f"Unknown default region: {default_key}")
    return regions, default_key


def region_from_mapping(
    key: str,
    value: dict[str, Any],
    shared_data_dir: Path | None = None,
    *,
    data_root: Path | None = None,
) -> RegionConfig:
    bounds = value.get("bounds")
    local_imagery_root = value.get("local_imagery_root")
    if data_root is not None:
        region_root = data_root / key
        data_dir = region_root / "raw"
        processed_dir = region_root / "processed"
        local_imagery_root = region_root / "raw" / "local_imagery"
    else:
        data_dir = project_path(value["data_dir"])
        processed_dir = project_path(value["processed_dir"])
        local_imagery_root = project_path(local_imagery_root) if local_imagery_root else data_dir / "local_imagery"
    return RegionConfig(
        key=key,
        name=str(value["name"]),
        data_dir=data_dir,
        processed_dir=processed_dir,
        cache_dir=project_path(value.get("cache_dir", f"data/cache/{key}")),
        shared_data_dir=shared_data_dir or project_path("data/shared"),
        bounds=tuple(float(item) for item in bounds) if bounds else None,
        geofabrik=str(value.get("geofabrik", "")),
        external_raster_mode=str(value.get("external_raster_mode", "clip")),
        local_imagery_root=(
            Path(local_imagery_root)
            if isinstance(local_imagery_root, Path)
            else project_path(local_imagery_root)
        )
        if local_imagery_root
        else None,
        esa_tiles=tuple(str(item) for item in value.get("esa_tiles", [])),
        jrc_tiles=tuple(str(item) for item in value.get("jrc_tiles", [])),
    )
