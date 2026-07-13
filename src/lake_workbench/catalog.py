"""Regional observation-site catalog and compatibility APIs."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pyogrio
from shapely.geometry import box, mapping

from lake_workbench.imagery.inventory import ImageryInventoryMixin
from lake_workbench.imagery.rendering import ImageryRenderingMixin
from lake_workbench.local_labels import LocalLabelCatalogMixin
from lake_workbench.models.validation import ModelValidationMixin
from lake_workbench.regions.config import RegionConfig
from lake_workbench.sentinel.catalog import SentinelCatalogMixin
from lake_workbench.training.catalog import TrainingCatalogMixin
from lake_workbench.utils import (
    area_in_bucket,
    clean_optional,
    count_values,
    display_path,
    jsonable,
    metadata_tiles,
    parse_float,
)
from lake_workbench.water.annotations import WaterAnnotationsMixin


@dataclass
class SiteRecord:
    site_id: str
    local_directory_id: str
    display_name: str
    suggested_name: str | None
    area_km2: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    properties: dict
    geometry: object

    @property
    def object_id(self) -> str:
        """Compatibility alias for code written before sites became explicit."""
        return self.site_id

    @property
    def lake_id(self) -> str:
        return self.site_id

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def source(self) -> str:
        return "local_imagery"


class SiteCatalog(
    ImageryInventoryMixin,
    ImageryRenderingMixin,
    LocalLabelCatalogMixin,
    SentinelCatalogMixin,
    ModelValidationMixin,
    TrainingCatalogMixin,
    WaterAnnotationsMixin,
):
    def __init__(self, region: RegionConfig, default_region_key: str) -> None:
        self.region = region
        self.default_region_key = default_region_key
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self.base_tci_by_tile = self._load_tci_index()
        self.tci_by_tile = dict(self.base_tci_by_tile)
        self.user_tci_rows = self._load_user_tci_rows()
        self.active_imagery = self._load_active_imagery()
        self._rebuild_effective_tci()
        self.tci_footprints = self._load_tci_footprints()
        self.sentinel_tile_index = self._load_sentinel_tile_index()
        self._valid_ratio_cache: dict[str, float] = {}
        self.sites = self._load_sites()
        self.lakes = self.sites
        self._site_lookup = self._build_site_lookup()
        self._lake_lookup = self._site_lookup
        self.external_water_features = self._load_external_water_features()
        self._summary_cache = {site.site_id: self._build_summary(site) for site in self.sites}
        self._detail_cache: dict[str, dict] = {}

    def _load_sites(self) -> list[SiteRecord]:
        if not self.region.site_metadata.exists():
            self.load_error = f"Region site metadata is not prepared: missing {display_path(self.region.site_metadata)}"
            return []
        data = pyogrio.read_dataframe(self.region.site_metadata, layer="sites").to_crs("EPSG:4326")
        records = []
        for _, row in data.sort_values("local_directory_id").reset_index(drop=True).iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            properties = {
                key: jsonable(row.get(key))
                for key in data.columns
                if key != "geometry"
            }
            records.append(
                SiteRecord(
                    site_id=str(row["site_id"]),
                    local_directory_id=str(row["local_directory_id"]),
                    display_name=str(row["display_name"]),
                    suggested_name=clean_optional(row.get("suggested_name")),
                    area_km2=float(row["coverage_area_km2"]),
                    bbox=(
                        float(row["bbox_west"]),
                        float(row["bbox_south"]),
                        float(row["bbox_east"]),
                        float(row["bbox_north"]),
                    ),
                    center=(float(row["center_lon"]), float(row["center_lat"])),
                    properties=properties,
                    geometry=geometry,
                )
            )
        return records

    def _load_external_water_features(self):
        if not self.region.site_metadata.exists():
            return None
        try:
            return pyogrio.read_dataframe(
                self.region.site_metadata,
                layer="external_water_features",
            ).to_crs("EPSG:4326")
        except Exception:
            return None

    def _build_site_lookup(self) -> dict[str, SiteRecord]:
        lookup = {}
        for site in self.sites:
            for key in (site.site_id, site.local_directory_id):
                text = clean_optional(key)
                if text:
                    lookup[text] = site
        return lookup

    def list_sites(
        self,
        query: str = "",
        limit: int = 200,
        offset: int = 0,
        filters: dict | None = None,
    ) -> dict:
        query = query.strip().lower()
        filtered = self.sites
        if query:
            filtered = [
                site
                for site in filtered
                if query in site.site_id.lower()
                or query in site.local_directory_id.lower()
                or query in site.display_name.lower()
                or query in str(site.suggested_name or "").lower()
            ]
        filtered = self._apply_filters(filtered, filters or {})
        page = filtered[offset : offset + limit]
        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "facets": self._facets(filtered),
            "items": [self._summary(site) for site in page],
        }

    def list_lakes(self, *args, **kwargs) -> dict:
        return self.list_sites(*args, **kwargs)

    def _apply_filters(self, sites: list[SiteRecord], filters: dict) -> list[SiteRecord]:
        result = sites
        flag_fields = {
            "has_osm": "osm_feature_count",
            "has_hydrolakes": "hydrolakes_feature_count",
            "has_local_labels": "label_feature_count",
        }
        for parameter, field in flag_fields.items():
            value = clean_optional(filters.get(parameter))
            if value in {"true", "false"}:
                expected = value == "true"
                result = [site for site in result if (int(site.properties.get(field) or 0) > 0) is expected]

        has_tci = clean_optional(filters.get("has_tci"))
        if has_tci in {"true", "false"}:
            expected = has_tci == "true"
            result = [site for site in result if self._has_tci(site) is expected]

        has_name = clean_optional(filters.get("has_name"))
        if has_name in {"true", "false"}:
            expected = has_name == "true"
            result = [site for site in result if bool(site.suggested_name) is expected]

        area_bucket = clean_optional(filters.get("area_bucket"))
        if area_bucket and area_bucket != "all":
            result = [site for site in result if area_in_bucket(site.area_km2, area_bucket)]
        min_area = parse_float(filters.get("min_area"))
        max_area = parse_float(filters.get("max_area"))
        if min_area is not None:
            result = [site for site in result if site.area_km2 >= min_area]
        if max_area is not None:
            result = [site for site in result if site.area_km2 <= max_area]
        return result

    def _facets(self, sites: list[SiteRecord]) -> dict:
        return {
            "has_name": count_values(bool(site.suggested_name) for site in sites),
            "sentinel_tiles": count_values(
                tile
                for site in sites
                for tile in metadata_tiles(site.properties.get("sentinel_tiles"))
            ),
        }

    def get_site(self, site_key: str) -> SiteRecord | None:
        return self._site_lookup.get(site_key)

    def get_lake(self, lake_key: str) -> SiteRecord | None:
        return self.get_site(lake_key)

    def get_site_detail(self, site: SiteRecord) -> dict:
        if site.site_id in self._detail_cache:
            return self._detail_cache[site.site_id]
        detail = {
            **self._summary(site),
            "properties": dict(site.properties),
            "layers": {
                "osm": self._osm_layer(site),
                "hydrolakes": self._match_hydrolakes(site),
                "esa": None,
                "jrc": None,
            },
            "geometry": mapping(site.geometry),
        }
        self._detail_cache[site.site_id] = detail
        return detail

    def get_lake_detail(self, lake: SiteRecord) -> dict:
        return self.get_site_detail(lake)

    def water_candidates_for_site(self, site: SiteRecord, source: str = ""):
        frame = self.external_water_features
        if frame is None or frame.empty:
            return frame
        selected = frame[frame["site_id"] == site.site_id]
        if source:
            selected = selected[selected["source"] == source]
        return selected.copy()

    def suggested_water_candidate(self, site: SiteRecord, source: str):
        candidates = self.water_candidates_for_site(site, source)
        if candidates is None or candidates.empty:
            return None
        return candidates.sort_values(
            ["is_suggested_primary", "intersection_area_km2", "feature_coverage_ratio"],
            ascending=[False, False, False],
        ).iloc[0]

    def _summary(self, site: SiteRecord) -> dict:
        return dict(self._summary_cache.get(site.site_id) or self._build_summary(site))

    def _build_summary(self, site: SiteRecord) -> dict:
        tiles = metadata_tiles(site.properties.get("sentinel_tiles"))
        return {
            "site_id": site.site_id,
            "local_directory_id": site.local_directory_id,
            "display_name": site.display_name,
            "suggested_name": site.suggested_name,
            "coverage_area_km2": site.area_km2,
            "core_area_km2": site.properties.get("core_area_km2"),
            "center": site.center,
            "bbox": site.bbox,
            "tiles": tiles,
            "image_count": int(site.properties.get("image_count") or 0),
            "label_asset_count": int(site.properties.get("label_asset_count") or 0),
            "label_feature_count": int(site.properties.get("label_feature_count") or 0),
            "external_feature_count": int(site.properties.get("external_feature_count") or 0),
            "has_imagery": int(site.properties.get("image_count") or 0) > 0,
            "has_tci": self._has_tci(site),
            # Compatibility fields for persisted training data and legacy clients.
            "lake_id": site.site_id,
            "object_id": site.site_id,
            "name": site.display_name,
            "area_km2": site.area_km2,
            "source": "local_imagery",
        }

    def _has_tci(self, site: SiteRecord) -> bool:
        if int(site.properties.get("image_count") or 0) > 0:
            return True
        return any(item["geometry"].intersects(box(*site.bbox)) for item in self.tci_footprints)


# Compatibility names while scripts and legacy clients migrate to site terminology.
LakeRecord = SiteRecord
LakeCatalog = SiteCatalog
