"""Open a region's raster inventory only when a regional operation needs it."""

import csv
import threading

import pyogrio

from lake_workbench.catalog import SiteCatalog
from lake_workbench.training.patch_catalog import TrainingPatchCatalogMixin


class DeferredSiteCatalog:
    def __init__(self, region, default_region_key):
        self.region = region
        self.default_region_key = default_region_key
        self._catalog = None
        self._lock = threading.Lock()
        self._inventory_summary = None
        self._metadata_sites = None
        self._metadata_error = None

    @property
    def load_error(self):
        if self._catalog is not None:
            return self._catalog.load_error
        return self._metadata_error or (None if self.region.site_metadata.exists() else "Region site metadata is not prepared")

    def site_count(self):
        if self._catalog is not None:
            return self._catalog.site_count()
        if not self.region.site_metadata.exists():
            return 0
        try:
            return int(pyogrio.read_info(self.region.site_metadata, layer="sites").get("features") or 0)
        except Exception as error:
            self._metadata_error = f"{type(error).__name__}: {error}"
            return 0

    def imagery_inventory_summary(self):
        if self._catalog is not None:
            return self._catalog.imagery_inventory_summary()
        if self._inventory_summary is None:
            tiles, products = set(), 0
            for path in (self.region.tci_index, self.region.user_sentinel_index):
                if not path.exists():
                    continue
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        tiles.add(str(row.get("tile", "")).upper().removeprefix("T"))
                        products += 1
            from lake_workbench.imagery.inventory_sources import load_active_imagery
            active = load_active_imagery(self.region.active_imagery)
            self._inventory_summary = {"tci_tile_count": len(tiles), "product_count": products,
                                       "active_imagery_count": len(active),
                                       "active_tile_count": len({key.rsplit(":", 1)[-1] for key in active})}
        return self._inventory_summary

    def get_site(self, site_id):
        if self._catalog is not None:
            return self._catalog.get_site(site_id)
        with self._lock:
            if self._metadata_sites is None:
                records = SiteCatalog._load_sites(self) if self.region.site_metadata.exists() else []
                self._metadata_sites = {site.site_id: site for site in records}
        return self._metadata_sites.get(site_id)

    def _logical_patch_summary(self, row, manifest_path):
        return TrainingPatchCatalogMixin._logical_patch_summary(self, row, manifest_path)

    def __getattr__(self, name):
        with self._lock:
            if self._catalog is None:
                self._catalog = SiteCatalog(self.region, self.default_region_key)
        return getattr(self._catalog, name)
