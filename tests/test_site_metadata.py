from __future__ import annotations

import unittest

from shapely.geometry import box

from scripts.build_site_metadata import site_display_name, suggested_site_name
from lake_workbench.catalog import SiteCatalog, SiteRecord


class SiteDisplayNameTests(unittest.TestCase):
    @staticmethod
    def site_record() -> SiteRecord:
        return SiteRecord(
            site_id="gansu_20307",
            local_directory_id="20307",
            display_name="区域 20307",
            suggested_name=None,
            area_km2=1.0,
            bbox=(100.0, 20.0, 101.0, 21.0),
            center=(100.5, 20.5),
            properties={},
            geometry=box(100.0, 20.0, 101.0, 21.0),
        )

    def test_site_record_has_no_legacy_identity_aliases(self) -> None:
        site = self.site_record()
        self.assertFalse(hasattr(site, "object_id"))
        self.assertFalse(hasattr(site, "lake_id"))

    def test_site_detail_contains_site_geometry_without_annotations(self) -> None:
        site = self.site_record()
        catalog = SiteCatalog.__new__(SiteCatalog)
        catalog._summary_cache = {site.site_id: {"site_id": site.site_id}}
        catalog._detail_cache = {}

        detail = catalog.get_site_detail(site)

        self.assertEqual(detail["geometry"]["type"], "Polygon")
        self.assertNotIn("layers", detail)

    def test_site_list_includes_usable_training_patch_count(self) -> None:
        site = self.site_record()
        catalog = SiteCatalog.__new__(SiteCatalog)
        catalog.sites = [site]
        catalog._summary_cache = {site.site_id: {"site_id": site.site_id}}
        catalog.usable_training_patch_counts = lambda: {site.site_id: 3}

        payload = catalog.list_sites()

        self.assertEqual(payload["items"][0]["usable_training_patch_count"], 3)

    def test_display_name_keeps_directory_identity_first(self) -> None:
        self.assertEqual(site_display_name("20307"), "区域 20307")
        self.assertEqual(site_display_name("17407", "苏干湖"), "区域 17407（苏干湖附近）")

    def test_tiny_incidental_water_name_is_not_used(self) -> None:
        candidates = [
            {
                "name": "仙女滩湖",
                "is_suggested_primary": 1,
                "intersection_area_km2": 0.001051,
                "site_coverage_ratio": 0.000009,
            }
        ]
        self.assertEqual(suggested_site_name(candidates), "")

    def test_spatially_meaningful_name_is_used_as_hint(self) -> None:
        candidates = [
            {
                "name": "苏干湖",
                "is_suggested_primary": 1,
                "intersection_area_km2": 191.2,
                "site_coverage_ratio": 0.226,
            }
        ]
        self.assertEqual(suggested_site_name(candidates), "苏干湖")


if __name__ == "__main__":
    unittest.main()
