from __future__ import annotations

import unittest

from scripts.build_site_metadata import site_display_name, suggested_site_name


class SiteDisplayNameTests(unittest.TestCase):
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
