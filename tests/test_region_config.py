from __future__ import annotations

import unittest

from lake_workbench.regions.config import load_region_configs


class RegionConfigTests(unittest.TestCase):
    def test_global_sources_use_one_configured_shared_root(self) -> None:
        regions, _ = load_region_configs()
        shared_roots = {region.shared_data_dir for region in regions.values()}
        hydrolakes = {region.hydrolakes for region in regions.values()}
        sentinel_indexes = {region.sentinel_tile_index_paths[0] for region in regions.values()}

        self.assertEqual(len(shared_roots), 1)
        self.assertEqual(len(hydrolakes), 1)
        self.assertEqual(len(sentinel_indexes), 1)

    def test_region_specific_osm_remains_under_region_raw_data(self) -> None:
        regions, _ = load_region_configs()
        for region in regions.values():
            self.assertTrue(region.osm_water.is_relative_to(region.data_dir))
            self.assertFalse(region.hydrolakes.is_relative_to(region.data_dir))
