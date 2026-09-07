from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_region_data_root_redirects_region_owned_paths(self) -> None:
        root = Path("/tmp/lakes-seasonal-regions")
        with patch.dict("os.environ", {"LAKES_REGION_DATA_ROOT": str(root)}):
            regions, _ = load_region_configs()

        gansu = regions["gansu"]
        self.assertEqual(gansu.data_dir, root / "gansu" / "raw")
        self.assertEqual(gansu.processed_dir, root / "gansu" / "processed")
        self.assertEqual(
            gansu.local_imagery_root,
            root / "gansu" / "raw" / "local_imagery",
        )
        self.assertFalse(gansu.shared_data_dir.is_relative_to(root))
