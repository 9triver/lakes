from __future__ import annotations

import unittest

from lake_workbench.http_handler import create_lake_handler


class HandlerFactoryTests(unittest.TestCase):
    def test_each_factory_call_binds_isolated_runtime_dependencies(self) -> None:
        first_catalogs = {"first": object()}
        second_catalogs = {"second": object()}
        first = create_lake_handler(
            catalogs=first_catalogs,
            downloads_by_region={"first": "download-1"},
            patch_exports_by_region={"first": "patch-1"},
            training_runs_by_scope={"first": "train-1"},
            all_patch_exports="all-patch-1",
            default_region_key="first",
            region_service="regions-1",
        )
        second = create_lake_handler(
            catalogs=second_catalogs,
            downloads_by_region={"second": "download-2"},
            patch_exports_by_region={"second": "patch-2"},
            training_runs_by_scope={"second": "train-2"},
            all_patch_exports="all-patch-2",
            default_region_key="second",
            region_service="regions-2",
        )

        self.assertIsNot(first, second)
        self.assertIs(first.catalog, first_catalogs["first"])
        self.assertIs(second.catalog, second_catalogs["second"])
        self.assertEqual(first.downloads, "download-1")
        self.assertEqual(second.downloads, "download-2")
        self.assertEqual(first.region_service, "regions-1")
        self.assertEqual(second.region_service, "regions-2")


if __name__ == "__main__":
    unittest.main()
