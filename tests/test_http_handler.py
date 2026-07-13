from __future__ import annotations

import unittest
from types import SimpleNamespace

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

    def test_request_context_binds_default_region(self) -> None:
        catalog = SimpleNamespace(region=SimpleNamespace(key="first"))
        handler_type = create_lake_handler(
            catalogs={"first": catalog},
            downloads_by_region={"first": "download-1"},
            patch_exports_by_region={"first": "patch-1"},
            training_runs_by_scope={"first": "train-1", "all": "train-all"},
            all_patch_exports="patch-all",
            default_region_key="first",
            region_service="regions",
        )
        handler = handler_type.__new__(handler_type)

        self.assertEqual(handler._bind_request_context("/api/lakes"), "/api/sites")
        self.assertIs(handler.catalog, catalog)
        self.assertEqual(handler.downloads, "download-1")
        self.assertEqual(handler.patch_exports, "patch-1")
        self.assertEqual(handler.training_runs, "train-1")

    def test_request_context_normalizes_region_and_all_paths(self) -> None:
        first = SimpleNamespace(region=SimpleNamespace(key="first"))
        second = SimpleNamespace(region=SimpleNamespace(key="second"))
        handler_type = create_lake_handler(
            catalogs={"first": first, "second": second},
            downloads_by_region={"first": "download-1", "second": "download-2"},
            patch_exports_by_region={"first": "patch-1", "second": "patch-2"},
            training_runs_by_scope={"first": "train-1", "second": "train-2", "all": "train-all"},
            all_patch_exports="patch-all",
            default_region_key="first",
            region_service="regions",
        )
        handler = handler_type.__new__(handler_type)

        self.assertEqual(handler._bind_request_context("/api/regions/second/lakes"), "/api/sites")
        self.assertEqual(handler._bind_request_context("/api/regions/second/sites"), "/api/sites")
        self.assertIs(handler.catalog, second)
        self.assertEqual(handler.downloads, "download-2")
        self.assertEqual(handler.training_runs, "train-2")
        self.assertEqual(handler._bind_request_context("/api/regions/all/training-runs"), "/api/all/training-runs")
        self.assertIs(handler.catalog, first)
        self.assertEqual(handler.patch_exports, "patch-all")
        self.assertEqual(handler.training_runs, "train-all")


if __name__ == "__main__":
    unittest.main()
