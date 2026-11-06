from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lake_workbench.training.review_pages import workspace_review_payload
from lake_workbench.training import runner
from lake_workbench.regions.deferred_catalog import DeferredSiteCatalog


class ReviewPageTests(unittest.TestCase):
    def setUp(self):
        self.rows = {str(i): {"logical_patch_id": str(i), "sample_id": "sample",
                            "site_id": "site", "water_pixels": str(i % 3),
                            "include": "true", "review_status": "excluded" if i % 2 else "included"}
                     for i in range(1100)}
        store = Mock()
        store.ensure_workspace_logical_patch_manifest.return_value = Path("unused/manifest.csv")
        store._workspace_logical_index.return_value = self.rows
        catalog = Mock()
        catalog.region = SimpleNamespace(key="test", name="Test")
        catalog.get_site.return_value = SimpleNamespace(display_name="Site")
        catalog._logical_patch_summary.side_effect = lambda row, manifest: row
        registry = SimpleNamespace(root=Mock())
        registry.root.glob.return_value = []
        handler_type = type("Handler", (), {"workspace_store": store, "dataset_registry": registry})
        self.handler = handler_type()
        self.handler.catalog = catalog
        self.handler.workspace_id = "default"
        self.catalog = catalog

    def test_page_limit_is_bounded_and_totals_cover_the_full_scope(self):
        result = workspace_review_payload(self.handler, {"limit": ["2000"], "offset": ["18"]})
        self.assertEqual(len(result["items"]), 1000)
        self.assertEqual(result["items"][0]["logical_patch_id"], "18")
        self.assertEqual(result["total"], 1100)
        self.assertEqual(result["counts"]["included"], 550)
        self.assertEqual(self.catalog._logical_patch_summary.call_count, 1000)

    def test_site_summary_never_materializes_patch_previews(self):
        result = workspace_review_payload(self.handler, {"view": ["sites"]})
        self.assertEqual(result["items"][0]["patchCount"], 1100)
        self.assertEqual(result["items"][0]["includedCount"], 550)
        self.catalog._logical_patch_summary.assert_not_called()

    def test_server_filters_water_and_review_status_before_paging(self):
        result = workspace_review_payload(self.handler, {"limit": ["18"], "include": ["included"], "water": ["water"]})
        self.assertTrue(all(int(row["logical_patch_id"]) % 2 == 0 and int(row["water_pixels"]) > 0 for row in result["items"]))
        self.assertGreater(result["total"], len(result["items"]))


class RequestedSampleTests(unittest.TestCase):
    def test_explicit_batch_does_not_export_other_workspace_samples(self):
        store = Mock()
        store.transaction.side_effect = lambda: nullcontext()
        store.ensure_workspace_logical_patch_manifest.return_value = Path("unused/manifest.csv")
        samples = [{"sample_id": "a"}, {"sample_id": "b"}]
        with patch.object(runner, "read_csv_records", side_effect=[[], samples]), patch.object(runner, "build_logical_patches", return_value={}) as build, patch.object(runner, "_finalize_patch_export", return_value={}):
            runner._run_patch_export_locked("chongqing", {"sample_ids": ["a"]}, store, "default")
        self.assertEqual(build.call_args.kwargs["sample_ids"], {"a"})
        self.assertEqual(build.call_args.kwargs["samples"], [samples[0]])

    def test_unknown_sample_fails_before_export(self):
        store = Mock()
        store.transaction.side_effect = lambda: nullcontext()
        store.ensure_workspace_logical_patch_manifest.return_value = Path("unused/manifest.csv")
        with patch.object(runner, "read_csv_records", side_effect=[[], [{"sample_id": "a"}]]), patch.object(runner, "build_logical_patches") as build:
            with self.assertRaises(ValueError):
                runner._run_patch_export_locked("chongqing", {"sample_ids": ["missing"]}, store, "default")
        build.assert_not_called()


class DeferredCatalogTests(unittest.TestCase):
    def test_missing_metadata_summary_does_not_open_raster_inventory(self):
        region = SimpleNamespace(site_metadata=Mock())
        region.site_metadata.exists.return_value = False
        with patch("lake_workbench.regions.deferred_catalog.SiteCatalog") as factory:
            catalog = DeferredSiteCatalog(region, "test")
            self.assertEqual(catalog.site_count(), 0)
            self.assertIsNotNone(catalog.load_error)
            factory.assert_not_called()
            catalog.some_regional_operation()
            catalog.some_regional_operation()
            factory.assert_called_once_with(region, "test")


class BatchWaterTests(unittest.TestCase):
    def test_threshold_keeps_exactly_one_percent_and_preserves_legacy(self):
        spec = importlib.util.spec_from_file_location("five_site_worker", Path(__file__).resolve().parents[1] / "scripts/chongqing_five_site_batches.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        rows = [{"logical_patch_id": key, "sample_id": "sample", "water_pixels": water, "valid_pixels": valid}
                for key, water, valid in [("legacy", "0", "100"), ("low", "99", "10000"),
                                         ("edge", "100", "10000"), ("invalid", "0", "0")]]
        new, low, kept = worker.filter_new_rows(rows, {"sample"}, {"legacy"})
        self.assertEqual({row["logical_patch_id"] for row in new}, {"low", "edge", "invalid"})
        self.assertEqual({row["logical_patch_id"] for row in low}, {"low", "invalid"})
        self.assertEqual({row["logical_patch_id"] for row in kept}, {"legacy", "edge"})


if __name__ == "__main__":
    unittest.main()
