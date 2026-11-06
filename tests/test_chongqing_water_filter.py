import importlib.util
import json
from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("water_filter", Path(__file__).parents[1] / "scripts" / "chongqing_patch_water_filter.py")
water_filter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(water_filter)

spec = importlib.util.spec_from_file_location("seed_api", Path(__file__).parents[1] / "scripts" / "chongqing_seed_api.py")
seed_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_api)


class ChongqingWaterFilterTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"minimum_water_ratio": "0.01", "previous_patch_ids": ["legacy"], "sample_ids": ["new"]}

    def row(self, patch_id, water, valid=10000, **values):
        return {"logical_patch_id": patch_id, "sample_id": "new", "water_pixels": str(water),
                "valid_pixels": str(valid), "include": "true", "review_status": "included", **values}

    def test_strict_boundary_uses_pixel_counts_and_valid_denominator(self):
        self.assertTrue(water_filter.below_threshold(self.row("low", 99, water_ratio_valid="0.010000"), "0.01"))
        self.assertFalse(water_filter.below_threshold(self.row("equal", 100), "0.01"))
        self.assertFalse(water_filter.below_threshold(self.row("high", 101), "0.01"))
        self.assertFalse(water_filter.below_threshold(self.row("partial", 50, valid=5000), "0.01"))

    def test_selector_preserves_existing_and_manual_exclusions(self):
        rows = [self.row("legacy", 0, sample_id="old", include="false", review_status="unreviewed", exclude_reason="manual"),
                self.row("low", 99), self.row("equal", 100),
                self.row("manual", 20, include="false", review_status="excluded", patch_notes="keep note"),
                self.row("unrelated", 0, sample_id="another-batch")]
        targets, preserved, stats = water_filter.select_patches(rows, self.policy)
        self.assertEqual(targets, ["low"])
        self.assertEqual(stats["below_threshold"], 2)
        self.assertEqual(stats["at_or_above_threshold"], 1)
        rows[1].update(include="false", review_status="excluded")
        report = water_filter.verify_patches(rows, self.policy, preserved)
        self.assertEqual(report["included_new_patches_after_filter"], 1)
        self.assertEqual(rows[3]["patch_notes"], "keep note")

    def test_verification_rejects_remaining_low_water_or_changed_legacy_review(self):
        rows = [self.row("legacy", 100), self.row("low", 99)]
        _, preserved, _ = water_filter.select_patches(rows, self.policy)
        with self.assertRaisesRegex(RuntimeError, "low-water"):
            water_filter.verify_patches(rows, self.policy, preserved)
        rows[1].update(include="false", review_status="excluded")
        rows[0]["include"] = "false"
        rows[0]["review_status"] = "excluded"
        with self.assertRaisesRegex(RuntimeError, "Existing patch review"):
            water_filter.verify_patches(rows, self.policy, preserved)

    def test_invalid_metrics_and_missing_baseline_stop_before_changes(self):
        with self.assertRaises(ValueError):
            water_filter.below_threshold(self.row("invalid", 1, valid=0), "0.01")
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            water_filter.select_patches([self.row("low", 1)], self.policy)

    def test_completed_receipt_can_resume_after_backend_job_memory_is_lost(self):
        receipt = {"status": "completed", "job_id": "export1", "result": {
            "region": "chongqing", "workspace_id": "default", "samples": 1,
            "generated_patches": 2, "patches": 3}}
        policy = {**self.policy, "job_id": "export1", "completed_export_receipt": receipt}
        self.assertEqual(water_filter.completed_export_receipt(policy), receipt)

    def test_completed_receipt_rejects_different_batch_or_incomplete_export(self):
        receipt = {"status": "completed", "job_id": "export1", "result": {
            "region": "chongqing", "workspace_id": "default", "samples": 1,
            "generated_patches": 2, "patches": 3}}
        for changes in ({"status": "running"}, {"job_id": "other"}, {"result": {**receipt["result"], "samples": 2}}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                water_filter.completed_export_receipt({**self.policy, "job_id": "export1", "completed_export_receipt": {**receipt, **changes}})


class WaterFilterBuildGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.policy = {"minimum_water_ratio": "0.01", "previous_patch_ids": ["legacy"]}
        (self.directory / "water_filter_policy.json").write_text(json.dumps(self.policy), encoding="utf-8")
        self.result = {"status": "completed", "job_id": "export1", "minimum_water_ratio": "0.01",
                       "verified": True, "remaining_below_threshold_included": 0, "preserved_existing_patches": 1}

    def write_result(self, value):
        (self.directory / "water_filter_result.json").write_text(json.dumps(value), encoding="utf-8")

    def test_waits_for_server_verification_before_returning(self):
        with patch.object(seed_api, "REPORT_DIR", self.directory), patch.object(seed_api, "stage_progress"), \
             patch.object(seed_api, "log"), patch.object(seed_api.time, "sleep", side_effect=lambda _: self.write_result(self.result)) as waiting:
            result = seed_api.wait_for_water_filter({"job_id": "export1"})
        waiting.assert_called_once_with(30)
        self.assertEqual(result, self.result)

    def test_unverified_or_wrong_export_results_block_build(self):
        for changes in ({"job_id": "other"}, {"minimum_water_ratio": "0.02"}, {"verified": False},
                        {"remaining_below_threshold_included": 1}, {"preserved_existing_patches": 0}):
            with self.subTest(changes=changes):
                self.write_result({**self.result, **changes})
                with patch.object(seed_api, "REPORT_DIR", self.directory), self.assertRaises(AssertionError):
                    seed_api.wait_for_water_filter({"job_id": "export1"})


class WaterFilterPollingTests(unittest.TestCase):
    def test_timeout_while_export_commits_retries_read_only_request(self):
        response = io.BytesIO(b'{"status":"completed"}')
        with patch.object(water_filter.urllib.request, "urlopen", side_effect=[TimeoutError(), response]) as opener, \
             patch.object(water_filter.time, "sleep") as waiting, patch("builtins.print"):
            result = water_filter.api({"api_base": "http://localhost"}, "/export-job")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(opener.call_count, 2)
        waiting.assert_called_once_with(30)

    def test_ambiguous_exclusion_timeout_is_not_replayed(self):
        with patch.object(water_filter.urllib.request, "urlopen", side_effect=TimeoutError()) as opener, \
             patch.object(water_filter.time, "sleep") as waiting, self.assertRaises(TimeoutError):
            water_filter.api({"api_base": "http://localhost"}, "/logical-patches", {"operation": "exclude"})
        self.assertEqual(opener.call_count, 1)
        waiting.assert_not_called()


if __name__ == "__main__":
    unittest.main()
