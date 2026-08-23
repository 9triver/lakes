from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lake_workbench.models.metadata import model_sort_key, persisted_training_job
from lake_workbench.training.datasets import split_rows_by_site
from lake_workbench.training.identity import bbox_iou, training_view_signature
from lake_workbench.training.runner import run_dataset_build


class TrainingIdentityTests(unittest.TestCase):
    def test_workspace_dataset_build_skips_region_without_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            region = SimpleNamespace(key="test")
            store = SimpleNamespace(
                members=lambda _workspace_id, _region: set(),
                get=lambda _workspace_id: {"status": "active"},
                workspace_dataset_dir=lambda _workspace_id, _region, config_id: Path(directory) / config_id,
            )
            with patch("lake_workbench.training.runner.REGIONS", {"test": region}):
                result = run_dataset_build("test", {"workspace_id": "workspace-1", "config_id": "resize256_v1"}, store)

        self.assertTrue(result["skipped"])
        self.assertEqual(result["status"], "missing_selection")

    def test_site_split_is_deterministic_and_has_no_site_overlap(self) -> None:
        rows = [
            {"patch_id": "a1", "sample_id": "sample-a1", "site_id": "site-a"},
            {"patch_id": "a2", "sample_id": "sample-a2", "site_id": "site-a"},
            {"patch_id": "b1", "sample_id": "sample-b1", "site_id": "site-b"},
            {"patch_id": "c1", "sample_id": "sample-c1", "site_id": "site-c"},
        ]

        first = split_rows_by_site(rows, val_ratio=0.34, seed=42)
        second = split_rows_by_site(rows, val_ratio=0.34, seed=42)

        self.assertEqual(first, second)
        train_sites = {row["site_id"] for row in first[0]}
        val_sites = {row["site_id"] for row in first[1]}
        self.assertFalse(train_sites & val_sites)
        self.assertEqual(train_sites | val_sites, {"site-a", "site-b", "site-c"})

    def test_site_split_falls_back_to_sample_id(self) -> None:
        rows = [
            {"patch_id": "a1", "sample_id": "sample-a"},
            {"patch_id": "a2", "sample_id": "sample-a"},
            {"patch_id": "b1", "sample_id": "sample-b"},
        ]

        train, val = split_rows_by_site(rows, val_ratio=0.5, seed=1)

        self.assertFalse({row["sample_id"] for row in train} & {row["sample_id"] for row in val})

    def test_prediction_diagnostics_do_not_change_training_identity(self) -> None:
        base = {
            "visible_layers": {"osm": True, "jrc": True, "model_prediction": True},
            "jrc_threshold": 75,
            "map": {"extent": [100.1234561, 20.0, 101.0, 21.0]},
            "model_validation": {"model_key": "first.pt", "predicted_ratio": 0.2},
        }
        changed = {
            **base,
            "visible_layers": {**base["visible_layers"], "model_prediction": False},
            "model_validation": {"model_key": "second.pt", "predicted_ratio": 0.9},
        }
        first = training_view_signature("site_1", "product", "current_view", "75", "current_view", "current_view", base)
        second = training_view_signature("site_1", "product", "current_view", "75", "current_view", "current_view", changed)
        self.assertEqual(first[:2], second[:2])

    def test_extent_is_rounded_for_stable_identity(self) -> None:
        first = {"visible_layers": {"osm": True}, "map": {"extent": [100.1234561, 20, 101, 21]}}
        second = {"visible_layers": {"osm": True}, "map": {"extent": [100.1234562, 20, 101, 21]}}
        a = training_view_signature("site_1", "product", "current_view", "", "current_view", "current_view", first)
        b = training_view_signature("site_1", "product", "current_view", "", "current_view", "current_view", second)
        self.assertEqual(a[0], b[0])

    def test_bbox_iou(self) -> None:
        self.assertEqual(bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        self.assertAlmostEqual(bbox_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)

    def test_model_sort_prefers_best_score_then_best_weight(self) -> None:
        items = [
            {"label": "last", "best_iou": 0.8, "weight": "last.pt", "epoch": 20},
            {"label": "best", "best_iou": 0.8, "weight": "best.pt", "epoch": 10},
            {"label": "lower", "best_iou": 0.7, "weight": "best.pt", "epoch": 30},
        ]
        ordered = sorted(items, key=model_sort_key)
        self.assertEqual([item["label"] for item in ordered], ["best", "last", "lower"])

    def test_persisted_job_without_training_artifacts_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "interrupted_run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps({"epochs": 30}), encoding="utf-8")

            job = persisted_training_job("all", run_dir)

        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["result"]["status"], "failed")
        self.assertEqual(job["message"], "训练未完成（服务重启或启动失败）")


if __name__ == "__main__":
    unittest.main()
