from __future__ import annotations

import unittest

from lake_workbench.models.metadata import model_sort_key
from lake_workbench.training.identity import bbox_iou, training_view_signature


class TrainingIdentityTests(unittest.TestCase):
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
        first = training_view_signature("lake_1", "product", "current_view", "75", "current_view", "current_view", base)
        second = training_view_signature("lake_1", "product", "current_view", "75", "current_view", "current_view", changed)
        self.assertEqual(first[:2], second[:2])

    def test_extent_is_rounded_for_stable_identity(self) -> None:
        first = {"visible_layers": {"osm": True}, "map": {"extent": [100.1234561, 20, 101, 21]}}
        second = {"visible_layers": {"osm": True}, "map": {"extent": [100.1234562, 20, 101, 21]}}
        a = training_view_signature("lake_1", "product", "current_view", "", "current_view", "current_view", first)
        b = training_view_signature("lake_1", "product", "current_view", "", "current_view", "current_view", second)
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


if __name__ == "__main__":
    unittest.main()
