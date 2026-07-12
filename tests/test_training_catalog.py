from __future__ import annotations

import unittest

from lake_workbench.training_catalog import TrainingCatalogMixin


class TrainingCatalogStub(TrainingCatalogMixin):
    def _osm_layer(self, lake):
        return {
            "source": "OSM",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "properties": {},
        }

    def _match_hydrolakes(self, lake):
        return None

    def _esa_smoothed_layer(self, lake):
        return None

    def _jrc_occurrence_layer(self, lake, threshold=75):
        return None


class CurrentViewLabelTests(unittest.TestCase):
    def test_model_prediction_is_diagnostic_not_training_truth(self) -> None:
        catalog = TrainingCatalogStub()
        layer = catalog.current_view_training_label_layer(
            object(),
            {
                "visible_layers": {"osm": True, "model_prediction": True},
                "jrc_threshold": 80,
            },
        )
        self.assertEqual(len(layer["features"]), 1)
        self.assertEqual(layer["features"][0]["properties"]["training_layer"], "osm")
        self.assertTrue(layer["properties"]["model_prediction_visible"])
        self.assertTrue(layer["properties"]["model_prediction_excluded"])

    def test_current_view_requires_visible_label_geometry(self) -> None:
        catalog = TrainingCatalogStub()
        with self.assertRaises(ValueError):
            catalog.current_view_training_label_layer(object(), {"visible_layers": {}})


if __name__ == "__main__":
    unittest.main()
