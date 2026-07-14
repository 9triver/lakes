from __future__ import annotations

import unittest

from lake_workbench.training.catalog import TrainingCatalogMixin


class TrainingCatalogStub(TrainingCatalogMixin):
    def annotation_for_site(self, site, source, options=None):
        annotation = None
        if source == "osm":
            annotation = {
                "source": "OSM",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                "properties": {},
            }
        elif source == "local":
            annotation = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"label_id": (options or {}).get("label_id")},
                    }
                ],
            }
        return {
            "site_id": "site_1",
            "source": source,
            "status": "available" if annotation else "missing",
            "parameters": options or {},
            "annotation": annotation,
        }


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

    def test_local_label_is_loaded_through_annotation_provider(self) -> None:
        catalog = TrainingCatalogStub()
        layer = catalog.current_view_training_label_layer(
            object(),
            {
                "visible_layers": {"local_label": True},
                "selected_local_label": {"id": "label_1"},
            },
        )

        self.assertEqual(len(layer["features"]), 1)
        self.assertEqual(layer["features"][0]["properties"]["label_id"], "label_1")
        self.assertEqual(layer["features"][0]["properties"]["training_layer"], "local_label")


if __name__ == "__main__":
    unittest.main()
