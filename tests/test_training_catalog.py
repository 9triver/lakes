from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

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

    def test_generated_labels_are_loaded_as_independent_annotation_sources(self) -> None:
        catalog = TrainingCatalogStub()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_id = "derived_0123456789abcdef0123"
            for index, source in enumerate(("spectral_water", "spectral_osm_consensus")):
                source_dir = root / source
                source_dir.mkdir()
                (source_dir / f"{label_id}.geojson").write_text(
                    json.dumps(
                        {
                            "type": "FeatureCollection",
                            "properties": {"site_id": "site_1", "source": source},
                            "features": [
                                {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [
                                            [
                                                [index, 0],
                                                [index + 1, 0],
                                                [index + 1, 1],
                                                [index, 0],
                                            ]
                                        ],
                                    },
                                    "properties": {"label_value": 1},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            layer = catalog.current_view_training_label_layer(
                SimpleNamespace(site_id="site_1"),
                {
                    "visible_layers": {
                        "spectral_water": True,
                        "spectral_osm_consensus": True,
                    },
                    "selected_generated_labels": {
                        "spectral_water": {"id": label_id},
                        "spectral_osm_consensus": {"id": label_id},
                    },
                },
                derived_label_dir=root,
            )

        self.assertEqual(len(layer["features"]), 2)
        self.assertEqual(
            {
                feature["properties"]["training_layer"]
                for feature in layer["features"]
            },
            {"spectral_water", "spectral_osm_consensus"},
        )

if __name__ == "__main__":
    unittest.main()
