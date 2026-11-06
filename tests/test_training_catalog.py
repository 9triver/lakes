from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lake_workbench.training.catalog import TrainingCatalogMixin
from lake_workbench.training.samples import _snapshot_raster_label_sources


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
            for index, source in enumerate(
                (
                    "spectral_water",
                    "spectral_osm_intersection",
                    "spectral_osm_consensus",
                    "osm_spectral_consensus",
                )
            ):
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
                        "spectral_osm_intersection": True,
                        "spectral_osm_consensus": True,
                        "osm_spectral_consensus": True,
                    },
                    "selected_generated_labels": {
                        "spectral_water": {"id": label_id},
                        "spectral_osm_intersection": {"id": label_id},
                        "spectral_osm_consensus": {"id": label_id},
                        "osm_spectral_consensus": {"id": label_id},
                    },
                },
                derived_label_dir=root,
            )

        self.assertEqual(len(layer["features"]), 4)
        self.assertEqual(
            {
                feature["properties"]["training_layer"]
                for feature in layer["features"]
            },
            {
                "spectral_water",
                "spectral_osm_intersection",
                "spectral_osm_consensus",
                "osm_spectral_consensus",
            },
        )

    def test_generated_raster_label_is_snapshotted_with_training_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "derived.npz"
            np.savez_compressed(
                source_path,
                labels=np.array([[0, 1], [255, 0]], dtype=np.uint8),
                valid=np.ones((2, 2), dtype=np.uint8),
                transform=np.array([1, 0, 0, 0, -1, 2], dtype=np.float64),
                crs=np.asarray("EPSG:4326"),
            )
            layer = {
                "type": "FeatureCollection",
                "features": [],
                "properties": {
                    "raster_label_sources": [
                        {
                            "source": "spectral_water",
                            "label_id": "derived_0123456789abcdef0123",
                            "path": str(source_path),
                        }
                    ]
                },
            }

            snapshot = _snapshot_raster_label_sources(
                layer, "site/sample", root / "training_labels"
            )
            copied_name = snapshot["properties"]["raster_label_sources"][0]["path"]
            copied_path = root / "training_labels" / copied_name

            self.assertTrue(copied_path.is_file())
            self.assertNotEqual(copied_path, source_path)
            source_path.unlink()
            with np.load(copied_path, allow_pickle=False) as copied:
                self.assertEqual(copied["labels"].tolist(), [[0, 1], [255, 0]])

if __name__ == "__main__":
    unittest.main()
