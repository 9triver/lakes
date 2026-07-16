from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lake_workbench.training.catalog import TrainingCatalogMixin
from lake_workbench.utils import write_csv_records


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


class TrainingPatchCountTests(unittest.TestCase):
    def test_counts_only_included_patches_with_existing_npz_files(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            manifest = root_path / "processed" / "training_patches" / "ps256_st128" / "manifest.csv"
            first_npz = root_path / "first.npz"
            second_npz = root_path / "second.npz"
            first_npz.touch()
            second_npz.touch()
            write_csv_records(
                manifest,
                [
                    {"patch_id": "included", "site_id": "site-a", "npz_path": str(first_npz), "include": "true"},
                    {"patch_id": "default-included", "site_id": "site-a", "npz_path": str(second_npz), "include": ""},
                    {"patch_id": "excluded", "site_id": "site-a", "npz_path": str(first_npz), "include": "false"},
                    {"patch_id": "missing", "site_id": "site-b", "npz_path": str(root_path / "missing.npz"), "include": "true"},
                ],
            )
            catalog = TrainingCatalogStub()
            catalog.region = SimpleNamespace(processed_dir=root_path / "processed")

            self.assertEqual(catalog.usable_training_patch_counts(), {"site-a": 2})

    def test_uses_only_the_latest_patch_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            patch_root = root_path / "processed" / "training_patches"
            old_manifest = patch_root / "old" / "manifest.csv"
            new_manifest = patch_root / "new" / "manifest.csv"
            npz_path = root_path / "patch.npz"
            npz_path.touch()
            write_csv_records(old_manifest, [{"patch_id": "old", "site_id": "old-site", "npz_path": str(npz_path), "include": "true"}])
            write_csv_records(new_manifest, [{"patch_id": "new", "site_id": "new-site", "npz_path": str(npz_path), "include": "true"}])
            os.utime(old_manifest, ns=(1_000_000_000, 1_000_000_000))
            os.utime(new_manifest, ns=(2_000_000_000, 2_000_000_000))
            catalog = TrainingCatalogStub()
            catalog.region = SimpleNamespace(processed_dir=root_path / "processed")

            self.assertEqual(catalog.usable_training_patch_counts(), {"new-site": 1})


if __name__ == "__main__":
    unittest.main()
