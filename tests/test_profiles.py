from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lake_workbench.profiles import ProfileError, ProfileStore
from lake_workbench.regions.config import RegionConfig
from lake_workbench.training.datasets import current_profile_training_dataset_summary
from lake_workbench.training.logical_patches import build_profile_training_dataset, profile_training_dataset_status
from lake_workbench.utils import read_csv_records, write_csv_records


class ProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.region = RegionConfig(
            key="test",
            name="Test",
            data_dir=self.root / "raw",
            processed_dir=self.root / "processed",
            cache_dir=self.root / "cache",
            shared_data_dir=self.root / "shared",
        )
        self.region.training_label_dir.mkdir(parents=True)
        self._sample("sample-osm", "site-1", {"osm": {}}, "label-osm.geojson")
        self._sample("sample-jrc", "site-1", {"jrc": {"threshold": 75}}, "label-jrc.geojson")
        write_csv_records(
            self.region.logical_patch_manifest,
            [
                {"logical_patch_id": "patch-osm", "sample_id": "sample-osm", "site_id": "site-1", "include": "true"},
                {"logical_patch_id": "patch-jrc", "sample_id": "sample-jrc", "site_id": "site-1", "include": "false"},
            ],
        )
        self.store = ProfileStore(
            {"test": self.region},
            root=self.root / "profiles",
            model_root=self.root / "models",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _sample(self, sample_id: str, site_id: str, sources: dict[str, dict], filename: str) -> None:
        features = []
        for index, source in enumerate(sources):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[index, 0], [index + 1, 0], [index + 1, 1], [index, 1], [index, 0]]],
                    },
                    "properties": {"training_layer": source},
                }
            )
        path = self.region.training_label_dir / filename
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "properties": {"jrc_threshold": 75},
                    "features": features,
                }
            ),
            encoding="utf-8",
        )
        rows = []
        if self.region.training_samples.exists():
            import pandas as pd

            rows = pd.read_csv(self.region.training_samples, dtype=str).fillna("").to_dict("records")
        rows.append(
            {
                "sample_id": sample_id,
                "site_id": site_id,
                "label_path": str(path),
                "label_source": "current_view",
                "context_sources": ",".join(sources),
            }
        )
        write_csv_records(self.region.training_samples, rows)

    def test_default_migration_imports_current_membership(self) -> None:
        profile = self.store.ensure_default_profile()

        self.assertEqual(profile["id"], "default")
        self.assertEqual(profile["status"], "active")
        self.assertEqual(self.store.members("default"), {("test", "patch-osm")})
        self.assertEqual(self.store.ensure_default_profile()["selected_patch_count"], 1)

    def test_member_source_conflict_requires_resolution(self) -> None:
        self.store.ensure_default_profile()

        self.store.update_members("default", "test", ["patch-jrc"], "include")

        self.assertEqual(self.store.get("default")["status"], "needs_resolution")
        conflict = self.store.conflicts("default")["items"][0]
        self.assertEqual(conflict["site_id"], "site-1")
        self.assertEqual({item["source"] for item in conflict["variants"]}, {"osm", "jrc"})
        selected = [item["id"] for item in conflict["variants"] if item["source"] == "jrc"]
        self.store.resolve_sources("default", "site-1", selected)
        self.assertEqual(self.store.get("default")["status"], "active")
        self.assertEqual([item["source"] for item in self.store.selected_variants("default", "site-1")], ["jrc"])

    def test_union_is_snapshot_and_does_not_copy_history(self) -> None:
        self.store.ensure_default_profile()
        first = self.store.create("First")
        self.store.update_members(first["id"], "test", ["patch-osm"], "include")
        union = self.store.create("Union", "union", ["default", first["id"]])

        self.store.update_members(first["id"], "test", ["patch-jrc"], "include")

        self.assertEqual(self.store.members(union["id"]), {("test", "patch-osm")})
        self.assertEqual(union["training_defaults"], {})
        union_rows = read_csv_records(self.store.profile_logical_patch_manifest(union["id"], "test"))
        self.assertEqual({row["logical_patch_id"] for row in union_rows}, {"patch-osm"})

    def test_empty_profile_does_not_inherit_default_logical_patch_catalog(self) -> None:
        self.store.ensure_default_profile()
        empty = self.store.create("Empty")

        default_manifest = self.store.profile_logical_patch_manifest("default", "test")
        empty_manifest = self.store.ensure_profile_logical_patch_manifest(empty["id"], "test")

        self.assertEqual({row["logical_patch_id"] for row in read_csv_records(default_manifest)}, {"patch-osm", "patch-jrc"})
        self.assertEqual(read_csv_records(empty_manifest), [])
        self.assertNotEqual(default_manifest, empty_manifest)

    def test_archive_and_restore_preserve_profile(self) -> None:
        self.store.ensure_default_profile()
        profile = self.store.create("Disposable")
        archived = self.store.archive(profile["id"])

        self.assertEqual(archived["status"], "archived")
        self.assertNotIn(profile["id"], {item["id"] for item in self.store.list()["items"]})
        self.assertEqual(self.store.restore(profile["id"])["status"], "active")
        with self.assertRaises(ProfileError):
            self.store.archive("default")

    def test_profile_dataset_rebuilds_masks_from_selected_source_snapshot(self) -> None:
        image_path = self.region.data_dir / "image.tif"
        image_path.parent.mkdir(parents=True)
        image = np.full((5, 512, 512), 1000, dtype=np.uint16)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            width=512,
            height=512,
            count=5,
            dtype=image.dtype,
            crs="EPSG:4326",
            transform=from_origin(0, 1, 1 / 512, 1 / 512),
        ) as target:
            target.write(image)
        rows = read_csv_records(self.region.logical_patch_manifest)
        for row in rows:
            row.update({"image_path": str(image_path), "row_off": 0, "col_off": 0})
        write_csv_records(self.region.logical_patch_manifest, rows)
        self.store.ensure_default_profile()

        result = build_profile_training_dataset(self.region, "resize256_v1", self.store, "default")

        self.assertEqual(result["patches"], 1)
        manifest = read_csv_records(self.store.profile_dataset_dir("default", "test", "resize256_v1") / "manifest.csv")
        self.assertEqual(manifest[0]["profile_id"], "default")
        with np.load(Path(manifest[0]["npz_path"])) as data:
            self.assertGreater(int(np.count_nonzero(data["mask"] == 1)), 0)
        self.assertEqual(profile_training_dataset_status(self.region, "resize256_v1", self.store, "default")["status"], "ready")

    def test_empty_profile_exposes_global_dataset_config_without_manifest_error(self) -> None:
        self.store.ensure_default_profile()
        empty = self.store.create("Empty")

        summary = current_profile_training_dataset_summary(self.store, empty["id"], "all", "resize256_v1")

        self.assertEqual(summary["status"], "missing_selection")
        self.assertIn("已选逻辑 Patch", summary["error"])


if __name__ == "__main__":
    unittest.main()
