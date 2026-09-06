from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lake_workbench.workspaces import WorkspaceError, WorkspaceStore
from lake_workbench.regions.config import RegionConfig
from lake_workbench.training.datasets import current_workspace_training_dataset_summary
from lake_workbench.training.logical_patches import build_workspace_training_dataset, workspace_training_dataset_status
from lake_workbench.utils import read_csv_records, write_csv_records


class WorkspaceStoreTests(unittest.TestCase):
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
        self.region_manifest = self.region.processed_dir / "logical_patches" / "manifest.csv"
        write_csv_records(
            self.region_manifest,
            [
                {"logical_patch_id": "patch-osm", "sample_id": "sample-osm", "site_id": "site-1", "include": "true"},
                {"logical_patch_id": "patch-jrc", "sample_id": "sample-jrc", "site_id": "site-1", "include": "false"},
            ],
        )
        self.store = WorkspaceStore(
            {"test": self.region},
            root=self.root / "workspaces",
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

    def _install_workspace_rows(self, workspace_id: str, *patch_ids: str) -> Path:
        rows = {
            row["logical_patch_id"]: row
            for row in read_csv_records(self.region_manifest)
        }
        manifest = self.store.workspace_logical_patch_manifest(workspace_id, "test")
        write_csv_records(manifest, [rows[patch_id] for patch_id in patch_ids])
        return manifest

    def test_default_workspace_does_not_import_region_logical_patches(self) -> None:
        workspace = self.store.ensure_default_workspace()

        self.assertEqual(workspace["id"], "default")
        self.assertEqual(workspace["status"], "active")
        self.assertEqual(self.store.members("default"), set())
        self.assertEqual(self.store.ensure_default_workspace()["selected_patch_count"], 0)
        self.assertFalse(self.store.workspace_logical_patch_manifest("default", "test").exists())

    def test_manifest_can_include_different_label_snapshots_for_one_site(self) -> None:
        self.store.ensure_default_workspace()
        self._install_workspace_rows("default", "patch-osm", "patch-jrc")
        self.store.update_members("default", "test", ["patch-osm"], "include")

        self.store.update_members("default", "test", ["patch-jrc"], "include")

        self.assertEqual(self.store.members("default"), {("test", "patch-osm"), ("test", "patch-jrc")})

    def test_empty_workspace_does_not_inherit_default_logical_patch_catalog(self) -> None:
        self.store.ensure_default_workspace()
        self._install_workspace_rows("default", "patch-osm", "patch-jrc")
        self.store.update_members("default", "test", ["patch-osm"], "include")
        empty = self.store.create("Empty")

        default_manifest = self.store.workspace_logical_patch_manifest("default", "test")
        empty_manifest = self.store.ensure_workspace_logical_patch_manifest(empty["id"], "test")

        self.assertEqual({row["logical_patch_id"] for row in read_csv_records(default_manifest)}, {"patch-osm", "patch-jrc"})
        self.assertEqual(read_csv_records(empty_manifest), [])
        self.assertNotEqual(default_manifest, empty_manifest)

    def test_empty_manifest_is_authoritative_for_membership(self) -> None:
        self.store.ensure_default_workspace()
        write_csv_records(self.store.workspace_logical_patch_manifest("default", "test"), [])

        self.assertEqual(self.store.members("default", "test"), set())

    def test_including_one_patch_preserves_other_exclusion_reasons(self) -> None:
        self.store.ensure_default_workspace()
        manifest = self._install_workspace_rows("default", "patch-osm", "patch-jrc")
        rows = read_csv_records(manifest)
        next(row for row in rows if row["logical_patch_id"] == "patch-jrc")[
            "exclude_reason"
        ] = "no_water"
        write_csv_records(manifest, rows)

        self.store.update_members("default", "test", ["patch-osm"], "include")

        updated = {
            row["logical_patch_id"]: row for row in read_csv_records(manifest)
        }
        self.assertEqual(updated["patch-jrc"]["exclude_reason"], "no_water")

    def test_archive_and_restore_preserve_workspace(self) -> None:
        self.store.ensure_default_workspace()
        workspace = self.store.create("Disposable")
        archived = self.store.archive(workspace["id"])

        self.assertEqual(archived["status"], "archived")
        self.assertNotIn(workspace["id"], {item["id"] for item in self.store.list()["items"]})
        self.assertEqual(self.store.restore(workspace["id"])["status"], "active")
        with self.assertRaises(WorkspaceError):
            self.store.archive("default")

    def test_workspace_dataset_rebuilds_masks_from_selected_source_snapshot(self) -> None:
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
        rows = read_csv_records(self.region_manifest)
        for row in rows:
            sample = next(item for item in read_csv_records(self.region.training_samples) if item["sample_id"] == row["sample_id"])
            row.update({"image_path": str(image_path), "label_path": sample["label_path"], "row_off": 0, "col_off": 0})
        self.store.ensure_default_workspace()
        write_csv_records(self.store.workspace_logical_patch_manifest("default", "test"), rows)
        self.store.update_members("default", "test", ["patch-osm"], "include")

        result = build_workspace_training_dataset(self.region, "resize256_v1", self.store, "default")

        self.assertEqual(result["patches"], 1)
        manifest = read_csv_records(self.store.workspace_dataset_dir("default", "test", "resize256_v1") / "manifest.csv")
        self.assertEqual(manifest[0]["workspace_id"], "default")
        with np.load(Path(manifest[0]["npz_path"])) as data:
            self.assertGreater(int(np.count_nonzero(data["mask"] == 1)), 0)
        self.assertEqual(workspace_training_dataset_status(self.region, "resize256_v1", self.store, "default")["status"], "ready")

    def test_empty_workspace_exposes_global_dataset_config_without_manifest_error(self) -> None:
        self.store.ensure_default_workspace()
        empty = self.store.create("Empty")

        summary = current_workspace_training_dataset_summary(self.store, empty["id"], "all", "resize256_v1")

        self.assertEqual(summary["status"], "missing_selection")
        self.assertIn("自动从已纳入的 Workspace Patch 准备数据", summary["error"])


if __name__ == "__main__":
    unittest.main()
