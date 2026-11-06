from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lake_workbench.regions.config import RegionConfig
from lake_workbench.training import logical_patches, runner
from lake_workbench.training.logical_patches import workspace_training_dataset_status
from lake_workbench.users import UserStore
from lake_workbench.utils import read_csv_records, write_csv_records
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.workspaces import membership
from lake_workbench.workspaces import store as workspace_store_module


class TrainingResponsivenessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.region = RegionConfig(
            key="test", name="Test", data_dir=root / "raw",
            processed_dir=root / "processed", cache_dir=root / "cache",
            shared_data_dir=root / "shared",
        )
        self.store = WorkspaceStore({"test": self.region}, root=root / "workspaces")
        self.store.ensure_default_workspace()
        self.users = UserStore(self.store, root=root / "users")
        self.users.ensure_default_user()
        image = root / "image.tif"
        with rasterio.open(
            image, "w", driver="GTiff", width=64, height=64, count=5,
            dtype="uint16", crs="EPSG:4326", transform=from_origin(0, 1, 1 / 64, 1 / 64),
        ) as target:
            target.write(np.full((5, 64, 64), 1000, dtype="uint16"))
        label = root / "label.geojson"
        label.write_text(json.dumps({
            "type": "FeatureCollection", "features": [{
                "type": "Feature", "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            }],
        }), encoding="utf-8")
        write_csv_records(self.store.workspace_training_samples("default", "test"), [{
            "sample_id": "sample", "site_id": "site", "image_path": str(image),
            "tci_path": str(image), "label_path": str(label),
        }])
        self.manifest = self.store.workspace_logical_patch_manifest("default", "test")
        self.options = {"workspace_id": "default", "patch_size": 32, "stride": 32}
        self.regions_patch = patch.object(runner, "REGIONS", {"test": self.region})
        self.regions_patch.start()
        self.addCleanup(self.regions_patch.stop)
        runner.run_patch_export("test", self.options, self.store)

    def _start(self, action):
        errors = []
        def run():
            try:
                action()
            except BaseException as error:
                errors.append(error)
        worker = Thread(target=run, daemon=True)
        worker.start()
        return worker, errors

    def test_authentication_and_members_reuse_unchanged_patch_manifest(self):
        expected = self.users.get("default")["workspace"]
        with patch.object(workspace_store_module, "read_csv_records", side_effect=AssertionError("unchanged catalog re-read")):
            self.assertEqual(self.users.get("default")["workspace"], expected)
            self.assertEqual(len(self.store.members("default", "test")), expected["selected_patch_count"])

    def test_summary_updates_after_review_and_external_export(self):
        before = self.store.get("default")
        rows = read_csv_records(self.manifest)
        self.store.update_members("default", "test", [rows[0]["logical_patch_id"]], "exclude")
        reviewed = self.store.get("default")
        self.assertEqual(reviewed["selected_patch_count"], before["selected_patch_count"] - 1)
        rows = read_csv_records(self.manifest)
        rows.append({**rows[1], "logical_patch_id": "external-export", "site_id": "other-site"})
        write_csv_records(self.manifest, rows)
        exported = self.users.get("default")["workspace"]
        self.assertEqual(exported["selected_patch_count"], before["selected_patch_count"])
        self.assertEqual(exported["site_count"], reviewed["site_count"] + 1)

    def test_export_allows_authentication_and_preserves_reviews_during_build(self):
        rows = read_csv_records(self.manifest)
        rebuilt_id = rows[0]["logical_patch_id"]
        legacy = {**rows[0], "logical_patch_id": "legacy", "sample_id": "legacy-sample"}
        write_csv_records(self.manifest, [*rows, legacy])
        entered, release, reviewed = Event(), Event(), Event()
        original = logical_patches._build_sample_logical_patches
        def slow_export(*args, **kwargs):
            result = original(*args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test export was not released")
            return result
        with patch.object(logical_patches, "_build_sample_logical_patches", side_effect=slow_export):
            worker, errors = self._start(lambda: runner.run_patch_export("test", self.options, self.store))
            try:
                self.assertTrue(entered.wait(2))
                def review():
                    self.assertEqual(self.users.get("default")["id"], "default")
                    self.store.update_members("default", "test", [rebuilt_id, "legacy"], "exclude")
                    reviewed.set()
                request, request_errors = self._start(review)
                self.assertTrue(reviewed.wait(1), "authentication/review waited for raster export")
            finally:
                release.set()
                worker.join(5)
            request.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors + request_errors, [])
        current = {row["logical_patch_id"]: row for row in read_csv_records(self.manifest)}
        self.assertEqual(current[rebuilt_id]["include"], "false")
        self.assertEqual(current["legacy"]["include"], "false")

    def test_dataset_build_allows_review_and_marks_changed_selection_stale(self):
        patch_id = read_csv_records(self.manifest)[0]["logical_patch_id"]
        entered, release, reviewed = Event(), Event(), Event()
        original = logical_patches.read_patch_window
        first = True
        def slow_read(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test dataset was not released")
            return original(*args, **kwargs)
        with patch.object(logical_patches, "read_patch_window", side_effect=slow_read):
            worker, errors = self._start(lambda: runner.run_dataset_build("test", {"workspace_id": "default"}, self.store))
            try:
                self.assertTrue(entered.wait(2))
                def review():
                    self.assertEqual(self.users.get("default")["id"], "default")
                    self.store.update_members("default", "test", [patch_id], "exclude")
                    reviewed.set()
                request, request_errors = self._start(review)
                self.assertTrue(reviewed.wait(1), "authentication/review waited for dataset materialization")
            finally:
                release.set()
                worker.join(5)
            request.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors + request_errors, [])
        status = workspace_training_dataset_status(self.region, "resize256_v1", self.store, "default")
        self.assertEqual(status["status"], "stale")

    def test_failed_export_keeps_published_manifest_and_previews(self):
        before = self.manifest.read_bytes()
        paths = [Path(row["preview_path"]) for row in read_csv_records(self.manifest)]
        with patch.object(logical_patches, "write_preview", side_effect=RuntimeError("simulated raster failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated raster failure"):
                runner.run_patch_export("test", self.options, self.store)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_conflicts_compare_only_matching_images(self):
        bounds = {"bounds_left": 0, "bounds_bottom": 0, "bounds_right": 1, "bounds_top": 1}
        canonical = {
            f"old-{i}": {**bounds, "image_fingerprint": f"image-{i}"}
            for i in range(100)
        }
        canonical.update({
            f"new-{i}": {**bounds, "image_fingerprint": "image-0" if i == 0 else f"new-image-{i}"}
            for i in range(100)
        })
        with patch.object(membership, "patch_bbox_iou", wraps=membership.patch_bbox_iou) as compare:
            conflicts = membership.find_patch_conflicts(
                canonical, iter(f"old-{i}" for i in range(100)), [f"new-{i}" for i in range(100)]
            )
        self.assertEqual([(row["patch_id"], row["existing_patch_id"]) for row in conflicts], [("new-0", "old-0")])
        self.assertEqual(compare.call_count, 1)
