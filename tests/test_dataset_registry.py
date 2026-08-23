from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lake_workbench.training.registry import DatasetConflict, DatasetRegistry
from lake_workbench.utils import write_csv_records


class DatasetRegistryTests(unittest.TestCase):
    def test_contribution_detects_duplicate_and_spatial_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "workspace" / "logical_patches" / "gansu" / "manifest.csv"
            write_csv_records(
                source,
                [
                    {
                        "logical_patch_id": "patch-1",
                        "region": "gansu",
                        "include": "true",
                        "image_fingerprint": "image-a",
                        "bounds_left": 0,
                        "bounds_bottom": 0,
                        "bounds_right": 1,
                        "bounds_top": 1,
                    },
                    {
                        "logical_patch_id": "patch-2",
                        "region": "gansu",
                        "include": "true",
                        "image_fingerprint": "image-a",
                        "bounds_left": 0,
                        "bounds_bottom": 0,
                        "bounds_right": 1,
                        "bounds_top": 1,
                    },
                ],
            )
            store = SimpleNamespace(
                regions={"gansu": object()},
                ensure_workspace_logical_patch_manifest=lambda _workspace, _region: source,
            )
            registry = DatasetRegistry(root / "global")

            first = registry.contribute(store, "workspace", "gansu", "gansu", ["patch-1"])
            self.assertEqual(first["added"], 1)
            self.assertEqual(registry.contribution_scopes("workspace", ["patch-1"]), {"patch-1": ["gansu"]})
            with self.assertRaises(DatasetConflict) as duplicate:
                registry.contribute(store, "workspace", "gansu", "gansu", ["patch-1"])
            self.assertEqual(duplicate.exception.conflicts[0]["type"], "duplicate")
            with self.assertRaises(DatasetConflict) as overlap:
                registry.contribute(store, "other-workspace", "gansu", "gansu", ["patch-2"])
            self.assertEqual(overlap.exception.conflicts[0]["type"], "spatial_overlap")

            withdrawn = registry.withdraw("workspace", "gansu", ["patch-1"])
            self.assertEqual(withdrawn["removed"], 1)
            self.assertEqual(registry.list("gansu")["total"], 0)

    def test_cross_region_contribution_is_not_committed_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {}
            for region, patch_id in (("gansu", "patch-gansu"), ("shaanxi", "patch-shaanxi")):
                manifests[region] = root / "workspace" / region / "manifest.csv"
                write_csv_records(manifests[region], [{
                    "logical_patch_id": patch_id,
                    "region": region,
                    "include": "true",
                    "image_fingerprint": f"image-{region}",
                    "bounds_left": 0,
                    "bounds_bottom": 0,
                    "bounds_right": 1,
                    "bounds_top": 1,
                }])
            store = SimpleNamespace(
                regions={"gansu": object(), "shaanxi": object()},
                ensure_workspace_logical_patch_manifest=lambda _workspace, region: manifests[region],
            )
            registry = DatasetRegistry(root / "global")

            with self.assertRaises(KeyError):
                registry.contribute_many(store, "workspace", "all", {
                    "gansu": ["patch-gansu"],
                    "shaanxi": ["missing"],
                })

            self.assertEqual(registry.list("all")["total"], 0)


if __name__ == "__main__":
    unittest.main()
