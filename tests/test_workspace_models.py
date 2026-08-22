from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lake_workbench.models.workspaces import WorkspaceModelRegistry
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.regions.config import RegionConfig


class WorkspaceModelRegistryTests(unittest.TestCase):
    def test_resolves_only_workspace_scoped_model_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            region = RegionConfig(
                key="test",
                name="Test",
                data_dir=root / "raw",
                processed_dir=root / "processed",
                cache_dir=root / "cache",
                shared_data_dir=root / "shared",
            )
            store = WorkspaceStore({"test": region}, root=root / "workspaces", model_root=root / "models")
            store.ensure_default_workspace()
            workspace = store.create("Experiment")
            new_path = store.workspace_model_dir(workspace["id"], "test") / "run" / "best.pt"
            new_path.parent.mkdir(parents=True)
            new_path.touch()
            current = WorkspaceModelRegistry(store, {"test": region}, workspace["id"])

            self.assertEqual(current.resolve(f"workspaces/{workspace['id']}/test/run/best.pt", "test"), new_path)
            with self.assertRaises(ValueError):
                current.resolve("run/best.pt", "test")
            with self.assertRaises(ValueError):
                current.resolve("../../secret.pt", "test")

    def test_regular_user_cannot_list_or_resolve_foreign_workspace_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            region = RegionConfig(
                key="test",
                name="Test",
                data_dir=root / "raw",
                processed_dir=root / "processed",
                cache_dir=root / "cache",
                shared_data_dir=root / "shared",
            )
            store = WorkspaceStore({"test": region}, root=root / "workspaces", model_root=root / "models")
            store.ensure_default_workspace()
            foreign = store.create("Foreign")
            registry = WorkspaceModelRegistry(store, {"test": region}, "default")

            with self.assertRaisesRegex(ValueError, "其他训练工作区"):
                registry.list("test", "all")
            with self.assertRaisesRegex(ValueError, "其他训练工作区"):
                registry.resolve(
                    f"workspaces/{foreign['id']}/test/run/best.pt",
                    "test",
                )

    def test_admin_can_resolve_foreign_workspace_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            region = RegionConfig(
                key="test",
                name="Test",
                data_dir=root / "raw",
                processed_dir=root / "processed",
                cache_dir=root / "cache",
                shared_data_dir=root / "shared",
            )
            store = WorkspaceStore({"test": region}, root=root / "workspaces", model_root=root / "models")
            store.ensure_default_workspace()
            foreign = store.create("Foreign")
            registry = WorkspaceModelRegistry(
                store,
                {"test": region},
                "default",
                allow_foreign=True,
            )
            key = f"workspaces/{foreign['id']}/test/run/best.pt"

            self.assertEqual(
                registry.resolve(key, "test"),
                root / "models" / "workspaces" / foreign["id"] / "test" / "run" / "best.pt",
            )


if __name__ == "__main__":
    unittest.main()
