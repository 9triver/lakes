from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lake_workbench.models.profiles import ProfileModelRegistry
from lake_workbench.profiles import ProfileStore
from lake_workbench.regions.config import RegionConfig


class ProfileModelRegistryTests(unittest.TestCase):
    def test_resolves_new_and_default_legacy_model_keys(self) -> None:
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
            store = ProfileStore({"test": region}, root=root / "profiles", model_root=root / "models")
            store.ensure_default_profile()
            profile = store.create("Experiment")
            new_path = store.profile_model_dir(profile["id"], "test") / "run" / "best.pt"
            new_path.parent.mkdir(parents=True)
            new_path.touch()
            legacy_path = root / "models" / "test" / "old" / "best.pt"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.touch()

            current = ProfileModelRegistry(store, {"test": region}, profile["id"])
            default = ProfileModelRegistry(store, {"test": region}, "default")

            self.assertEqual(current.resolve(f"profiles/{profile['id']}/test/run/best.pt", "test"), new_path)
            self.assertEqual(default.resolve("old/best.pt", "test"), legacy_path)
            with self.assertRaises(ValueError):
                current.resolve("../../secret.pt", "test")


if __name__ == "__main__":
    unittest.main()
