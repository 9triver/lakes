from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lake_workbench.model_validation import ModelValidationMixin


class CatalogStub(ModelValidationMixin):
    def __init__(self, root: Path) -> None:
        self.region = SimpleNamespace(
            key="test",
            model_dir=root / "models" / "test",
            legacy_model_dir=root / "legacy_models",
            processed_dir=root / "processed",
        )
        self.lakes = []


class ModelPathTests(unittest.TestCase):
    def test_region_model_key_resolves_under_region_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CatalogStub(Path(directory))
            self.assertEqual(
                catalog._model_path_from_key("run/best.pt"),
                catalog.region.model_dir / "run" / "best.pt",
            )

    def test_legacy_model_key_resolves_under_legacy_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CatalogStub(Path(directory))
            self.assertEqual(
                catalog._model_path_from_key("legacy/run/best.pt"),
                catalog.region.legacy_model_dir / "run" / "best.pt",
            )

    def test_absolute_and_traversal_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = CatalogStub(Path(directory))
            with self.assertRaises(ValueError):
                catalog._model_path_from_key("/tmp/model.pt")
            with self.assertRaises(ValueError):
                catalog._model_path_from_key("../best.pt")


if __name__ == "__main__":
    unittest.main()
