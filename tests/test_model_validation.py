from __future__ import annotations

import unittest
from types import SimpleNamespace

from lake_workbench.models.validation import ModelValidationMixin


class CatalogStub(ModelValidationMixin):
    def __init__(self) -> None:
        self.region = SimpleNamespace(key="test")
        self.sites = []


class LoadedModelTests(unittest.TestCase):
    def test_random_validation_requires_loaded_workspace_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "loaded workspace model"):
            CatalogStub().model_validation_random()

    def test_site_prediction_requires_loaded_workspace_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "loaded workspace model"):
            CatalogStub().model_prediction_for_site(SimpleNamespace(site_id="site-1"))


if __name__ == "__main__":
    unittest.main()
