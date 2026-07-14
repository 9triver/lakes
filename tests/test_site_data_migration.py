from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.migrate_site_data import canonicalize, migrate_csv


class SiteDataMigrationTests(unittest.TestCase):
    def test_json_identity_and_asset_metadata_are_canonicalized(self) -> None:
        payload = canonicalize(
            {
                "site_id": "gansu_17407",
                "lake_id": "gansu_17407",
                "lake": {"lake_id": "gansu_17407"},
                "asset": {
                    "asset_type": "lake_native",
                    "asset_scope": "lake",
                    "is_lake_native": True,
                    "applies_to_lake": True,
                },
            }
        )
        self.assertNotIn("lake_id", payload)
        self.assertNotIn("lake", payload)
        self.assertEqual(payload["site"], {"site_id": "gansu_17407"})
        self.assertEqual(payload["asset"]["asset_type"], "site_native")
        self.assertEqual(payload["asset"]["asset_scope"], "site")
        self.assertTrue(payload["asset"]["is_site_native"])
        self.assertTrue(payload["asset"]["applies_to_site"])

    def test_csv_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            pd.DataFrame(
                [
                    {
                        "lake_id": "gansu_17407",
                        "lake_name": "区域 17407",
                        "imagery_asset_type": "lake_native",
                        "imagery_asset_scope": "lake",
                    }
                ]
            ).to_csv(path, index=False)

            self.assertTrue(migrate_csv(path))
            self.assertFalse(migrate_csv(path))
            row = pd.read_csv(path).iloc[0].to_dict()
            self.assertEqual(row["site_id"], "gansu_17407")
            self.assertEqual(row["site_name"], "区域 17407")
            self.assertEqual(row["imagery_asset_type"], "site_native")
            self.assertEqual(row["imagery_asset_scope"], "site")


if __name__ == "__main__":
    unittest.main()
