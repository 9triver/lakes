import json
import tempfile
import unittest
from pathlib import Path

from lake_workbench.imagery.inventory_sources import (
    load_active_imagery,
    save_active_imagery,
)


class ImageryInventorySourceTests(unittest.TestCase):
    def test_active_imagery_normalizes_legacy_tile_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text(
                json.dumps(
                    {
                        "T49SDT": "product-a",
                        "site-1:T49SWB": "product-b",
                        "site:site-2": "product-c",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_active_imagery(path),
                {
                    "49SDT": "product-a",
                    "site-1:49SWB": "product-b",
                    "site:site-2": "product-c",
                },
            )

    def test_active_imagery_writer_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed" / "active.json"

            save_active_imagery(path, {"49SDT": "product-a"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"49SDT": "product-a"}
            )


if __name__ == "__main__":
    unittest.main()
