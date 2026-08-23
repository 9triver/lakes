from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from lake_workbench.imagery.raster import render_tci_xyz_tile


class ImageryRasterTests(unittest.TestCase):
    def test_empty_xyz_pixels_are_transparent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tci.tif"
            data = np.zeros((3, 4, 4), dtype=np.uint8)
            data[:, :2, :2] = 120
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=3,
                dtype="uint8",
                crs="EPSG:3857",
                transform=from_bounds(0, 0, 4, 4, 4, 4),
            ) as dataset:
                dataset.write(data)

            payload = render_tci_xyz_tile(
                [{"tci_path": path, "valid_ratio": 1.0}],
                (0, 0, 4, 4),
                tile_size=4,
            )

            with Image.open(BytesIO(payload)) as image:
                self.assertEqual(image.mode, "RGBA")
                alpha = np.asarray(image.getchannel("A"))
            self.assertGreater(int(np.count_nonzero(alpha)), 0)
            self.assertLess(int(np.count_nonzero(alpha)), alpha.size)


if __name__ == "__main__":
    unittest.main()
