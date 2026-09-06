from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from lake_workbench.imagery.inventory import ImageryInventoryMixin
from lake_workbench.imagery.formats import local_imagery_paths, local_imagery_paths_from_roots
from lake_workbench.imagery.display import display_band_indexes
from lake_workbench.imagery.tiles import render_tci_xyz_tile
from lake_workbench.imagery.validity import valid_pixel_mask


class ImageryRasterTests(unittest.TestCase):
    def test_valid_pixel_requires_every_input_band(self) -> None:
        image = np.array(
            [
                [[1, 1], [1, 0]],
                [[1, -9999], [1, 0]],
                [[1, 1], [np.nan, 0]],
            ],
            dtype="float32",
        )
        mask = valid_pixel_mask(image, nodata=-9999)
        self.assertEqual(mask.tolist(), [[True, False], [False, False]])

    def test_valid_pixel_supports_per_band_nodata_and_masks(self) -> None:
        image = np.array(
            [
                [[1, -1], [1, 1]],
                [[1, 2], [-2, 2]],
                [[1, 3], [3, 3]],
            ],
            dtype="float32",
        )
        masks = np.full(image.shape, 255, dtype="uint8")
        masks[2, 1, 1] = 0

        valid = valid_pixel_mask(image, nodata=(-1, -2, None), masks=masks)

        self.assertEqual(valid.tolist(), [[True, False], [False, False]])

    def test_display_bands_use_band_descriptions_for_local_rasters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=5,
                dtype="uint16",
                crs="EPSG:3857",
                transform=from_bounds(0, 0, 2, 2, 2, 2),
            ) as dataset:
                dataset.write(np.ones((5, 2, 2), dtype="uint16"))
                for index, description in enumerate(("rhot_492", "rhot_560", "rhot_665", "rhot_833", "rhot_1614"), 1):
                    dataset.set_band_description(index, description)
            with rasterio.open(path) as dataset:
                self.assertEqual(display_band_indexes(dataset, {"source": "local_imagery"}), [3, 2, 1])

    def test_local_imagery_paths_prefers_geotiff_for_same_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            img = root / "S2A_MSIL1C_20260101.img"
            tif = root / "S2A_MSIL1C_20260101.tif"
            other = root / "S2B_MSIL1C_20260101.img"
            img.touch()
            tif.touch()
            other.touch()

            self.assertEqual(local_imagery_paths(root), [tif, other])

    def test_local_imagery_paths_from_roots_prefers_first_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            compressed = root / "compressed"
            original.mkdir()
            compressed.mkdir()
            (original / "S2A_MSIL1C_20260101.img").touch()
            (compressed / "S2A_MSIL1C_20260101.tif").touch()

            self.assertEqual(
                local_imagery_paths_from_roots([compressed, original]),
                [compressed / "S2A_MSIL1C_20260101.tif"],
            )

    def test_local_label_path_supports_both_suffix_styles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "S2A_MSIL1C_20260101.img"
            dotted_label = image_path.with_name("S2A_MSIL1C_20260101.Swater.shp")
            underscored_label = image_path.with_name("S2A_MSIL1C_20260101_Swater.shp")

            self.assertEqual(ImageryInventoryMixin._local_label_path(image_path), dotted_label)
            dotted_label.touch()
            self.assertEqual(ImageryInventoryMixin._local_label_path(image_path), dotted_label)
            dotted_label.unlink()
            underscored_label.touch()
            self.assertEqual(ImageryInventoryMixin._local_label_path(image_path), underscored_label)

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

    def test_per_band_nodata_is_hidden_from_png_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nodata.tif"
            data = np.full((3, 4, 4), 100, dtype="uint16")
            data[:, :2, :2] = 9999
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=3,
                dtype="uint16",
                nodata=9999,
                crs="EPSG:4326",
                transform=from_bounds(0, 0, 4, 4, 4, 4),
            ) as dataset:
                dataset.write(data)

            from lake_workbench.imagery.mosaic import render_tci_png

            payload, _meta = render_tci_png(path, (0, 0, 4, 4), size=64, padding=0)
            with Image.open(BytesIO(payload)) as image:
                pixels = np.asarray(image)
            self.assertEqual(pixels[0, 0].tolist(), [0, 0, 0])
            self.assertGreater(int(pixels.max()), 0)


if __name__ == "__main__":
    unittest.main()
