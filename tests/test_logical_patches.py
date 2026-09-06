from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from shapely.geometry import shape

from lake_workbench.regions.config import RegionConfig
from lake_workbench.training.logical_patches import (
    build_logical_patches,
    workspace_logical_patch_preview,
)
from lake_workbench.training.samples import _clip_label_layer
from lake_workbench.utils import read_csv_records, write_csv_records


class LogicalPatchPipelineTests(unittest.TestCase):
    @staticmethod
    def _write_raster(path: Path, driver: str = "GTiff") -> None:
        image = np.full((5, 1024, 1024), 1000, dtype=np.uint16)
        with rasterio.open(
            path,
            "w",
            driver=driver,
            width=1024,
            height=1024,
            count=5,
            dtype=image.dtype,
            crs="EPSG:4326",
            transform=from_origin(100, 30, 0.001, 0.001),
        ) as target:
            target.write(image)

    @staticmethod
    def _write_label(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[100, 30], [101, 30], [101, 29], [100, 29], [100, 30]]
                                ],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_current_view_limits_patch_windows_and_marks_outside_pixels_invalid(self) -> None:
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
            image_path = region.data_dir / "image.tif"
            image_path.parent.mkdir(parents=True)
            self._write_raster(image_path)
            label_path = region.training_label_dir / "sample.geojson"
            self._write_label(label_path)
            samples = [
                {
                    "sample_id": "view-sample",
                    "site_id": "site-1",
                    "tci_path": str(image_path),
                    "label_path": str(label_path),
                    "view_west": 100,
                    "view_south": 29.8,
                    "view_east": 100.2,
                    "view_north": 30,
                }
            ]

            output_dir = root / "patches"
            build_logical_patches(region, output_dir, samples=samples)
            rows = read_csv_records(output_dir / "manifest.csv")

            self.assertEqual(len(rows), 1)
            self.assertEqual((int(rows[0]["row_off"]), int(rows[0]["col_off"])), (0, 0))
            self.assertEqual(int(rows[0]["valid_pixels"]), 200 * 200)
            self.assertEqual(int(rows[0]["water_pixels"]), 200 * 200)
            self.assertEqual(int(rows[0]["ignore_pixels"]), 512 * 512 - 200 * 200)

    def test_label_snapshot_is_clipped_to_captured_view(self) -> None:
        layer = {
            "type": "FeatureCollection",
            "properties": {"source": "current_view"},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"training_layer": "osm"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]],
                    },
                }
            ],
        }

        clipped = _clip_label_layer(layer, [1, 1, 2, 2])

        self.assertIsNotNone(clipped)
        assert clipped is not None
        geometry = clipped["features"][0]["geometry"]
        self.assertEqual(
            tuple(round(value, 6) for value in shape(geometry).bounds),
            (1, 1, 2, 2),
        )
        self.assertEqual(clipped["properties"], {"source": "current_view"})

    def test_window_generation_supports_tiff_and_img_sources(self) -> None:
        with rasterio.Env() as environment:
            if "HFA" not in environment.drivers():
                self.skipTest("Rasterio HFA driver is unavailable")
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
            region.data_dir.mkdir(parents=True)
            tiff_path = region.data_dir / "first.tif"
            img_path = region.data_dir / "second.img"
            self._write_raster(tiff_path)
            self._write_raster(img_path, driver="HFA")
            label_path = region.training_label_dir / "sample.geojson"
            self._write_label(label_path)
            samples = [
                {
                    "sample_id": "dual-format",
                    "site_id": "site-1",
                    "tci_path": f"{tiff_path};{img_path}",
                    "label_path": str(label_path),
                }
            ]

            output_dir = root / "patches"
            build_logical_patches(region, output_dir, samples=samples)
            rows = read_csv_records(output_dir / "manifest.csv")

            self.assertEqual(len(rows), 8)
            self.assertEqual({Path(row["image_path"]).suffix for row in rows}, {".tif", ".img"})

    def test_preview_falls_back_to_saved_png_when_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview_path = root / "preview.png"
            Image.new("RGB", (8, 8), (12, 34, 56)).save(preview_path)
            base_preview_path = root / "preview.base.png"
            Image.new("RGB", (8, 8), (65, 76, 87)).save(base_preview_path)
            region = RegionConfig(
                key="test",
                name="Test",
                data_dir=root / "raw",
                processed_dir=root / "processed",
                cache_dir=root / "cache",
                shared_data_dir=root / "shared",
            )
            row = {
                "logical_patch_id": "missing-source",
                "image_path": "data/regions/test/missing.tif",
                "label_path": "data/regions/test/missing.geojson",
                "preview_path": str(preview_path),
                "preview_base_path": str(base_preview_path),
            }

            self.assertEqual(workspace_logical_patch_preview(region, row, None, "workspace"), preview_path.read_bytes())
            self.assertEqual(workspace_logical_patch_preview(region, row, None, "workspace", overlay=False), base_preview_path.read_bytes())

            legacy_row = {key: value for key, value in row.items() if key != "preview_base_path"}
            self.assertEqual(workspace_logical_patch_preview(region, legacy_row, None, "workspace", overlay=False), preview_path.read_bytes())

    def test_workspace_logical_grid_preserves_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            raw = root / "raw"
            raw.mkdir()
            region = RegionConfig(
                key="test",
                name="Test",
                data_dir=raw,
                processed_dir=processed,
                cache_dir=root / "cache",
                shared_data_dir=root / "shared",
            )
            image_path = raw / "image.tif"
            image = np.full((5, 700, 600), 1000, dtype=np.uint16)
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=600,
                height=700,
                count=5,
                dtype=image.dtype,
                crs="EPSG:4326",
                transform=from_origin(100, 30, 0.0001, 0.0001),
            ) as dst:
                dst.write(image)
            label_path = processed / "training_labels" / "sample.geojson"
            label_path.parent.mkdir(parents=True)
            label_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[[100, 30], [100.03, 30], [100.03, 29.965], [100, 29.965], [100, 30]]],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            write_csv_records(
                region.training_samples,
                [
                    {
                        "sample_id": "test_sample",
                        "site_id": "test_site",
                        "site_name": "Test site",
                        "tci_path": str(image_path),
                        "label_path": str(label_path),
                        "product_name": "test_product",
                        "product_date": "2026-07-16",
                    }
                ],
            )

            workspace_dir = root / "workspaces" / "user-a" / "logical_patches" / "test"
            result = build_logical_patches(region, output_dir=workspace_dir, sample_ids={"test_sample"})
            self.assertEqual(result["patches"], 4)
            manifest = workspace_dir / "manifest.csv"
            logical_rows = read_csv_records(manifest)
            self.assertEqual({(int(row["row_off"]), int(row["col_off"])) for row in logical_rows}, {(0, 0), (0, 512), (512, 0), (512, 512)})
            self.assertTrue(all(Path(row["preview_path"]).exists() for row in logical_rows))
            self.assertTrue(all(Path(row["preview_base_path"]).exists() for row in logical_rows))
            zero_water = [row for row in logical_rows if int(row["water_pixels"]) == 0]
            self.assertTrue(zero_water)
            self.assertTrue(all(row["include"] == "false" for row in zero_water))
            self.assertTrue(all(row["review_status"] == "excluded" for row in zero_water))
            self.assertTrue(all(row["exclude_reason"] == "no_water" for row in zero_water))

            empty_workspace_dir = root / "workspaces" / "empty" / "logical_patches" / "test"
            empty = build_logical_patches(region, output_dir=empty_workspace_dir, sample_ids=set())
            self.assertEqual(empty["patches"], 0)
            self.assertFalse((empty_workspace_dir / "manifest.csv").exists())
            self.assertTrue(all(workspace_dir in Path(row["preview_path"]).parents for row in read_csv_records(workspace_dir / "manifest.csv")))

            edge_row = next(row for row in logical_rows if int(row["row_off"]) == 512 and int(row["col_off"]) == 512)
            edge_row["include"] = "false"
            excluded_id = edge_row["logical_patch_id"]
            manual_row = next(row for row in zero_water if row["logical_patch_id"] != excluded_id)
            manual_row["include"] = "true"
            manual_row["review_status"] = "included"
            manual_row["exclude_reason"] = ""
            write_csv_records(manifest, logical_rows)
            build_logical_patches(region, output_dir=workspace_dir, sample_ids={"test_sample"})
            rebuilt = {row["logical_patch_id"]: row for row in read_csv_records(manifest)}
            self.assertEqual(rebuilt[excluded_id]["include"], "false")
            self.assertEqual(rebuilt[manual_row["logical_patch_id"]]["include"], "true")

            build_logical_patches(region, output_dir=workspace_dir, sample_ids={"test_sample"}, patch_size=256, stride=256)
            variants = read_csv_records(manifest)
            self.assertEqual({int(row["logical_size"]) for row in variants}, {256, 512})
            self.assertEqual(len({row["logical_patch_id"] for row in variants}), len(variants))


if __name__ == "__main__":
    unittest.main()
