from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from lake_workbench.regions.config import RegionConfig
from lake_workbench.training.logical_patches import (
    build_logical_patches,
    workspace_logical_patch_preview,
)
from lake_workbench.utils import read_csv_records, write_csv_records


class LogicalPatchPipelineTests(unittest.TestCase):
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
