from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lake_workbench.regions.config import RegionConfig
from lake_workbench.training.logical_patches import (
    build_logical_patches,
)
from lake_workbench.utils import read_csv_records, write_csv_records


class LogicalPatchPipelineTests(unittest.TestCase):
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
                                    "coordinates": [[[100, 30], [100.06, 30], [100.06, 29.93], [100, 29.93], [100, 30]]],
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

            empty_workspace_dir = root / "workspaces" / "empty" / "logical_patches" / "test"
            empty = build_logical_patches(region, output_dir=empty_workspace_dir, sample_ids=set())
            self.assertEqual(empty["patches"], 0)
            self.assertFalse((empty_workspace_dir / "manifest.csv").exists())
            self.assertTrue(all(workspace_dir in Path(row["preview_path"]).parents for row in read_csv_records(workspace_dir / "manifest.csv")))

            edge_row = next(row for row in logical_rows if int(row["row_off"]) == 512 and int(row["col_off"]) == 512)
            edge_row["include"] = "false"
            excluded_id = edge_row["logical_patch_id"]
            write_csv_records(manifest, logical_rows)
            build_logical_patches(region, output_dir=workspace_dir, sample_ids={"test_sample"})
            rebuilt = {row["logical_patch_id"]: row for row in read_csv_records(manifest)}
            self.assertEqual(rebuilt[excluded_id]["include"], "false")

            build_logical_patches(region, output_dir=workspace_dir, sample_ids={"test_sample"}, patch_size=256, stride=256)
            variants = read_csv_records(manifest)
            self.assertEqual({int(row["logical_size"]) for row in variants}, {256, 512})
            self.assertEqual(len({row["logical_patch_id"] for row in variants}), len(variants))


if __name__ == "__main__":
    unittest.main()
