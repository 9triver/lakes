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
    build_training_dataset,
    training_dataset_status,
)
from lake_workbench.utils import read_csv_records, write_csv_records


class LogicalPatchPipelineTests(unittest.TestCase):
    def test_logical_grid_state_and_derived_datasets(self) -> None:
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

            result = build_logical_patches(region)
            self.assertEqual(result["patches"], 4)
            logical_rows = read_csv_records(region.logical_patch_manifest)
            self.assertEqual({(int(row["row_off"]), int(row["col_off"])) for row in logical_rows}, {(0, 0), (0, 512), (512, 0), (512, 512)})
            self.assertTrue(all(Path(row["preview_path"]).exists() for row in logical_rows))

            profile_dir = root / "profiles" / "user-a" / "logical_patches" / "test"
            empty = build_logical_patches(region, output_dir=profile_dir, sample_ids=set())
            self.assertEqual(empty["patches"], 0)
            self.assertFalse((profile_dir / "manifest.csv").exists())
            isolated = build_logical_patches(region, output_dir=profile_dir, sample_ids={"test_sample"})
            self.assertEqual(isolated["patches"], 4)
            self.assertTrue(all(profile_dir in Path(row["preview_path"]).parents for row in read_csv_records(profile_dir / "manifest.csv")))

            edge_row = next(row for row in logical_rows if int(row["row_off"]) == 512 and int(row["col_off"]) == 512)
            edge_row["include"] = "false"
            excluded_id = edge_row["logical_patch_id"]
            write_csv_records(region.logical_patch_manifest, logical_rows)
            build_logical_patches(region)
            rebuilt = {row["logical_patch_id"]: row for row in read_csv_records(region.logical_patch_manifest)}
            self.assertEqual(rebuilt[excluded_id]["include"], "false")

            for config_id, expected_size in (("resize256_v1", 256), ("native512_v1", 512)):
                dataset = build_training_dataset(region, config_id)
                self.assertGreater(dataset["patches"], 0)
                rows = read_csv_records(region.training_dataset_dir / config_id / "manifest.csv")
                with np.load(Path(rows[0]["npz_path"])) as patch:
                    self.assertEqual(patch["image"].shape, (5, expected_size, expected_size))
                    self.assertEqual(patch["mask"].shape, (expected_size, expected_size))
                    self.assertTrue(set(np.unique(patch["mask"])).issubset({0, 1, 255}))
                self.assertTrue(training_dataset_status(region, config_id)["ready"])

            rebuilt[next(key for key in rebuilt if key != excluded_id)]["include"] = "false"
            write_csv_records(region.logical_patch_manifest, list(rebuilt.values()))
            self.assertEqual(training_dataset_status(region, "resize256_v1")["status"], "stale")


if __name__ == "__main__":
    unittest.main()
