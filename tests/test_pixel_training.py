from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_unet import train_model  # noqa: E402


class PixelTrainingTests(unittest.TestCase):
    def test_one_epoch_writes_generic_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            rows = []
            for index, site_id in enumerate(("site-a", "site-b")):
                npz_path = root / f"patch-{index}.npz"
                image = np.full((5, 8, 8), index + 1, dtype=np.int16)
                mask = np.zeros((8, 8), dtype=np.uint8)
                mask[:, :4] = 1
                np.savez_compressed(npz_path, image=image, mask=mask, valid=np.ones((8, 8), dtype=np.uint8))
                rows.append(
                    {
                        "patch_id": f"patch-{index}",
                        "sample_id": f"sample-{index}",
                        "site_id": site_id,
                        "npz_path": str(npz_path),
                        "include": "true",
                    }
                )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            output_dir = root / "run"
            args = argparse.Namespace(
                region="gansu",
                model_type="pixel_mlp",
                manifest=manifest,
                patch_dir=None,
                output_dir=output_dir,
                epochs=1,
                batch_size=1,
                lr=1e-3,
                weight_decay=1e-4,
                base_channels=32,
                val_ratio=0.5,
                seed=42,
                num_workers=0,
                device="cpu",
                threshold=0.5,
                pos_weight="auto",
                max_norm_patches=0,
                no_augment=False,
                dry_run=False,
            )

            result = train_model(args)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["config"]["model_type"], "pixel_mlp")
            self.assertEqual(result["config"]["split_group"], "site_id")
            self.assertFalse(result["config"]["augmentation_enabled"])
            self.assertTrue((output_dir / "best.pt").exists())
            self.assertTrue((output_dir / "last.pt").exists())
            self.assertTrue((output_dir / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
