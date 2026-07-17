from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lake_workbench.models.runtime import build_model, load_model_checkpoint, predict_array


class RegisteredModelTests(unittest.TestCase):
    def test_pixel_mlp_uses_smaller_default_architecture(self) -> None:
        model = build_model("pixel_mlp", 5)

        self.assertEqual(model[0].out_channels, 16)
        self.assertEqual(model[2].out_channels, 8)

    def test_pixel_mlp_is_pointwise_and_preserves_image_shape(self) -> None:
        model = build_model("pixel_mlp", 5).eval()
        first = torch.zeros(1, 5, 4, 4)
        changed_neighbor = first.clone()
        changed_neighbor[:, :, 0, 0] = 100

        with torch.no_grad():
            first_output = model(first)
            changed_output = model(changed_neighbor)

        self.assertEqual(tuple(first_output.shape), (1, 1, 4, 4))
        self.assertTrue(torch.equal(first_output[:, :, 2, 2], changed_output[:, :, 2, 2]))

    def test_pixel_mlp_checkpoint_round_trip_and_tiled_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixel.pt"
            network = build_model("pixel_mlp", 5, {"hidden_channels": [32, 16]})
            torch.save(
                {
                    "epoch": 3,
                    "model_state": network.state_dict(),
                    "config": {
                        "format_version": 2,
                        "model_type": "pixel_mlp",
                        "model_options": {"hidden_channels": [32, 16]},
                        "in_channels": 5,
                    },
                    "normalization": {"mean": [0] * 5, "std": [1] * 5},
                },
                path,
            )

            loaded = load_model_checkpoint(path, device="cpu")
            probability = predict_array(loaded, np.zeros((5, 513, 517), dtype=np.float32))

            self.assertEqual(loaded.model_type, "pixel_mlp")
            self.assertEqual(loaded.architecture_label, "5 -> 32 -> 16 -> 1")
            self.assertEqual(probability.shape, (513, 517))
            self.assertTrue(np.isfinite(probability).all())

    def test_legacy_checkpoint_without_model_type_defaults_to_unet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            network = build_model("unet", 5, {"base_channels": 4})
            torch.save(
                {
                    "epoch": 1,
                    "model_state": network.state_dict(),
                    "config": {"in_channels": 5, "base_channels": 4},
                    "normalization": {"mean": [0] * 5, "std": [1] * 5},
                },
                path,
            )

            loaded = load_model_checkpoint(path, device="cpu")

            self.assertEqual(loaded.model_type, "unet")
            self.assertEqual(loaded.base_channels, 4)


if __name__ == "__main__":
    unittest.main()
