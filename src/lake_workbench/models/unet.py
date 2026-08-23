"""Backward-compatible U-Net model exports."""

from lake_workbench.models.runtime import (
    LoadedWaterModel,
    build_model,
    init_torch,
    load_model_checkpoint,
    predict_array,
)


LoadedUNet = LoadedWaterModel


def load_unet_checkpoint(path, device: str = "auto") -> LoadedWaterModel:
    model = load_model_checkpoint(path, device=device)
    if model.model_type != "unet":
        raise ValueError(f"Checkpoint is {model.model_type}, not unet: {path}")
    return model


def create_unet(in_channels: int, base_channels: int = 32):
    return build_model("unet", in_channels, {"base_channels": base_channels})


__all__ = [
    "LoadedUNet",
    "create_unet",
    "init_torch",
    "load_unet_checkpoint",
    "predict_array",
]
