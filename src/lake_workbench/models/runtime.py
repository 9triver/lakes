"""Registered water-segmentation models, checkpoint loading, and inference."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_MODEL_TYPES = ("unet", "pixel_mlp")
PIXEL_MLP_HIDDEN_CHANNELS = (32, 16)
PIXEL_MLP_INFERENCE_TILE = 512

torch = None
nn = None
_TORCH_INIT_LOCK = threading.Lock()


def init_torch() -> None:
    global torch, nn
    if torch is not None:
        return
    with _TORCH_INIT_LOCK:
        if torch is not None:
            return
        import torch as _torch
        import torch.nn as _nn

        torch_threads = max(1, int(os.environ.get("LAKES_TORCH_THREADS", "1")))
        _torch.set_num_threads(torch_threads)
        try:
            _torch.set_num_interop_threads(torch_threads)
        except RuntimeError:
            pass
        torch = _torch
        nn = _nn


def normalize_model_type(value: Any) -> str:
    model_type = str(value or "unet").strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unknown model type: {model_type}")
    return model_type


def model_options(model_type: str, config: dict | None = None) -> dict:
    config = config or {}
    stored = config.get("model_options") if isinstance(config.get("model_options"), dict) else {}
    if model_type == "pixel_mlp":
        hidden = stored.get("hidden_channels") or PIXEL_MLP_HIDDEN_CHANNELS
        hidden = [int(value) for value in hidden]
        if hidden != list(PIXEL_MLP_HIDDEN_CHANNELS):
            raise ValueError(f"pixel_mlp hidden_channels must be {list(PIXEL_MLP_HIDDEN_CHANNELS)}")
        return {"hidden_channels": hidden}
    return {"base_channels": int(stored.get("base_channels") or config.get("base_channels") or 32)}


def architecture_label(model_type: str, in_channels: int, options: dict) -> str:
    if model_type == "pixel_mlp":
        hidden = options["hidden_channels"]
        return " -> ".join(str(value) for value in [in_channels, *hidden, 1])
    return f"U-Net (base {options['base_channels']})"


def _build_pixel_mlp(in_channels: int, options: dict):
    hidden_a, hidden_b = options["hidden_channels"]
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_a, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_a, hidden_b, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_b, 1, kernel_size=1),
    )


def _build_unet(in_channels: int, options: dict):
    class DoubleConv(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.block(value)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = options["base_channels"]
            self.down1 = DoubleConv(in_channels, channels)
            self.down2 = DoubleConv(channels, channels * 2)
            self.down3 = DoubleConv(channels * 2, channels * 4)
            self.down4 = DoubleConv(channels * 4, channels * 8)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(channels * 8, channels * 16)
            self.up4 = nn.ConvTranspose2d(channels * 16, channels * 8, kernel_size=2, stride=2)
            self.conv4 = DoubleConv(channels * 16, channels * 8)
            self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
            self.conv3 = DoubleConv(channels * 8, channels * 4)
            self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
            self.conv2 = DoubleConv(channels * 4, channels * 2)
            self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(channels * 2, channels)
            self.head = nn.Conv2d(channels, 1, kernel_size=1)

        def forward(self, value):
            down1 = self.down1(value)
            down2 = self.down2(self.pool(down1))
            down3 = self.down3(self.pool(down2))
            down4 = self.down4(self.pool(down3))
            value = self.bottleneck(self.pool(down4))
            value = self.conv4(torch.cat([self.up4(value), down4], dim=1))
            value = self.conv3(torch.cat([self.up3(value), down3], dim=1))
            value = self.conv2(torch.cat([self.up2(value), down2], dim=1))
            value = self.conv1(torch.cat([self.up1(value), down1], dim=1))
            return self.head(value)

    return UNet()


MODEL_BUILDERS = {
    "unet": _build_unet,
    "pixel_mlp": _build_pixel_mlp,
}


def build_model(model_type: str, in_channels: int, options: dict | None = None):
    init_torch()
    model_type = normalize_model_type(model_type)
    resolved_options = model_options(model_type, {"model_options": options or {}})
    return MODEL_BUILDERS[model_type](in_channels, resolved_options)


@dataclass(frozen=True)
class LoadedWaterModel:
    path: Path
    device: object
    model: object
    model_type: str
    model_options: dict
    architecture_label: str
    in_channels: int
    base_channels: int | None
    mean: np.ndarray
    std: np.ndarray
    epoch: int
    config: dict


_MODEL_CACHE: dict[tuple[Path, str], LoadedWaterModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_model_checkpoint(path: Path, device: str = "auto") -> LoadedWaterModel:
    init_torch()
    path = path.resolve()
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    cache_key = (path, str(resolved_device))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    checkpoint = torch.load(path, map_location=resolved_device)
    config = checkpoint.get("config") or {}
    normalization = checkpoint.get("normalization") or config.get("normalization") or {}
    in_channels = int(config.get("in_channels") or len(normalization.get("mean") or []))
    if in_channels <= 0:
        raise ValueError(f"Cannot determine model input channels: {path}")
    model_type = normalize_model_type(config.get("model_type") or "unet")
    options = model_options(model_type, config)
    network = build_model(model_type, in_channels, options).to(resolved_device)
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape[0] != in_channels or std.shape[0] != in_channels:
        raise ValueError(f"Normalization channel count does not match model: {path}")
    loaded = LoadedWaterModel(
        path=path,
        device=resolved_device,
        model=network,
        model_type=model_type,
        model_options=options,
        architecture_label=architecture_label(model_type, in_channels, options),
        in_channels=in_channels,
        base_channels=options.get("base_channels"),
        mean=mean,
        std=np.maximum(std, 1e-6),
        epoch=int(checkpoint.get("epoch") or 0),
        config=config,
    )
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = loaded
    return loaded


def predict_array(model: LoadedWaterModel, image: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    init_torch()
    if image.ndim != 3:
        raise ValueError("image must have shape (bands, height, width)")
    if image.shape[0] < model.in_channels:
        raise ValueError(f"Raster has {image.shape[0]} bands, model needs {model.in_channels}")
    data = image[: model.in_channels].astype(np.float32, copy=False)
    data = (data - model.mean[:, None, None]) / model.std[:, None, None]
    if valid is not None:
        data = np.where(valid[None, :, :], data, 0)
    if model.model_type == "pixel_mlp":
        return _predict_pixel_mlp(model, data)
    return _predict_unet(model, data)


def _predict_pixel_mlp(model: LoadedWaterModel, data: np.ndarray) -> np.ndarray:
    height, width = data.shape[1:]
    result = np.empty((height, width), dtype=np.float32)
    with torch.no_grad():
        for row in range(0, height, PIXEL_MLP_INFERENCE_TILE):
            for col in range(0, width, PIXEL_MLP_INFERENCE_TILE):
                tile = np.ascontiguousarray(data[:, row : row + PIXEL_MLP_INFERENCE_TILE, col : col + PIXEL_MLP_INFERENCE_TILE])
                logits = model.model(torch.from_numpy(tile[None]).to(model.device))
                result[row : row + tile.shape[1], col : col + tile.shape[2]] = (
                    torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
                )
    return result


def _predict_unet(model: LoadedWaterModel, data: np.ndarray) -> np.ndarray:
    height, width = data.shape[1:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(data[None])).to(model.device)
        probabilities = torch.sigmoid(model.model(tensor)).squeeze(0).squeeze(0).detach().cpu().numpy()
    return probabilities[:height, :width]
