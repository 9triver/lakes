"""U-Net checkpoint loading and raster inference helpers."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


torch = None
nn = None
F = None
UNet = None
_TORCH_INIT_LOCK = threading.Lock()


def init_torch() -> None:
    global torch, nn, F, UNet
    if torch is not None:
        return
    with _TORCH_INIT_LOCK:
        if torch is not None:
            return
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F

        torch_threads = max(1, int(os.environ.get("LAKES_TORCH_THREADS", "1")))
        _torch.set_num_threads(torch_threads)
        try:
            _torch.set_num_interop_threads(torch_threads)
        except RuntimeError:
            pass

        class DoubleConv(_nn.Module):
            def __init__(self, in_channels: int, out_channels: int) -> None:
                super().__init__()
                self.block = _nn.Sequential(
                    _nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    _nn.BatchNorm2d(out_channels),
                    _nn.ReLU(inplace=True),
                    _nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    _nn.BatchNorm2d(out_channels),
                    _nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.block(x)

        class _UNet(_nn.Module):
            def __init__(self, in_channels: int, base_channels: int = 32) -> None:
                super().__init__()
                c = base_channels
                self.down1 = DoubleConv(in_channels, c)
                self.down2 = DoubleConv(c, c * 2)
                self.down3 = DoubleConv(c * 2, c * 4)
                self.down4 = DoubleConv(c * 4, c * 8)
                self.pool = _nn.MaxPool2d(2)
                self.bottleneck = DoubleConv(c * 8, c * 16)
                self.up4 = _nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
                self.conv4 = DoubleConv(c * 16, c * 8)
                self.up3 = _nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
                self.conv3 = DoubleConv(c * 8, c * 4)
                self.up2 = _nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
                self.conv2 = DoubleConv(c * 4, c * 2)
                self.up1 = _nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
                self.conv1 = DoubleConv(c * 2, c)
                self.head = _nn.Conv2d(c, 1, kernel_size=1)

            def forward(self, x):
                d1 = self.down1(x)
                d2 = self.down2(self.pool(d1))
                d3 = self.down3(self.pool(d2))
                d4 = self.down4(self.pool(d3))
                x = self.bottleneck(self.pool(d4))
                x = self.conv4(_torch.cat([self.up4(x), d4], dim=1))
                x = self.conv3(_torch.cat([self.up3(x), d3], dim=1))
                x = self.conv2(_torch.cat([self.up2(x), d2], dim=1))
                x = self.conv1(_torch.cat([self.up1(x), d1], dim=1))
                return self.head(x)

        torch = _torch
        nn = _nn
        F = _F
        UNet = _UNet


@dataclass(frozen=True)
class LoadedUNet:
    path: Path
    device: object
    model: object
    in_channels: int
    base_channels: int
    mean: np.ndarray
    std: np.ndarray
    epoch: int
    config: dict


_MODEL_CACHE: dict[Path, LoadedUNet] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_unet_checkpoint(path: Path, device: str = "auto") -> LoadedUNet:
    init_torch()
    path = path.resolve()
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(path)
        if cached is not None:
            return cached
    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    checkpoint = torch.load(path, map_location=resolved_device)
    config = checkpoint.get("config") or {}
    normalization = checkpoint.get("normalization") or config.get("normalization") or {}
    in_channels = int(config.get("in_channels") or len(normalization.get("mean") or []))
    if in_channels <= 0:
        raise ValueError(f"Cannot determine model input channels: {path}")
    base_channels = int(config.get("base_channels") or 32)
    model = UNet(in_channels=in_channels, base_channels=base_channels).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape[0] != in_channels or std.shape[0] != in_channels:
        raise ValueError(f"Normalization channel count does not match model: {path}")
    std = np.maximum(std, 1e-6)
    loaded = LoadedUNet(
        path=path,
        device=resolved_device,
        model=model,
        in_channels=in_channels,
        base_channels=base_channels,
        mean=mean,
        std=std,
        epoch=int(checkpoint.get("epoch") or 0),
        config=config,
    )
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[path] = loaded
    return loaded


def predict_array(model: LoadedUNet, image: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    init_torch()
    if image.ndim != 3:
        raise ValueError("image must have shape (bands, height, width)")
    if image.shape[0] < model.in_channels:
        raise ValueError(f"Raster has {image.shape[0]} bands, model needs {model.in_channels}")
    data = image[: model.in_channels].astype(np.float32, copy=False)
    data = (data - model.mean[:, None, None]) / model.std[:, None, None]
    if valid is not None:
        data = np.where(valid[None, :, :], data, 0)
    height, width = data.shape[1:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(data[None])).to(model.device)
        logits = model.model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
    return probs[:height, :width]
