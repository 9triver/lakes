"""Torch device selection shared by training and model inference."""

from __future__ import annotations


SUPPORTED_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def mps_is_available(torch_module) -> bool:
    """Return whether this PyTorch build can use Apple's Metal backend."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(value: str, torch_module=None):
    """Resolve an explicit device or select CUDA, MPS, then CPU for ``auto``."""
    if torch_module is None:
        import torch as torch_module

    requested = str(value or "auto").strip().lower()
    if requested not in SUPPORTED_DEVICE_CHOICES:
        raise SystemExit(f"unsupported device: {requested}; choose one of {', '.join(SUPPORTED_DEVICE_CHOICES)}")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        return torch_module.device("cuda")
    if requested == "mps":
        if not mps_is_available(torch_module):
            raise SystemExit("MPS requested but not available")
        return torch_module.device("mps")
    if requested == "cpu":
        return torch_module.device("cpu")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if mps_is_available(torch_module):
        return torch_module.device("mps")
    return torch_module.device("cpu")
