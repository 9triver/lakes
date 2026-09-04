"""Torch-specific training loop utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from lake_workbench.training.unet.data import IGNORE_INDEX, Normalization


def run_epoch(model, loader, device, optimizer, pos_weight, threshold: float, cancel_event=None) -> dict:
    import torch
    import torch.nn.functional as functional

    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "valid": 0, "tp": 0, "fp": 0, "fn": 0}
    for batch in loader:
        if cancel_event is not None and cancel_event.is_set():
            raise KeyboardInterrupt("training cancelled")
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True) & (mask != IGNORE_INDEX)
        target = mask.clamp(0, 1).float()
        with torch.set_grad_enabled(training):
            logits = model(image).squeeze(1)
            raw_loss = functional.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = raw_loss[valid].mean() if valid.any() else raw_loss.mean() * 0
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred = torch.sigmoid(logits) >= threshold
            target_bool = target >= 0.5
            totals["loss"] += float(loss.detach().cpu()) * int(valid.sum().item())
            totals["valid"] += int(valid.sum().item())
            totals["tp"] += int((pred & target_bool & valid).sum().item())
            totals["fp"] += int((pred & ~target_bool & valid).sum().item())
            totals["fn"] += int((~pred & target_bool & valid).sum().item())
    return metrics_from_totals(totals)


def metrics_from_totals(totals: dict) -> dict:
    valid = max(1, totals["valid"])
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        "loss": totals["loss"] / valid,
        "iou": tp / max(1, tp + fp + fn),
        "dice": (2 * tp) / max(1, 2 * tp + fp + fn),
        "valid_pixels": totals["valid"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, normalization: Normalization, history: list) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "normalization": normalization.to_json(),
            "history": history,
        },
        path,
    )


def choose_device(value: str):
    import torch

    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def emit_progress(callback, status: str, **payload) -> None:
    if callback is None:
        message = payload.get("message")
        if message:
            print(message)
        return
    callback({"status": status, **payload})


def format_epoch(record: dict) -> str:
    train = record["train"]
    val = record["val"]
    val_text = f" val_loss={val['loss']:.4f} val_iou={val['iou']:.4f} val_dice={val['dice']:.4f}" if val else ""
    return (
        f"epoch={record['epoch']:03d}"
        f" lr={record['lr']:.2e}"
        f" train_loss={train['loss']:.4f} train_iou={train['iou']:.4f} train_dice={train['dice']:.4f}"
        f"{val_text}"
    )
