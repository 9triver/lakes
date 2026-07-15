#!/usr/bin/env python3
"""Train a registered water-segmentation model from exported patch npz files."""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lake_workbench.regions.config import DEFAULT_CONFIG_PATH, load_region_configs  # noqa: E402
from lake_workbench.models.runtime import (  # noqa: E402
    SUPPORTED_MODEL_TYPES,
    architecture_label,
    build_model,
    model_options,
    normalize_model_type,
)
from lake_workbench.training.datasets import split_rows_by_site  # noqa: E402


REGIONS, DEFAULT_REGION_KEY = load_region_configs(DEFAULT_CONFIG_PATH)
MODEL_ROOT = PROJECT_ROOT / "data" / "models"
IGNORE_INDEX = 255
torch = None
F = None
DataLoader = None
Dataset = object
PatchDataset = None


def init_torch() -> None:
    global torch, F, DataLoader, Dataset, PatchDataset
    if torch is not None:
        return
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in this environment.\n"
            "Install it first, for example:\n"
            "  .venv/bin/python -m pip install torch\n"
            "Then rerun this script."
        ) from exc

    class _PatchDataset(Dataset):
        def __init__(self, rows: list[dict], normalization: Normalization, augment: bool = False) -> None:
            self.rows = rows
            self.normalization = normalization
            self.augment = augment

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            with np.load(resolve_project_path(row["npz_path"])) as data:
                image = data["image"].astype("float32")
                mask = data["mask"].astype("uint8")
                valid = data["valid"].astype("uint8")
            image = (image - self.normalization.mean[:, None, None]) / self.normalization.std[:, None, None]
            if self.augment:
                image, mask, valid = augment_patch(image, mask, valid)
            return {
                "image": torch.from_numpy(np.ascontiguousarray(image)),
                "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
                "valid": torch.from_numpy(np.ascontiguousarray(valid)).bool(),
                "patch_id": row.get("patch_id", ""),
            }

    PatchDataset = _PatchDataset


@dataclass
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS) + ["all"], default=DEFAULT_REGION_KEY)
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default="unet")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", default="auto", help="'auto' or a numeric BCE positive weight.")
    parser.add_argument("--max-norm-patches", type=int, default=0, help="0 means use every training patch.")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load data, compute splits, then exit without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(args)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def train_model(args: argparse.Namespace, progress_callback=None, cancel_event: threading.Event | None = None) -> dict:
    init_torch()
    seed_everything(args.seed)
    selected_model_type = normalize_model_type(getattr(args, "model_type", "unet"))
    region = REGIONS[args.region] if args.region != "all" else None
    manifest = args.manifest or latest_manifest(region, args.patch_dir)
    output_dir = args.output_dir or MODEL_ROOT / args.region / f"{selected_model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest)
    if not rows:
        raise SystemExit(f"no included training patches found: {manifest}")
    in_channels = patch_channels(rows[0])
    train_rows, val_rows = split_rows_by_site(rows, args.val_ratio, args.seed)
    normalization = compute_normalization(train_rows, max_patches=args.max_norm_patches)
    pos_weight = compute_pos_weight(train_rows) if args.pos_weight == "auto" else float(args.pos_weight)
    selected_model_options = model_options(
        selected_model_type,
        {"base_channels": args.base_channels},
    )

    config = {
        "format_version": 2,
        "model_type": selected_model_type,
        "model_options": selected_model_options,
        "architecture_label": architecture_label(selected_model_type, in_channels, selected_model_options),
        "scope": args.region,
        "region": args.region if args.region != "all" else "",
        "regions": sorted(REGIONS) if args.region == "all" else [args.region],
        "manifest": display_path(manifest),
        "output_dir": display_path(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "split_group": "site_id",
        "seed": args.seed,
        "threshold": args.threshold,
        "no_augment": selected_model_type == "pixel_mlp" or bool(args.no_augment),
        "augmentation_enabled": selected_model_type == "unet" and not args.no_augment,
        "device_requested": args.device,
        "max_norm_patches": args.max_norm_patches,
        "in_channels": in_channels,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_site_count": len({row.get("site_id") for row in train_rows if row.get("site_id")}),
        "val_site_count": len({row.get("site_id") for row in val_rows if row.get("site_id")}),
        "normalization": normalization.to_json(),
        "pos_weight": pos_weight,
    }
    if selected_model_type == "unet":
        config["base_channels"] = args.base_channels
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_progress(progress_callback, "configured", message="训练配置已生成", progress=5, config=config)
    if args.dry_run:
        return {
            "status": "dry_run",
            "config": config,
            "output_dir": display_path(output_dir),
            "manifest": display_path(manifest),
        }

    device = choose_device(args.device)
    config["device"] = str(device)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    model = build_model(selected_model_type, in_channels, selected_model_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    train_loader = DataLoader(
        PatchDataset(train_rows, normalization, augment=selected_model_type == "unet" and not args.no_augment),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PatchDataset(val_rows, normalization, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_iou = -1.0
    history = []
    emit_progress(progress_callback, "running", message="开始训练", progress=10, config=config)
    for epoch in range(1, args.epochs + 1):
        if cancel_event is not None and cancel_event.is_set():
            emit_progress(progress_callback, "cancelled", message="训练已取消", progress=100, history=history)
            return {
                "status": "cancelled",
                "output_dir": display_path(output_dir),
                "manifest": display_path(manifest),
                "best_model": display_path(output_dir / "best.pt") if (output_dir / "best.pt").exists() else "",
                "last_model": display_path(output_dir / "last.pt") if (output_dir / "last.pt").exists() else "",
                "history": history,
                "config": config,
            }
        try:
            train_metrics = run_epoch(model, train_loader, device, optimizer, pos_weight_tensor, args.threshold, cancel_event=cancel_event)
            val_metrics = (
                run_epoch(model, val_loader, device, None, pos_weight_tensor, args.threshold, cancel_event=cancel_event)
                if val_rows
                else {}
            )
        except KeyboardInterrupt:
            emit_progress(progress_callback, "cancelled", message="训练已取消", progress=100, history=history)
            return {
                "status": "cancelled",
                "output_dir": display_path(output_dir),
                "manifest": display_path(manifest),
                "best_model": display_path(output_dir / "best.pt") if (output_dir / "best.pt").exists() else "",
                "last_model": display_path(output_dir / "last.pt") if (output_dir / "last.pt").exists() else "",
                "history": history,
                "config": config,
            }
        score = val_metrics.get("iou", train_metrics["iou"])
        scheduler.step(score)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, normalization, history)
        saved_best = False
        if score > best_iou:
            best_iou = score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, normalization, history)
            saved_best = True
        emit_progress(
            progress_callback,
            "running",
            message=format_epoch(record),
            progress=10 + int(epoch * 85 / max(1, args.epochs)),
            epoch=epoch,
            epochs=args.epochs,
            record=record,
            best_iou=best_iou,
            saved_best=saved_best,
            output_dir=display_path(output_dir),
        )
    result = {
        "status": "completed",
        "output_dir": display_path(output_dir),
        "manifest": display_path(manifest),
        "best_model": display_path(output_dir / "best.pt") if (output_dir / "best.pt").exists() else "",
        "last_model": display_path(output_dir / "last.pt") if (output_dir / "last.pt").exists() else "",
        "best_iou": best_iou,
        "history": history,
        "config": config,
    }
    emit_progress(progress_callback, "completed", message="训练完成", progress=100, result=result)
    return result


def emit_progress(callback, status: str, **payload) -> None:
    if callback is None:
        message = payload.get("message")
        if message:
            print(message)
        return
    callback({"status": status, **payload})


def run_epoch(model, loader, device, optimizer, pos_weight, threshold: float, cancel_event: threading.Event | None = None) -> dict:
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
            raw_loss = F.binary_cross_entropy_with_logits(
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
    iou = tp / max(1, tp + fp + fn)
    dice = (2 * tp) / max(1, 2 * tp + fp + fn)
    return {
        "loss": totals["loss"] / valid,
        "iou": iou,
        "dice": dice,
        "valid_pixels": totals["valid"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_manifest_rows(path: Path) -> list[dict]:
    table = pd.read_csv(path, dtype=str).fillna("")
    rows = []
    for row in table.to_dict("records"):
        include = row.get("include") or row.get("included") or "true"
        if include.strip().lower() in {"0", "false", "no", "n"}:
            continue
        npz_path = resolve_project_path(row.get("npz_path", ""))
        if npz_path.exists():
            rows.append(row)
    return rows


def latest_manifest(region, patch_dir: Path | None) -> Path:
    if patch_dir:
        manifest = patch_dir / "manifest.csv" if patch_dir.is_dir() else patch_dir
        if manifest.exists():
            return manifest
        raise SystemExit(f"manifest not found: {manifest}")
    if region is None:
        raise SystemExit("all-scope training requires --manifest or --patch-dir with a combined manifest")
    root = region.processed_dir / "training_patches"
    manifests = sorted(root.glob("*/manifest.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not manifests:
        raise SystemExit(f"no patch manifest found under {root}; run scripts/export_training_patches.py first")
    return manifests[0]


def compute_normalization(rows: list[dict], max_patches: int = 0) -> Normalization:
    selected = rows if max_patches <= 0 else rows[:max_patches]
    sums = None
    squares = None
    count = 0
    for row in selected:
        with np.load(resolve_project_path(row["npz_path"])) as data:
            image = data["image"].astype("float64")
            valid = data["valid"].astype(bool)
        values = image[:, valid]
        if values.size == 0:
            continue
        if sums is None:
            sums = values.sum(axis=1)
            squares = (values * values).sum(axis=1)
        else:
            sums += values.sum(axis=1)
            squares += (values * values).sum(axis=1)
        count += values.shape[1]
    if sums is None or count == 0:
        raise SystemExit("cannot compute normalization: no valid pixels")
    mean = sums / count
    variance = np.maximum(squares / count - mean * mean, 1e-6)
    std = np.sqrt(variance)
    return Normalization(mean.astype("float32"), std.astype("float32"))


def compute_pos_weight(rows: list[dict]) -> float:
    positive = 0
    negative = 0
    for row in rows:
        with np.load(resolve_project_path(row["npz_path"])) as data:
            mask = data["mask"].astype("uint8")
            valid = data["valid"].astype(bool) & (mask != IGNORE_INDEX)
        positive += int(((mask == 1) & valid).sum())
        negative += int(((mask == 0) & valid).sum())
    if positive == 0:
        return 1.0
    return float(min(max(negative / positive, 1.0), 50.0))


def patch_channels(row: dict) -> int:
    with np.load(resolve_project_path(row["npz_path"])) as data:
        return int(data["image"].shape[0])


def augment_patch(image: np.ndarray, mask: np.ndarray, valid: np.ndarray):
    if random.random() < 0.5:
        image = image[:, :, ::-1]
        mask = mask[:, ::-1]
        valid = valid[:, ::-1]
    if random.random() < 0.5:
        image = image[:, ::-1, :]
        mask = mask[::-1, :]
        valid = valid[::-1, :]
    turns = random.randint(0, 3)
    if turns:
        image = np.rot90(image, turns, axes=(1, 2))
        mask = np.rot90(mask, turns, axes=(0, 1))
        valid = np.rot90(valid, turns, axes=(0, 1))
    return image, mask, valid


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, normalization: Normalization, history: list) -> None:
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
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def resolve_project_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# Compatibility for callers that imported the old training function directly.
train_unet = train_model


if __name__ == "__main__":
    main()
