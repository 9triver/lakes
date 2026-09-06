"""Train a registered water-segmentation model from materialized Patch data."""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime
from pathlib import Path

from lake_workbench.models.runtime import (
    PIXEL_MLP_HIDDEN_CHANNELS,
    SUPPORTED_MODEL_TYPES,
    architecture_label,
    build_model,
    model_options,
    normalize_model_type,
)
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import DEFAULT_CONFIG_PATH, load_region_configs
from lake_workbench.training.datasets import split_rows_by_site
from lake_workbench.training.unet.data import (
    compute_normalization,
    compute_pos_weight,
    load_manifest_rows,
    make_patch_dataset,
    patch_channels,
)
from lake_workbench.training.unet.engine import (
    choose_device,
    emit_progress,
    format_epoch,
    run_epoch,
    save_checkpoint,
    seed_everything,
)
from lake_workbench.utils import display_path


REGIONS, DEFAULT_REGION_KEY = load_region_configs(DEFAULT_CONFIG_PATH)
MODEL_ROOT = PROJECT_ROOT / "data" / "models"
torch = None
DataLoader = None
Dataset = object
PatchDataset = None


def init_torch() -> None:
    global torch, DataLoader, Dataset, PatchDataset
    if torch is not None:
        return
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in this environment.\n"
            "Install it first, for example:\n"
            "  .venv/bin/python -m pip install torch\n"
            "Then rerun this script."
        ) from exc

    def patch_dataset(rows, normalization, augment=False):
        return make_patch_dataset(torch, Dataset, rows, normalization, augment=augment)

    PatchDataset = patch_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS) + ["all"], default=DEFAULT_REGION_KEY)
    parser.add_argument("--workspace", dest="workspace_id", default="default")
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default="unet")
    parser.add_argument("--dataset-config", dest="dataset_config_id", default="resize256_v1")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patch-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--hidden-channels",
        type=int,
        nargs=2,
        default=list(PIXEL_MLP_HIDDEN_CHANNELS),
        metavar=("HIDDEN_1", "HIDDEN_2"),
    )
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
    manifest = args.manifest or latest_manifest(
        region,
        args.patch_dir,
        getattr(args, "dataset_config_id", "resize256_v1"),
        getattr(args, "workspace_id", "default"),
    )
    output_dir = (
        args.output_dir
        or MODEL_ROOT
        / "workspaces"
        / getattr(args, "workspace_id", "default")
        / args.region
        / f"{selected_model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest)
    if not rows:
        raise SystemExit(f"no included training patches found: {manifest}")
    in_channels = patch_channels(rows[0])
    train_rows, val_rows = split_rows_by_site(rows, args.val_ratio, args.seed)
    normalization = compute_normalization(train_rows, max_patches=args.max_norm_patches)
    pos_weight = compute_pos_weight(train_rows) if args.pos_weight == "auto" else float(args.pos_weight)
    configured_hidden_channels = getattr(args, "hidden_channels", None)
    selected_model_options = model_options(
        selected_model_type,
        {
            "base_channels": args.base_channels,
            "model_options": {"hidden_channels": configured_hidden_channels} if configured_hidden_channels else {},
        },
    )

    config = {
        "format_version": 2,
        "workspace_id": getattr(args, "workspace_id", ""),
        "model_type": selected_model_type,
        "model_options": selected_model_options,
        "architecture_label": architecture_label(selected_model_type, in_channels, selected_model_options),
        "scope": args.region,
        "region": args.region if args.region != "all" else "",
        "regions": sorted(REGIONS) if args.region == "all" else [args.region],
        "manifest": display_path(manifest),
        "dataset_config_id": getattr(args, "dataset_config_id", ""),
        "dataset_source": getattr(args, "dataset_source", "workspace"),
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


def latest_manifest(region, patch_dir: Path | None, dataset_config_id: str = "resize256_v1", workspace_id: str = "default") -> Path:
    if patch_dir:
        manifest = patch_dir / "manifest.csv" if patch_dir.is_dir() else patch_dir
        if manifest.exists():
            return manifest
        raise SystemExit(f"manifest not found: {manifest}")
    if region is None:
        raise SystemExit("all-scope training requires --manifest or --patch-dir with a combined manifest")
    manifest = PROJECT_ROOT / "data" / "workspaces" / workspace_id / "training_datasets" / region.key / dataset_config_id / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"training dataset manifest not found: {manifest}; run scripts/build_training_dataset.py first")
    return manifest


if __name__ == "__main__":
    main()
