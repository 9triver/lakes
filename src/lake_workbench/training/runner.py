"""Patch export and registered model training job adapters."""

import argparse
import sys
import threading
import time
from pathlib import Path

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.models.runtime import PIXEL_MLP_HIDDEN_CHANNELS, normalize_model_type
from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.logical_patches import build_logical_patches, build_training_dataset, training_dataset_status
from lake_workbench.utils import (
    clean_optional,
    display_path,
    parse_float_or_default,
    parse_int_or_default,
    read_csv_records,
    resolve_data_path,
    safe_filename,
    truthy_flag,
    write_csv_records,
)


REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def run_patch_export(region_key: str, options: dict) -> dict:
    return build_logical_patches(REGIONS[region_key])


def run_dataset_build(region_key: str, options: dict) -> dict:
    config_id = clean_optional(options.get("config_id")) or "resize256_v1"
    if not REGIONS[region_key].logical_patch_manifest.exists():
        return {"region": region_key, "config_id": config_id, "patches": 0, "skipped": True}
    return build_training_dataset(REGIONS[region_key], config_id)


def prepare_training_args(scope: str, options: dict) -> argparse.Namespace:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scope = scope if scope == "all" else (scope if scope in REGIONS else DEFAULT_REGION_KEY)
    model_type = normalize_model_type(options.get("model_type") or "unet")
    dataset_config_id = clean_optional(options.get("dataset_config_id")) or "resize256_v1"
    run_name = safe_filename(clean_optional(options.get("run_name")) or f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir = PROJECT_ROOT / "data" / "models" / scope / run_name
    manifest_text = clean_optional(options.get("manifest")) or ""
    patch_dir_text = clean_optional(options.get("patch_dir")) or ""
    manifest = resolve_data_path(manifest_text, REGIONS[DEFAULT_REGION_KEY]) if manifest_text else None
    patch_dir = resolve_data_path(patch_dir_text, REGIONS[DEFAULT_REGION_KEY]) if patch_dir_text else None
    if manifest is None and patch_dir is None:
        if scope == "all":
            manifest = build_combined_training_manifest(output_dir, dataset_config_id)
        else:
            status = training_dataset_status(REGIONS[scope], dataset_config_id)
            if not status["ready"]:
                raise RuntimeError(f"training dataset {dataset_config_id} is {status['status']} for {scope}")
            manifest = REGIONS[scope].training_dataset_dir / dataset_config_id / "manifest.csv"
    return argparse.Namespace(
        region=scope,
        model_type=model_type,
        dataset_config_id=dataset_config_id,
        manifest=manifest,
        patch_dir=patch_dir,
        output_dir=output_dir,
        epochs=parse_int_or_default(options.get("epochs"), 30),
        batch_size=parse_int_or_default(options.get("batch_size"), 8),
        lr=parse_float_or_default(options.get("lr"), 1e-3),
        weight_decay=parse_float_or_default(options.get("weight_decay"), 1e-4),
        base_channels=parse_int_or_default(options.get("base_channels"), 32),
        hidden_channels=options.get("hidden_channels") or list(PIXEL_MLP_HIDDEN_CHANNELS),
        val_ratio=parse_float_or_default(options.get("val_ratio"), 0.25),
        seed=parse_int_or_default(options.get("seed"), 42),
        num_workers=parse_int_or_default(options.get("num_workers"), 0),
        device=clean_optional(options.get("device")) or "auto",
        threshold=parse_float_or_default(options.get("threshold"), 0.5),
        pos_weight=clean_optional(options.get("pos_weight")) or "auto",
        max_norm_patches=parse_int_or_default(options.get("max_norm_patches"), 0),
        no_augment=truthy_flag(options.get("no_augment"), default=False),
        dry_run=truthy_flag(options.get("dry_run"), default=False),
    )


def build_combined_training_manifest(output_dir: Path, dataset_config_id: str = "resize256_v1") -> Path:
    rows = []
    for region in REGIONS.values():
        status = training_dataset_status(region, dataset_config_id)
        if status["status"] == "missing_logical":
            continue
        if not status["ready"]:
            raise RuntimeError(f"training dataset {dataset_config_id} is {status['status']} for {region.key}")
        manifest = region.training_dataset_dir / dataset_config_id / "manifest.csv"
        for row in read_csv_records(manifest):
            include = (row.get("include") or row.get("included") or "true").strip().lower()
            if include in {"0", "false", "no", "n"}:
                continue
            rows.append({**row, "source_region": region.key, "source_manifest": display_path(manifest), "dataset_config_id": dataset_config_id})
    if not rows:
        raise FileNotFoundError("no included patch rows found for all-region training")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    write_csv_records(manifest, rows)
    return manifest


def run_training_job(scope: str, options: dict, progress_callback=None, cancel_event: threading.Event | None = None) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_unet import train_model  # pyright: ignore[reportMissingImports]

    return train_model(prepare_training_args(scope, options), progress_callback=progress_callback, cancel_event=cancel_event)
