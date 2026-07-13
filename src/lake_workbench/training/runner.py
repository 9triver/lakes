"""Patch export and U-Net training job adapters."""

import argparse
import contextlib
import io
import sys
import threading
import time
from pathlib import Path

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.datasets import latest_patch_manifest_for_region
from lake_workbench.utils import (
    clean_optional,
    display_path,
    parse_float_or_default,
    parse_int_or_default,
    read_csv_records,
    resolve_data_path,
    safe_filename,
    split_commas,
    truthy_flag,
    write_csv_records,
)


REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def run_patch_export(region_key: str, options: dict) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_training_patches import export_training_patches

    patch_size = parse_int_or_default(options.get("patch_size"), 256)
    stride = parse_int_or_default(options.get("stride"), 128)
    min_valid_ratio = parse_float_or_default(options.get("min_valid_ratio"), 0.6)
    min_water_pixels = parse_int_or_default(options.get("min_water_pixels"), 1)
    negative_ratio = parse_float_or_default(options.get("negative_ratio"), 0.25)
    preview_scale = parse_int_or_default(options.get("preview_scale"), 2)
    preview_limit = parse_int_or_default(options.get("preview_limit"), 0)
    sample_ids = split_commas(options.get("sample_id") or options.get("sample_ids"))
    output_dir_text = clean_optional(options.get("output_dir")) or ""
    args = argparse.Namespace(
        region=region_key,
        sample_id=sample_ids or None,
        patch_size=patch_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        min_water_pixels=min_water_pixels,
        negative_ratio=negative_ratio,
        all_touched=truthy_flag(options.get("all_touched"), default=False),
        output_dir=resolve_data_path(output_dir_text, REGIONS[region_key]) if output_dir_text else None,
        overwrite=truthy_flag(options.get("overwrite"), default=False),
        preview_limit=preview_limit,
        preview_scale=max(1, preview_scale),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = export_training_patches(args)
    return {
        **result,
        "manifest": display_path(Path(result["manifest"])),
        "npz_dir": display_path(Path(result["npz_dir"])),
        "preview_dir": display_path(Path(result["preview_dir"])),
        "options": {
            "patch_size": patch_size,
            "stride": stride,
            "min_valid_ratio": min_valid_ratio,
            "min_water_pixels": min_water_pixels,
            "negative_ratio": negative_ratio,
            "preview_scale": max(1, preview_scale),
            "preview_limit": preview_limit,
            "all_touched": truthy_flag(options.get("all_touched"), default=False),
            "overwrite": truthy_flag(options.get("overwrite"), default=False),
            "sample_ids": sample_ids,
        },
    }


def prepare_training_args(scope: str, options: dict) -> argparse.Namespace:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scope = scope if scope == "all" else (scope if scope in REGIONS else DEFAULT_REGION_KEY)
    run_name = safe_filename(clean_optional(options.get("run_name")) or f"unet_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir = PROJECT_ROOT / "data" / "models" / scope / run_name
    manifest_text = clean_optional(options.get("manifest")) or ""
    patch_dir_text = clean_optional(options.get("patch_dir")) or ""
    manifest = resolve_data_path(manifest_text, REGIONS[DEFAULT_REGION_KEY]) if manifest_text else None
    patch_dir = resolve_data_path(patch_dir_text, REGIONS[DEFAULT_REGION_KEY]) if patch_dir_text else None
    if scope == "all" and manifest is None and patch_dir is None:
        manifest = build_combined_training_manifest(output_dir)
    return argparse.Namespace(
        region=scope,
        manifest=manifest,
        patch_dir=patch_dir,
        output_dir=output_dir,
        epochs=parse_int_or_default(options.get("epochs"), 30),
        batch_size=parse_int_or_default(options.get("batch_size"), 8),
        lr=parse_float_or_default(options.get("lr"), 1e-3),
        weight_decay=parse_float_or_default(options.get("weight_decay"), 1e-4),
        base_channels=parse_int_or_default(options.get("base_channels"), 32),
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


def build_combined_training_manifest(output_dir: Path) -> Path:
    rows = []
    for region in REGIONS.values():
        try:
            manifest = latest_patch_manifest_for_region(region)
        except FileNotFoundError:
            continue
        for row in read_csv_records(manifest):
            include = (row.get("include") or row.get("included") or "true").strip().lower()
            if include in {"0", "false", "no", "n"}:
                continue
            rows.append({**row, "source_region": region.key, "source_manifest": display_path(manifest)})
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
    from train_unet import train_unet

    return train_unet(prepare_training_args(scope, options), progress_callback=progress_callback, cancel_event=cancel_event)
