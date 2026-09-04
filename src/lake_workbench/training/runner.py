"""Patch export and registered model training job adapters."""

import argparse
import sys
import threading
import time
from pathlib import Path

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.models.runtime import PIXEL_MLP_HIDDEN_CHANNELS, normalize_model_type
from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.logical_patches import (
    build_global_training_dataset,
    build_logical_patches,
    build_workspace_training_dataset,
    workspace_training_dataset_status,
)
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
from lake_workbench.workspaces import WorkspacePatchConflict


REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def run_patch_export(region_key: str, options: dict, workspace_store) -> dict:
    workspace_id = clean_optional(options.get("workspace_id"))
    if not workspace_id:
        raise ValueError("workspace_id is required")
    manifest_path = workspace_store.ensure_workspace_logical_patch_manifest(workspace_id, region_key)
    existing_rows = read_csv_records(manifest_path)
    before = {row.get("logical_patch_id", "") for row in existing_rows}
    output_dir = manifest_path.parent
    requested_sample_id = clean_optional(options.get("sample_id"))
    sample_path = workspace_store.ensure_workspace_training_samples(workspace_id, region_key)
    samples = read_csv_records(sample_path)
    sample_ids = {row.get("sample_id", "") for row in samples if row.get("sample_id")}
    if requested_sample_id:
        sample_ids.add(requested_sample_id)
    patch_size = parse_int_or_default(options.get("patch_size"), 512)
    stride = parse_int_or_default(options.get("stride"), patch_size)
    result = build_logical_patches(
        REGIONS[region_key],
        output_dir=output_dir,
        sample_ids=sample_ids,
        samples=samples,
        patch_size=patch_size,
        stride=stride,
    )
    after = {
        row.get("logical_patch_id", "")
        for row in read_csv_records(manifest_path)
    }
    created = sorted(value for value in after - before if value)
    rows_by_id = {
        row.get("logical_patch_id", ""): row
        for row in read_csv_records(manifest_path)
    }
    created = [
        patch_id
        for patch_id in created
        if rows_by_id.get(patch_id, {}).get("review_status") != "excluded"
        and truthy_flag(rows_by_id.get(patch_id, {}).get("include"), default=True)
    ]
    requested_patch_ids = [
        clean_optional(value)
        for value in options.get("patch_ids") or []
        if clean_optional(value)
    ]
    candidates = requested_patch_ids or created
    auto_excluded = [
        patch_id
        for patch_id, row in rows_by_id.items()
        if row.get("exclude_reason") == "no_water"
        and row.get("review_status") == "excluded"
    ]
    if auto_excluded:
        workspace_store.update_members(workspace_id, region_key, auto_excluded, "exclude")
    if candidates:
        try:
            workspace_store.update_members(
                workspace_id,
                region_key,
                candidates,
                "include",
                replace=truthy_flag(options.get("overwrite"), default=False)
                or truthy_flag(options.get("replace"), default=False),
            )
        except WorkspacePatchConflict as exc:
            exc.patch_ids = candidates
            raise
    result["workspace_id"] = workspace_id
    result["added_to_workspace"] = len(candidates)
    return result


def run_dataset_build(region_key: str, options: dict, workspace_store) -> dict:
    config_id = clean_optional(options.get("config_id")) or "resize256_v1"
    workspace_id = clean_optional(options.get("workspace_id"))
    if not workspace_id:
        raise ValueError("workspace_id is required")
    status = workspace_training_dataset_status(REGIONS[region_key], config_id, workspace_store, workspace_id)
    if status["status"] == "missing_selection":
        return {
            "region": region_key,
            "workspace_id": workspace_id,
            "config_id": config_id,
            "patches": 0,
            "skipped": True,
            "status": "missing_selection",
            "message": "该区域没有已选 Patch，已跳过",
        }
    return build_workspace_training_dataset(REGIONS[region_key], config_id, workspace_store, workspace_id)


def run_global_dataset_build(options: dict, workspace_store, dataset_registry) -> dict:
    scope = clean_optional(options.get("scope")) or "all"
    config_id = clean_optional(options.get("config_id")) or "resize256_v1"
    return build_global_training_dataset(scope, config_id, REGIONS, workspace_store, dataset_registry)


def prepare_training_args(scope: str, options: dict, workspace_store, dataset_registry=None) -> argparse.Namespace:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scope = scope if scope == "all" else (scope if scope in REGIONS else DEFAULT_REGION_KEY)
    model_type = normalize_model_type(options.get("model_type") or "unet")
    dataset_config_id = clean_optional(options.get("dataset_config_id")) or "resize256_v1"
    dataset_source = clean_optional(options.get("dataset_source")) or "workspace"
    if dataset_source not in {"workspace", "global"}:
        raise ValueError(f"unknown training dataset source: {dataset_source}")
    run_name = safe_filename(clean_optional(options.get("run_name")) or f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    workspace_id = clean_optional(options.get("workspace_id"))
    if not workspace_id:
        raise ValueError("workspace_id is required")
    output_dir = workspace_store.workspace_model_dir(workspace_id, scope) / run_name
    manifest_text = clean_optional(options.get("manifest")) or ""
    patch_dir_text = clean_optional(options.get("patch_dir")) or ""
    manifest = resolve_data_path(manifest_text, REGIONS[DEFAULT_REGION_KEY]) if manifest_text else None
    patch_dir = resolve_data_path(patch_dir_text, REGIONS[DEFAULT_REGION_KEY]) if patch_dir_text else None
    if manifest is None and patch_dir is None:
        if dataset_source == "global":
            if dataset_registry is None:
                raise RuntimeError("global Dataset registry is unavailable")
            build_global_training_dataset(scope, dataset_config_id, REGIONS, workspace_store, dataset_registry)
            source = dataset_registry.dataset_dir(scope, dataset_config_id) / "manifest.csv"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = output_dir / "manifest.csv"
            write_csv_records(manifest, read_csv_records(source))
        elif scope == "all":
            workspace_store.assert_trainable(workspace_id)
            for region in REGIONS.values():
                status = workspace_training_dataset_status(region, dataset_config_id, workspace_store, workspace_id)
                if status["status"] not in {"missing_selection", "ready"}:
                    build_workspace_training_dataset(region, dataset_config_id, workspace_store, workspace_id)
            manifest = build_combined_workspace_training_manifest(
                output_dir,
                dataset_config_id,
                workspace_store,
                workspace_id,
            )
        else:
            workspace_store.assert_trainable(workspace_id)
            status = workspace_training_dataset_status(
                REGIONS[scope], dataset_config_id, workspace_store, workspace_id
            )
            if not status["ready"]:
                build_workspace_training_dataset(REGIONS[scope], dataset_config_id, workspace_store, workspace_id)
            source = workspace_store.workspace_dataset_dir(workspace_id, scope, dataset_config_id) / "manifest.csv"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = output_dir / "manifest.csv"
            write_csv_records(manifest, read_csv_records(source))
    return argparse.Namespace(
        region=scope,
        workspace_id=workspace_id,
        model_type=model_type,
        dataset_config_id=dataset_config_id,
        dataset_source=dataset_source,
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


def build_combined_workspace_training_manifest(
    output_dir: Path,
    dataset_config_id: str,
    workspace_store,
    workspace_id: str,
) -> Path:
    rows = []
    for region in REGIONS.values():
        status = workspace_training_dataset_status(region, dataset_config_id, workspace_store, workspace_id)
        if status["status"] == "missing_selection":
            continue
        if not status["ready"]:
            raise RuntimeError(
                f"Workspace training dataset {dataset_config_id} is {status['status']} "
                f"for {workspace_id}/{region.key}"
            )
        source = workspace_store.workspace_dataset_dir(workspace_id, region.key, dataset_config_id) / "manifest.csv"
        rows.extend(
            {**row, "source_region": region.key, "source_manifest": display_path(source)}
            for row in read_csv_records(source)
        )
    if not rows:
        raise FileNotFoundError(f"no Workspace patches found for all-region training: {workspace_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    write_csv_records(manifest, rows)
    return manifest


def run_training_job(
    scope: str,
    options: dict,
    workspace_store,
    dataset_registry=None,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_unet import train_model  # pyright: ignore[reportMissingImports]

    return train_model(
        prepare_training_args(scope, options, workspace_store, dataset_registry),
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
