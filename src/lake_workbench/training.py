"""Training sample identity, patch export, dataset summaries, and model metadata."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lake_workbench.region_config import RegionConfig, load_region_configs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "all"
REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def clean_optional(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def parse_float(value) -> float | None:
    text = clean_optional(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_float_or_default(value, default: float) -> float:
    parsed = parse_float(value)
    return default if parsed is None else parsed


def parse_int_or_default(value, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def truthy_flag(value, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none", "null"}:
            return default
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def split_commas(value) -> list[str]:
    text = clean_optional(value)
    return [item.strip() for item in text.split(",") if item.strip()] if text else []


def resolve_data_path(value, region: RegionConfig | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    region = region or REGIONS[DEFAULT_REGION_KEY]
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "data_download" and parts[1] == "downloads":
        return region.data_dir.joinpath(*parts[2:])
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path, dtype=str).fillna("").to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_csv_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def rounded_float(value, digits: int = 5):
    parsed = parse_float(value)
    return None if parsed is None else round(parsed, digits)


def normalized_view_extent(view_state: dict, digits: int = 5) -> list[float]:
    map_state = view_state.get("map") if isinstance(view_state.get("map"), dict) else {}
    extent = map_state.get("extent") if isinstance(map_state.get("extent"), list) else []
    if len(extent) != 4:
        return []
    values = [rounded_float(item, digits) for item in extent]
    return [] if any(item is None for item in values) else values


def bbox_iou(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    bottom = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    top = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def bbox_from_row(row: dict) -> list[float]:
    values = [
        rounded_float(row.get("view_west")),
        rounded_float(row.get("view_south")),
        rounded_float(row.get("view_east")),
        rounded_float(row.get("view_north")),
    ]
    return [] if any(item is None for item in values) else values


def training_view_signature(
    lake_id: str,
    product_key: str,
    label_source: str,
    label_threshold: str,
    label_scope: str,
    mask_policy: str,
    view_state: dict,
) -> tuple[str, str, list[float]]:
    visible = view_state.get("visible_layers") if isinstance(view_state.get("visible_layers"), dict) else {}
    label_layers = {
        key: bool(visible.get(key))
        for key in ("osm", "hydrolakes", "context_osm", "context_hydrolakes", "esa", "jrc", "local_label")
    }
    local_label = view_state.get("selected_local_label") if isinstance(view_state.get("selected_local_label"), dict) else {}
    extent = normalized_view_extent(view_state)
    base_payload = {
        "lake_id": lake_id,
        "product_key": product_key,
        "label_source": label_source,
        "label_threshold": str(label_threshold or ""),
        "label_scope": label_scope,
        "mask_policy": mask_policy,
        "label_layers": label_layers,
        "jrc_threshold": parse_int_or_default(view_state.get("jrc_threshold"), 75),
        "local_label_id": clean_optional(local_label.get("id")) or "",
        "local_label_path": clean_optional(local_label.get("path")) or "",
    }
    exact_payload = {**base_payload, "extent": extent}
    base_json = json.dumps(base_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    exact_json = json.dumps(exact_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        hashlib.sha1(exact_json.encode("utf-8")).hexdigest(),
        hashlib.sha1(base_json.encode("utf-8")).hexdigest(),
        extent,
    )


def run_patch_export(region_key: str, options: dict) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_training_patches import export_training_patches  # noqa: PLC0415

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


def latest_patch_manifest_for_region(region: RegionConfig) -> Path:
    root = region.processed_dir / "training_patches"
    manifests = sorted(root.glob("*/manifest.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"no patch manifest found under {display_path(root)}")
    return manifests[0]


def prepare_training_args(scope: str, options: dict) -> argparse.Namespace:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scope = scope if scope == "all" else (scope if scope in REGIONS else DEFAULT_REGION_KEY)
    run_name = clean_optional(options.get("run_name")) or f"unet_{time.strftime('%Y%m%d_%H%M%S')}"
    run_name = safe_filename(run_name)
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
            row = {**row, "source_region": region.key, "source_manifest": display_path(manifest)}
            rows.append(row)
    if not rows:
        raise FileNotFoundError("no included patch rows found for all-region training")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    write_csv_records(manifest, rows)
    return manifest


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def timestamp_for_path(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime))
    except OSError:
        return ""


def region_key_from_patch_row(row: dict, fallback: str = "") -> str:
    region = clean_optional(row.get("source_region") or row.get("region"))
    if region:
        return region
    lake_id = clean_optional(row.get("lake_id")) or ""
    if "_" in lake_id:
        prefix = lake_id.split("_", 1)[0]
        if prefix in REGIONS:
            return prefix
    return fallback


def summarize_training_manifest(manifest_path: Path, region_key: str = "") -> dict:
    rows = read_csv_records(manifest_path)
    included_rows = []
    excluded = 0
    usable = 0
    sample_ids = set()
    lake_ids = set()
    regions = set()
    water_pixels = 0
    valid_pixels = 0
    npz_shape = None
    first_npz = ""
    for row in rows:
        include = clean_optional(row.get("include") or row.get("included"))
        included = True if include is None else truthy_flag(include, default=True)
        if included:
            included_rows.append(row)
        else:
            excluded += 1
        sample_id = clean_optional(row.get("sample_id"))
        lake_id = clean_optional(row.get("lake_id"))
        if sample_id:
            sample_ids.add(sample_id)
        if lake_id:
            lake_ids.add(lake_id)
        row_region = region_key_from_patch_row(row, region_key)
        if row_region:
            regions.add(row_region)
        if included:
            water_pixels += parse_int_or_default(row.get("water_pixels"), 0)
            valid_pixels += parse_int_or_default(row.get("valid_pixels"), 0)
            npz_path_text = clean_optional(row.get("npz_path"))
            if npz_path_text:
                npz_path = resolve_data_path(npz_path_text, REGIONS.get(row_region or region_key) or REGIONS[DEFAULT_REGION_KEY])
                if npz_path.exists():
                    usable += 1
                    if npz_shape is None:
                        try:
                            with np.load(npz_path) as data:
                                npz_shape = list(data["image"].shape)
                            first_npz = display_path(npz_path)
                        except Exception:
                            npz_shape = None
    channels = int(npz_shape[0]) if npz_shape else 0
    patch_size = list(npz_shape[1:]) if npz_shape and len(npz_shape) >= 3 else []
    return {
        "manifest": display_path(manifest_path),
        "modified_at": timestamp_for_path(manifest_path),
        "total_patches": len(rows),
        "included_patches": len(included_rows),
        "excluded_patches": excluded,
        "usable_patches": usable,
        "sample_count": len(sample_ids),
        "lake_count": len(lake_ids),
        "regions": sorted(regions),
        "water_pixels": water_pixels,
        "valid_pixels": valid_pixels,
        "water_ratio": (water_pixels / valid_pixels) if valid_pixels else 0,
        "in_channels": channels,
        "patch_size": patch_size,
        "first_npz": first_npz,
    }


def merge_training_dataset_summaries(scope: str, summaries: list[dict]) -> dict:
    totals = {
        "scope": scope,
        "manifests": summaries,
        "total_patches": sum(item.get("total_patches", 0) for item in summaries),
        "included_patches": sum(item.get("included_patches", 0) for item in summaries),
        "excluded_patches": sum(item.get("excluded_patches", 0) for item in summaries),
        "usable_patches": sum(item.get("usable_patches", 0) for item in summaries),
        "sample_count": sum(item.get("sample_count", 0) for item in summaries),
        "lake_count": sum(item.get("lake_count", 0) for item in summaries),
        "water_pixels": sum(item.get("water_pixels", 0) for item in summaries),
        "valid_pixels": sum(item.get("valid_pixels", 0) for item in summaries),
    }
    regions = set()
    for item in summaries:
        regions.update(item.get("regions") or [])
    totals["regions"] = sorted(regions)
    totals["water_ratio"] = (totals["water_pixels"] / totals["valid_pixels"]) if totals["valid_pixels"] else 0
    first = next((item for item in summaries if item.get("in_channels")), {})
    totals["in_channels"] = first.get("in_channels", 0)
    totals["patch_size"] = first.get("patch_size", [])
    return totals


def current_training_dataset_summary(scope: str) -> dict:
    try:
        if scope == "all":
            summaries = []
            for region in REGIONS.values():
                try:
                    summaries.append(summarize_training_manifest(latest_patch_manifest_for_region(region), region.key))
                except FileNotFoundError:
                    continue
            if not summaries:
                raise FileNotFoundError("no patch manifest found for any region")
            return merge_training_dataset_summaries(scope, summaries)
        region = REGIONS.get(scope) or REGIONS[DEFAULT_REGION_KEY]
        summary = summarize_training_manifest(latest_patch_manifest_for_region(region), region.key)
        return merge_training_dataset_summaries(scope, [summary])
    except Exception as exc:  # noqa: BLE001 - reported in local UI.
        return {
            "scope": scope,
            "manifests": [],
            "total_patches": 0,
            "included_patches": 0,
            "excluded_patches": 0,
            "usable_patches": 0,
            "sample_count": 0,
            "lake_count": 0,
            "regions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def dataset_summary_from_config(config: dict) -> dict:
    manifest_text = clean_optional(config.get("manifest"))
    scope = clean_optional(config.get("scope") or config.get("region")) or "all"
    if not manifest_text:
        return {"scope": scope, "manifests": [], "error": "missing manifest"}
    manifest = resolve_data_path(manifest_text, REGIONS[DEFAULT_REGION_KEY])
    if not manifest.exists():
        return {"scope": scope, "manifests": [{"manifest": manifest_text}], "error": f"manifest not found: {manifest_text}"}
    summary = summarize_training_manifest(manifest, "" if scope == "all" else scope)
    merged = merge_training_dataset_summaries(scope, [summary])
    if config.get("train_count") is not None:
        merged["train_count"] = config.get("train_count")
    if config.get("val_count") is not None:
        merged["val_count"] = config.get("val_count")
    return merged


def model_training_metadata(model_path: Path, scope: str) -> dict:
    run_dir = model_path.parent
    config = read_json_file(run_dir / "config.json", {})
    if not isinstance(config, dict):
        config = {}
    history = read_json_file(run_dir / "history.json", [])
    if not isinstance(history, list):
        history = []
    best_iou = None
    best_epoch = None
    for record in history:
        val = record.get("val") or {}
        train = record.get("train") or {}
        score = parse_float(val.get("iou"))
        if score is None:
            score = parse_float(train.get("iou"))
        if score is None:
            continue
        if best_iou is None or score > best_iou:
            best_iou = score
            best_epoch = parse_int_or_default(record.get("epoch"), 0)
    latest = history[-1] if history else {}
    return {
        "run_name": run_dir.name,
        "scope": clean_optional(config.get("scope")) or scope,
        "config": {
            key: config.get(key)
            for key in (
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "base_channels",
                "val_ratio",
                "seed",
                "threshold",
                "no_augment",
                "device",
                "device_requested",
                "train_count",
                "val_count",
                "manifest",
            )
            if key in config
        },
        "dataset": dataset_summary_from_config(config) if config else {},
        "history_count": len(history),
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "latest": latest,
        "updated_at": timestamp_for_path(run_dir / "history.json") or timestamp_for_path(model_path),
    }


def model_sort_key(item: dict) -> tuple:
    score = parse_float(item.get("best_iou"))
    valid_rank = 1 if item.get("error") else 0
    missing_score = 1 if score is None else 0
    weight_rank = 0 if item.get("weight") == "best.pt" else 1
    return (
        valid_rank,
        missing_score,
        -(score or 0),
        weight_rank,
        -(parse_int_or_default(item.get("epoch"), 0)),
        item.get("label", ""),
    )


def run_training_job(scope: str, options: dict, progress_callback=None, cancel_event: threading.Event | None = None) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_unet import train_unet  # noqa: PLC0415

    args = prepare_training_args(scope, options)
    return train_unet(args, progress_callback=progress_callback, cancel_event=cancel_event)


def iter_global_model_paths() -> list[Path]:
    if not GLOBAL_MODEL_DIR.exists():
        return []
    return sorted(GLOBAL_MODEL_DIR.glob("*/*.pt"))


def global_model_key(path: Path) -> str:
    try:
        return f"all/{path.resolve().relative_to(GLOBAL_MODEL_DIR.resolve())}"
    except ValueError:
        return f"all/{path.name}"


def global_model_path_from_key(model_key: str) -> Path:
    key = clean_optional(model_key) or ""
    path = Path(key)
    if path.is_absolute():
        raise ValueError("absolute model paths are not allowed")
    parts = path.parts
    if len(parts) == 3 and parts[0] == "all":
        run_name, weight = parts[1], parts[2]
    elif len(parts) == 2:
        run_name, weight = parts
    else:
        raise ValueError(f"invalid model key: {key}")
    if run_name in {"", ".", ".."} or weight in {"", ".", ".."}:
        raise ValueError(f"invalid model key: {key}")
    return GLOBAL_MODEL_DIR / run_name / weight
def persisted_training_job(scope: str, run_dir: Path) -> dict | None:
    config = read_json_file(run_dir / "config.json", {})
    if not isinstance(config, dict) or not config:
        return None
    history = read_json_file(run_dir / "history.json", [])
    if not isinstance(history, list):
        history = []
    result = {
        "status": "completed",
        "output_dir": display_path(run_dir),
        "manifest": config.get("manifest") or display_path(run_dir / "manifest.csv"),
        "best_model": display_path(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
        "last_model": display_path(run_dir / "last.pt") if (run_dir / "last.pt").exists() else "",
        "history": history,
        "config": config,
    }
    if history:
        result["best_iou"] = max(
            (parse_float((record.get("val") or {}).get("iou")) or 0 for record in history),
            default=0,
        )
    status = "completed" if result["best_model"] or result["last_model"] or history else "configured"
    epoch = parse_int_or_default((history[-1] if history else {}).get("epoch"), 0)
    epochs = parse_int_or_default(config.get("epochs"), epoch)
    config_path = run_dir / "config.json"
    return {
        "job_id": run_dir.name,
        "scope": scope,
        "run_name": run_dir.name,
        "status": status,
        "message": "历史训练任务",
        "progress": 100 if status == "completed" else 5,
        "epoch": epoch,
        "epochs": epochs,
        "history": history,
        "config": config,
        "dataset": dataset_summary_from_config(config),
        "result": result,
        "output_dir": display_path(run_dir),
        "created_at": timestamp_for_path(config_path),
        "updated_at": timestamp_for_path(run_dir / "history.json") or timestamp_for_path(config_path),
        "persisted": True,
    }
