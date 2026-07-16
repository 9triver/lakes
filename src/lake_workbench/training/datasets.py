"""Training patch manifests and dataset summaries."""

import json
import random
import time
from pathlib import Path

import numpy as np

from lake_workbench.regions.config import RegionConfig, load_region_configs
from lake_workbench.utils import (
    clean_optional,
    display_path,
    parse_int_or_default,
    read_csv_records,
    resolve_data_path,
    truthy_flag,
)


REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def split_rows_by_site(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split complete observation sites so related patches cannot cross the boundary."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group_key = row.get("site_id") or row.get("sample_id") or row.get("patch_id") or ""
        groups.setdefault(group_key, []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    val_group_count = 0 if len(keys) <= 1 else max(1, round(len(keys) * val_ratio))
    val_keys = set(keys[:val_group_count])
    train = [row for key in keys if key not in val_keys for row in groups[key]]
    val = [row for key in keys if key in val_keys for row in groups[key]]
    if not train and val:
        train, val = val, []
    return train, val


def latest_patch_manifest_for_region(region: RegionConfig, config_id: str = "resize256_v1") -> Path:
    manifest = region.training_dataset_dir / config_id / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"training dataset manifest not found: {display_path(manifest)}")
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
    site_id = clean_optional(row.get("site_id")) or ""
    if "_" in site_id:
        prefix = site_id.split("_", 1)[0]
        if prefix in REGIONS:
            return prefix
    return fallback


def summarize_training_manifest(manifest_path: Path, region_key: str = "") -> dict:
    rows = read_csv_records(manifest_path)
    included_rows = []
    excluded = 0
    usable = 0
    sample_ids = set()
    site_ids = set()
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
        site_id = clean_optional(row.get("site_id"))
        if sample_id:
            sample_ids.add(sample_id)
        if site_id:
            site_ids.add(site_id)
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
        "site_count": len(site_ids),
        "lake_count": len(site_ids),
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
        "site_count": sum(item.get("site_count", item.get("lake_count", 0)) for item in summaries),
        "lake_count": sum(item.get("site_count", item.get("lake_count", 0)) for item in summaries),
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


def current_training_dataset_summary(scope: str, config_id: str = "resize256_v1") -> dict:
    try:
        if scope == "all":
            summaries = []
            for region in REGIONS.values():
                try:
                    summaries.append(summarize_training_manifest(latest_patch_manifest_for_region(region, config_id), region.key))
                except FileNotFoundError:
                    continue
            if not summaries:
                raise FileNotFoundError("no patch manifest found for any region")
            return merge_training_dataset_summaries(scope, summaries)
        region = REGIONS.get(scope) or REGIONS[DEFAULT_REGION_KEY]
        summary = summarize_training_manifest(latest_patch_manifest_for_region(region, config_id), region.key)
        return merge_training_dataset_summaries(scope, [summary])
    except Exception as exc:
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
