#!/usr/bin/env python3
"""Migrate persisted observation data to the canonical site identity schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lake_workbench.regions.config import load_region_configs  # noqa: E402


LEGACY_KEYS = {"lake_id", "lake_name", "lake_display_name", "lake"}
KEY_RENAMES = {
    "is_lake_native": "is_site_native",
    "applies_to_lake": "applies_to_site",
}
ASSET_TYPE_COLUMNS = {"imagery_asset_type", "imagery_asset_types"}
ASSET_SCOPE_COLUMNS = {"imagery_asset_scope", "imagery_asset_scopes"}


def canonicalize(value):
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if not isinstance(value, dict):
        return "site_native" if value == "lake_native" else value
    result = {
        KEY_RENAMES.get(key, key): canonicalize(item)
        for key, item in value.items()
        if key not in LEGACY_KEYS
    }
    for key in ("asset_scope", "imagery_asset_scope"):
        if result.get(key) == "lake":
            result[key] = "site"
    if not result.get("site_id") and value.get("lake_id"):
        result["site_id"] = value["lake_id"]
    if not result.get("site_name") and value.get("lake_name"):
        result["site_name"] = value["lake_name"]
    if not result.get("site") and isinstance(value.get("lake"), dict):
        result["site"] = canonicalize(value["lake"])
    return result


def migrate_csv(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return False
    original = frame.copy()
    if "site_id" not in frame and "lake_id" in frame:
        frame["site_id"] = frame["lake_id"]
    if "site_name" not in frame and "lake_name" in frame:
        frame["site_name"] = frame["lake_name"]
    if "view_state_json" in frame:
        frame["view_state_json"] = frame["view_state_json"].map(canonicalize_json_text)
    for column in ASSET_TYPE_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].str.replace("lake_native", "site_native", regex=False)
    for column in ASSET_SCOPE_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].map(
            lambda value: ",".join("site" if item == "lake" else item for item in value.split(","))
        )
    legacy_columns = [column for column in LEGACY_KEYS if column in frame]
    frame = frame.drop(columns=legacy_columns, errors="ignore")
    if frame.equals(original):
        return False
    frame.to_csv(path, index=False)
    return True


def canonicalize_json_text(value: str) -> str:
    if not value:
        return value
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    return json.dumps(canonicalize(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def migrate_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    migrated = canonicalize(payload)
    if migrated == payload:
        return False
    path.write_text(json.dumps(migrated, ensure_ascii=False), encoding="utf-8")
    return True


def main() -> None:
    regions, _default = load_region_configs()
    csv_paths = []
    json_paths = []
    for region in regions.values():
        csv_paths.extend([region.user_sentinel_index, region.training_samples])
        csv_paths.extend((region.processed_dir / "training_patches").glob("*/manifest.csv"))
        json_paths.extend(region.training_label_dir.glob("*.geojson"))
    csv_paths.extend((PROJECT_ROOT / "data" / "models").glob("*/*/manifest.csv"))
    json_paths.extend((PROJECT_ROOT / "data" / "model_predictions").rglob("*.geojson"))

    changed_csv = [path for path in csv_paths if migrate_csv(path)]
    changed_json = [path for path in json_paths if migrate_json(path)]
    print(f"migrated CSV files: {len(changed_csv)}")
    print(f"migrated JSON files: {len(changed_json)}")


if __name__ == "__main__":
    main()
