"""Dataset configuration and Workspace Dataset freshness checks."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import RegionConfig
from lake_workbench.utils import display_path, read_csv_records

if TYPE_CHECKING:
    from lake_workbench.workspaces import WorkspaceStore


DATASET_CONFIG_PATH = PROJECT_ROOT / "config" / "training_datasets.toml"


def dataset_configs(path: Path = DATASET_CONFIG_PATH) -> dict[str, dict]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        key: {"id": key, **value} for key, value in payload.get("datasets", {}).items()
    }


def workspace_dataset_signature(
    region: RegionConfig,
    config: dict,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
) -> str:
    members = {
        patch_id
        for _region, patch_id in workspace_store.members(workspace_id, region.key)
    }
    manifest_path = workspace_store.ensure_workspace_logical_patch_manifest(
        workspace_id, region.key
    )
    logical_rows = [
        row
        for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id") in members
    ]
    payload = {"config": config, "logical_rows": logical_rows}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def workspace_training_dataset_status(
    region: RegionConfig,
    config_id: str,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
) -> dict:
    config = dataset_configs()[config_id]
    output_dir = workspace_store.workspace_dataset_dir(
        workspace_id, region.key, config_id
    )
    metadata_path = output_dir / "build.json"
    manifest_path = output_dir / "manifest.csv"
    members = workspace_store.members(workspace_id, region.key)
    if not members:
        status, metadata = "missing_selection", {}
    elif not metadata_path.exists() or not manifest_path.exists():
        status, metadata = "missing", {}
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = (
                "ready"
                if metadata.get("signature")
                == workspace_dataset_signature(
                    region, config, workspace_store, workspace_id
                )
                else "stale"
            )
        except (OSError, json.JSONDecodeError):
            status, metadata = "stale", {}
    return {
        "workspace_id": workspace_id,
        "config": config,
        "config_id": config_id,
        "region": region.key,
        "status": status,
        "ready": status == "ready",
        "patches": int(metadata.get("patches") or 0),
        "manifest": display_path(manifest_path),
    }
