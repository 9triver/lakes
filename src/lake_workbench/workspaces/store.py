"""Workspace-owned samples, Patch catalogs, Dataset memberships, and artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import RegionConfig
from lake_workbench.utils import read_csv_records, write_csv_records
from lake_workbench.workspaces.membership import (
    find_patch_conflicts,
    included_members,
)


DEFAULT_WORKSPACE_ID = "default"
WORKSPACE_FORMAT_VERSION = 1


class WorkspaceError(ValueError):
    """Raised when a Workspace operation violates a domain invariant."""


class WorkspacePatchConflict(WorkspaceError):
    def __init__(self, conflicts: list[dict], patch_ids: Iterable[str] | None = None):
        self.conflicts = conflicts
        self.patch_ids = list(patch_ids or [])
        super().__init__("当前 Workspace Dataset 中存在重复或空间重叠 Patch")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class WorkspaceStore:
    """Own Workspace metadata, Patch manifests, datasets, and model artifacts."""

    def __init__(
        self,
        regions: dict[str, RegionConfig],
        root: Path | None = None,
        model_root: Path | None = None,
    ) -> None:
        self.regions = regions
        self.root = root or PROJECT_ROOT / "data" / "workspaces"
        self.model_root = model_root or PROJECT_ROOT / "data" / "models"
        self.registry_path = self.root / "workspaces.json"
        self._lock = threading.RLock()
        self._logical_rows_cache: dict[tuple[str, str], tuple[int, dict[str, dict]]] = {}

    def ensure_default_workspace(self) -> dict:
        with self._lock:
            registry = self._read_registry()
            if registry["items"]:
                return self.get(DEFAULT_WORKSPACE_ID)
            now = _timestamp()
            workspace = {
                "id": DEFAULT_WORKSPACE_ID,
                "name": "默认",
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "training_defaults": {},
            }
            registry["items"].append(workspace)
            self._write_registry(registry)
            return self._summary(workspace)

    @contextmanager
    def transaction(self):
        """Serialize a Workspace filesystem transaction within this server."""
        with self._lock:
            yield

    def workspace_samples_dir(self, workspace_id: str, region: str) -> Path:
        self._workspace_record(workspace_id)
        if region not in self.regions:
            raise WorkspaceError(f"Unknown region: {region}")
        return self.root / workspace_id / "samples" / region

    def workspace_training_samples(self, workspace_id: str, region: str) -> Path:
        return self.workspace_samples_dir(workspace_id, region) / "manifest.csv"

    def workspace_training_label_dir(self, workspace_id: str, region: str) -> Path:
        return self.workspace_samples_dir(workspace_id, region) / "labels"

    def workspace_derived_label_dir(self, workspace_id: str, region: str) -> Path:
        self._workspace_record(workspace_id)
        if region not in self.regions:
            raise WorkspaceError(f"Unknown region: {region}")
        return self.root / workspace_id / "derived_labels" / region

    def ensure_workspace_training_samples(self, workspace_id: str, region: str) -> Path:
        """Return the workspace sample manifest, migrating legacy default rows once."""
        path = self.workspace_training_samples(workspace_id, region)
        if path.exists():
            return path
        if workspace_id == DEFAULT_WORKSPACE_ID:
            legacy = self.regions[region].training_samples
            rows = read_csv_records(legacy)
            if rows:
                write_csv_records(path, rows)
        return path

    def list(self, include_archived: bool = False) -> dict:
        with self._lock:
            items = [
                self._summary(item)
                for item in self._read_registry()["items"]
                if include_archived or item.get("status") != "archived"
            ]
            items.sort(key=lambda item: (item["id"] != DEFAULT_WORKSPACE_ID, item["name"].casefold()))
            return {"default": DEFAULT_WORKSPACE_ID, "items": items}

    def get(self, workspace_id: str, *, allow_archived: bool = True) -> dict:
        with self._lock:
            workspace = self._workspace_record(workspace_id)
            if not allow_archived and workspace.get("status") == "archived":
                raise WorkspaceError(f"训练工作区已归档: {workspace_id}")
            return self._summary(workspace)

    def create(self, name: str) -> dict:
        name = str(name or "").strip()
        if not name:
            raise WorkspaceError("训练工作区名称不能为空")
        with self._lock:
            registry = self._read_registry()
            self._validate_unique_name(registry, name)
            workspace_id = uuid.uuid4().hex[:12]
            now = _timestamp()
            workspace = {
                "id": workspace_id,
                "name": name,
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "training_defaults": {},
            }
            registry["items"].append(workspace)
            self._write_registry(registry)
            return self._summary(workspace)

    def rename(self, workspace_id: str, name: str) -> dict:
        name = str(name or "").strip()
        if not name:
            raise WorkspaceError("训练工作区名称不能为空")
        with self._lock:
            registry = self._read_registry()
            workspace = self._workspace_record(workspace_id, registry)
            self._validate_unique_name(registry, name, exclude_id=workspace_id)
            workspace["name"] = name
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(workspace)

    def archive(self, workspace_id: str) -> dict:
        if workspace_id == DEFAULT_WORKSPACE_ID:
            raise WorkspaceError("默认训练工作区不能归档")
        with self._lock:
            registry = self._read_registry()
            workspace = self._workspace_record(workspace_id, registry)
            workspace["status"] = "archived"
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(workspace)

    def restore(self, workspace_id: str) -> dict:
        with self._lock:
            registry = self._read_registry()
            workspace = self._workspace_record(workspace_id, registry)
            workspace["status"] = "active"
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(workspace)

    def members(self, workspace_id: str, region: str = "") -> set[tuple[str, str]]:
        with self._lock:
            self._workspace_record(workspace_id)
            members: set[tuple[str, str]] = set()
            for key in ([region] if region else self.regions):
                path = self.workspace_logical_patch_manifest(workspace_id, key)
                if path.exists():
                    members.update(included_members(read_csv_records(path), key))
            return members

    def patch_counts(self, workspace_id: str, region: str) -> dict[str, int]:
        with self._lock:
            member_ids = {
                patch_id
                for _region, patch_id in self.members(workspace_id, region)
            }
            counts: dict[str, int] = {}
            for row in self._workspace_logical_index(workspace_id, region).values():
                if row.get("logical_patch_id") not in member_ids or not row.get(
                    "site_id"
                ):
                    continue
                counts[row["site_id"]] = counts.get(row["site_id"], 0) + 1
            return counts

    def update_members(
        self,
        workspace_id: str,
        region: str,
        patch_ids: Iterable[str],
        operation: str,
        *,
        replace: bool = False,
    ) -> dict:
        wanted = {str(value) for value in patch_ids if str(value)}
        if not wanted:
            raise WorkspaceError("logical_patch_ids is required")
        if operation not in {"include", "exclude", "restore"}:
            raise WorkspaceError("operation must be include, restore, or exclude")
        if region not in self.regions:
            raise WorkspaceError(f"Unknown region: {region}")
        with self._lock:
            workspace = self._workspace_record(workspace_id)
            if workspace.get("status") == "archived":
                raise WorkspaceError(f"训练工作区已归档: {workspace_id}")
            self.ensure_workspace_logical_patch_manifest(workspace_id, region)
            canonical = self._workspace_logical_index(workspace_id, region)
            missing = sorted(wanted - set(canonical))
            if missing:
                raise KeyError(f"logical patches not found: {', '.join(missing)}")
            current_ids = {
                patch_id for patch_id, row in canonical.items() if (region, patch_id) in included_members([row], region)
            }
            conflicts = find_patch_conflicts(canonical, current_ids, wanted) if operation in {"include", "restore"} else []
            if conflicts and not replace:
                raise WorkspacePatchConflict(conflicts, sorted(wanted))
            selected = set(current_ids)
            if operation in {"include", "restore"}:
                if replace:
                    selected.difference_update(
                        str(conflict.get("existing_patch_id"))
                        for conflict in conflicts
                        if conflict.get("existing_patch_id")
                    )
                selected.update(wanted)
            else:
                selected.difference_update(wanted)
            changed = current_ids ^ selected
            self._write_workspace_logical_rows(
                workspace_id,
                region,
                [
                    {
                        **row,
                        "include": "true" if patch_id in selected else "false",
                        "review_status": "included" if patch_id in selected else "excluded",
                        "exclude_reason": ""
                        if patch_id in wanted and operation in {"include", "restore"}
                        else row.get("exclude_reason", ""),
                    }
                    for patch_id, row in canonical.items()
                ],
            )
            self._refresh_status(workspace_id)
            return {
                "workspace_id": workspace_id,
                "operation": operation,
                "updated": len(changed),
                "logical_patch_ids": sorted(wanted),
            }

    def update_training_defaults(self, workspace_id: str, values: dict) -> dict:
        with self._lock:
            registry = self._read_registry()
            workspace = self._workspace_record(workspace_id, registry)
            workspace["training_defaults"] = dict(values)
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(workspace)

    def workspace_dataset_dir(self, workspace_id: str, region: str, config_id: str) -> Path:
        self._workspace_record(workspace_id)
        return self.root / workspace_id / "training_datasets" / region / config_id

    def workspace_training_patch_cache_dir(self, workspace_id: str, region: str, config_id: str) -> Path:
        self._workspace_record(workspace_id)
        return self.root / workspace_id / "training_patch_cache" / region / config_id

    def workspace_logical_patch_dir(self, workspace_id: str, region: str) -> Path:
        self._workspace_record(workspace_id)
        return self.root / workspace_id / "logical_patches" / region

    def workspace_logical_patch_manifest(self, workspace_id: str, region: str) -> Path:
        return self.workspace_logical_patch_dir(workspace_id, region) / "manifest.csv"

    def ensure_workspace_logical_patch_manifest(self, workspace_id: str, region: str) -> Path:
        """Return the Workspace-owned manifest path without importing shared state."""
        self._workspace_record(workspace_id)
        if region not in self.regions:
            raise WorkspaceError(f"Unknown region: {region}")
        path = self.workspace_logical_patch_manifest(workspace_id, region)
        return path

    def remove_workspace_logical_patches(self, sample_id: str) -> int:
        removed = 0
        with self._lock:
            for workspace in self._read_registry().get("items", []):
                workspace_id = workspace.get("id", "")
                for region in self.regions:
                    path = self.workspace_logical_patch_manifest(workspace_id, region)
                    if not path.exists():
                        continue
                    rows = read_csv_records(path)
                    kept = [row for row in rows if row.get("sample_id") != sample_id]
                    removed += len(rows) - len(kept)
                    if len(kept) != len(rows):
                        self._write_workspace_logical_rows(workspace_id, region, kept)
        return removed

    def workspace_model_dir(self, workspace_id: str, scope: str) -> Path:
        self._workspace_record(workspace_id)
        return self.model_root / "workspaces" / workspace_id / scope

    def assert_trainable(self, workspace_id: str) -> None:
        workspace = self.get(workspace_id, allow_archived=False)
        if workspace["status"] == "archived":
            raise WorkspaceError(f"训练工作区已归档: {workspace_id}")

    def _member_sites(self, workspace_id: str, members: set[tuple[str, str]]) -> set[str]:
        by_region: dict[str, set[str]] = {}
        for region, patch_id in members:
            by_region.setdefault(region, set()).add(patch_id)
        result = set()
        for region, wanted in by_region.items():
            for row in self._workspace_logical_index(workspace_id, region).values():
                if row.get("logical_patch_id") in wanted and row.get("site_id"):
                    result.add(row["site_id"])
        return result

    def _workspace_logical_index(self, workspace_id: str, region: str) -> dict[str, dict]:
        path = self.workspace_logical_patch_manifest(workspace_id, region)
        return self._logical_index_for_path((workspace_id, region), path)

    def _logical_index_for_path(self, cache_key: tuple[str, str], path: Path) -> dict[str, dict]:
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        cached = self._logical_rows_cache.get(cache_key)
        if cached and cached[0] == mtime:
            return cached[1]
        rows = {
            row.get("logical_patch_id", ""): row
            for row in read_csv_records(path)
            if row.get("logical_patch_id")
        }
        self._logical_rows_cache[cache_key] = (mtime, rows)
        return rows

    def _refresh_status(self, workspace_id: str) -> None:
        with self._lock:
            registry = self._read_registry()
            workspace = self._workspace_record(workspace_id, registry)
            if workspace.get("status") != "archived":
                workspace["status"] = "active"
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)

    def _summary(self, workspace: dict) -> dict:
        workspace_id = workspace["id"]
        members = self.members(workspace_id)
        return {
            **workspace,
            "selected_patch_count": len(members),
            "site_count": len(self._member_sites(workspace_id, members)) if members else 0,
            "default": workspace_id == DEFAULT_WORKSPACE_ID,
        }

    def _workspace_record(
        self,
        workspace_id: str,
        registry: dict | None = None,
        *,
        allow_archived: bool = True,
    ) -> dict:
        item = next((value for value in (registry or self._read_registry())["items"] if value.get("id") == workspace_id), None)
        if item is None:
            raise KeyError(f"Workspace not found: {workspace_id}")
        if not allow_archived and item.get("status") == "archived":
            raise WorkspaceError(f"训练工作区已归档: {workspace_id}")
        return item

    def _validate_unique_name(self, registry: dict, name: str, exclude_id: str = "") -> None:
        if any(item.get("id") != exclude_id and str(item.get("name", "")).casefold() == name.casefold() for item in registry["items"]):
            raise WorkspaceError(f"训练工作区名称已存在: {name}")

    def _read_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"version": WORKSPACE_FORMAT_VERSION, "default_workspace_id": DEFAULT_WORKSPACE_ID, "items": []}
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        payload.setdefault("items", [])
        return payload

    def _write_registry(self, payload: dict) -> None:
        _atomic_text(self.registry_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _write_workspace_logical_rows(self, workspace_id: str, region: str, rows: list[dict]) -> None:
        logical_dir = self.workspace_logical_patch_dir(workspace_id, region)
        owned_rows = [
            {
                **row,
                "preview_path": str(logical_dir / "preview" / f"{row.get('logical_patch_id', '')}.png"),
                "preview_base_path": str(logical_dir / "preview" / f"{row.get('logical_patch_id', '')}.base.png"),
            }
            for row in rows
        ]
        write_csv_records(self.workspace_logical_patch_manifest(workspace_id, region), owned_rows)
        self._logical_rows_cache.pop((workspace_id, region), None)
