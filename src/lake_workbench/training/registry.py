"""Shared Dataset membership registry built from workspace Patch snapshots."""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.utils import display_path, read_csv_records, resolve_data_path, write_csv_records


class DatasetConflict(ValueError):
    def __init__(self, conflicts: list[dict]):
        self.conflicts = conflicts
        super().__init__("目标 Dataset 中存在重复或空间重叠 Patch")


def _number(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def _bbox(row: dict) -> tuple[float, float, float, float] | None:
    values = [_number(row, key) for key in ("bounds_left", "bounds_bottom", "bounds_right", "bounds_top")]
    return tuple(values) if all(value is not None for value in values) else None  # type: ignore[return-value]


def _bbox_iou(first: tuple[float, float, float, float] | None, second: tuple[float, float, float, float] | None) -> float:
    if not first or not second:
        return 0.0
    left = max(first[0], second[0])
    bottom = max(first[1], second[1])
    right = min(first[2], second[2])
    top = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class DatasetRegistry:
    """Persist global Dataset memberships without mutating source workspace Patches."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT / "data" / "global_datasets"

    def manifest_path(self, scope: str) -> Path:
        return self.root / scope / "manifest.csv"

    def dataset_dir(self, scope: str, config_id: str) -> Path:
        return self.root / scope / "datasets" / config_id

    def list(self, scope: str) -> dict:
        rows = read_csv_records(self.manifest_path(scope))
        versions = sorted((self.root / scope / "versions").glob("*.csv")) if (self.root / scope / "versions").exists() else []
        return {
            "dataset_id": scope,
            "scope": scope,
            "total": len(rows),
            "items": rows,
            "manifest": display_path(self.manifest_path(scope)),
            "latest_version": versions[-1].stem if versions else "",
        }

    def contribution_scopes(self, workspace_id: str, patch_ids: list[str]) -> dict[str, list[str]]:
        """Return shared Dataset scopes containing each Workspace Patch."""
        wanted = set(patch_ids)
        result = {patch_id: [] for patch_id in wanted}
        if not wanted or not self.root.exists():
            return result
        for manifest in sorted(self.root.glob("*/manifest.csv")):
            scope = manifest.parent.name
            for row in read_csv_records(manifest):
                patch_id = row.get("source_patch_id", "")
                if row.get("source_workspace_id") == workspace_id and patch_id in wanted:
                    result[patch_id].append(scope)
        return result

    def withdraw(self, workspace_id: str, scope: str, patch_ids: list[str]) -> dict:
        """Remove contributed snapshots without modifying their source Workspace Patches."""
        wanted = set(patch_ids)
        path = self.manifest_path(scope)
        existing = read_csv_records(path)
        removed = [
            row for row in existing
            if row.get("source_workspace_id") == workspace_id and row.get("source_patch_id") in wanted
        ]
        final_rows = [row for row in existing if row not in removed]
        write_csv_records(path, final_rows)
        version = time.strftime("v%Y%m%d_%H%M%S")
        version_path = self.root / scope / "versions" / f"{version}.csv"
        write_csv_records(version_path, final_rows)
        return {
            "dataset_id": scope,
            "scope": scope,
            "removed": len(removed),
            "missing": sorted(wanted - {row.get("source_patch_id", "") for row in removed}),
            "total": len(final_rows),
            "manifest": display_path(path),
            "version": version,
            "version_manifest": display_path(version_path),
        }

    def contribute(
        self,
        workspace_store,
        workspace_id: str,
        scope: str,
        region: str,
        patch_ids: list[str],
        *,
        replace: bool = False,
    ) -> dict:
        if scope != "all" and scope not in workspace_store.regions:
            raise ValueError(f"Unknown Dataset scope: {scope}")
        if region not in workspace_store.regions:
            raise ValueError(f"Unknown Patch region: {region}")
        source_manifest = workspace_store.ensure_workspace_logical_patch_manifest(workspace_id, region)
        source_rows = {
            row.get("logical_patch_id", ""): row
            for row in read_csv_records(source_manifest)
            if row.get("logical_patch_id")
        }
        wanted = [source_rows[patch_id] for patch_id in dict.fromkeys(patch_ids) if patch_id in source_rows]
        if len(wanted) != len(set(patch_ids)):
            missing = sorted(set(patch_ids) - set(source_rows))
            raise KeyError(f"logical patches not found: {', '.join(missing)}")
        excluded = [
            row.get("logical_patch_id", "")
            for row in wanted
            if (row.get("review_status") == "excluded" or str(row.get("include", "true")).lower() in {"false", "0", "no"})
        ]
        if excluded:
            raise ValueError(f"只能贡献已包含的 Patch: {', '.join(excluded)}")

        path = self.manifest_path(scope)
        existing = read_csv_records(path)
        source_keys = {(row.get("source_workspace_id", ""), row.get("source_patch_id", "")) for row in existing}
        conflicts: list[dict] = []
        for row in wanted:
            source_patch_id = row.get("logical_patch_id", "")
            source_key = (workspace_id, source_patch_id)
            if source_key in source_keys:
                conflicts.append({"type": "duplicate", "patch_id": source_patch_id})
                continue
            for old in existing:
                if old.get("region") != region:
                    continue
                same_image = old.get("image_fingerprint") == row.get("image_fingerprint") and row.get("image_fingerprint")
                overlap = _bbox_iou(_bbox(old), _bbox(row))
                if same_image and overlap >= 0.9:
                    conflicts.append({
                        "type": "spatial_overlap",
                        "patch_id": source_patch_id,
                        "existing_patch_id": old.get("source_patch_id", ""),
                        "overlap": round(overlap, 6),
                    })
                    break
        if conflicts and not replace:
            raise DatasetConflict(conflicts)

        if replace and conflicts:
            remove_keys = {
                conflict.get("existing_patch_id")
                for conflict in conflicts
                if conflict.get("existing_patch_id")
            }
            existing = [row for row in existing if row.get("source_patch_id") not in remove_keys]

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        added = []
        for row in wanted:
            source_patch_id = row.get("logical_patch_id", "")
            if any(item.get("source_workspace_id") == workspace_id and item.get("source_patch_id") == source_patch_id for item in existing):
                continue
            label_snapshot = ""
            label_path = resolve_data_path(row.get("label_path", ""), workspace_store.regions[region]) if row.get("label_path") else None
            if label_path and label_path.exists():
                label_snapshot = label_path.read_text(encoding="utf-8")
            added.append({
                **row,
                "global_dataset_id": scope,
                "source_workspace_id": workspace_id,
                "source_patch_id": source_patch_id,
                "source_variant_ids": "",
                "source_variants_json": "[]",
                "label_snapshot_json": label_snapshot,
                "contributed_at": timestamp,
                "contribution_status": "accepted",
            })
        final_rows = existing + added
        write_csv_records(path, final_rows)
        version = time.strftime("v%Y%m%d_%H%M%S")
        version_path = self.root / scope / "versions" / f"{version}.csv"
        write_csv_records(version_path, final_rows)
        return {
            "dataset_id": scope,
            "scope": scope,
            "added": len(added),
            "replaced": len([item for item in conflicts if item.get("existing_patch_id")]),
            "total": len(existing) + len(added),
            "conflicts": conflicts,
            "manifest": display_path(path),
            "version": version,
            "version_manifest": display_path(version_path),
        }

    def contribute_many(
        self,
        workspace_store,
        workspace_id: str,
        scope: str,
        source_groups: dict[str, list[str]],
        *,
        replace: bool = False,
    ) -> dict:
        """Validate a cross-region contribution before committing one Dataset version."""
        path = self.manifest_path(scope)
        with tempfile.TemporaryDirectory() as directory:
            staged = DatasetRegistry(Path(directory))
            write_csv_records(staged.manifest_path(scope), read_csv_records(path))
            results = [
                staged.contribute(
                    workspace_store,
                    workspace_id,
                    scope,
                    region,
                    patch_ids,
                    replace=replace,
                )
                for region, patch_ids in sorted(source_groups.items())
                if patch_ids
            ]
            final_rows = read_csv_records(staged.manifest_path(scope))
        write_csv_records(path, final_rows)
        version = time.strftime("v%Y%m%d_%H%M%S")
        version_path = self.root / scope / "versions" / f"{version}.csv"
        write_csv_records(version_path, final_rows)
        return {
            "dataset_id": scope,
            "scope": scope,
            "added": sum(item["added"] for item in results),
            "replaced": sum(item["replaced"] for item in results),
            "total": len(final_rows),
            "conflicts": [conflict for item in results for conflict in item.get("conflicts", [])],
            "manifest": display_path(path),
            "version": version,
            "version_manifest": display_path(version_path),
        }
