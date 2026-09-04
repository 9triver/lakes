"""Workspace-owned samples, Patch catalogs, Dataset memberships, and artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import RegionConfig
from lake_workbench.utils import clean_optional, read_csv_records, truthy_flag, write_csv_records


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


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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


def _patch_is_included(row: dict) -> bool:
    status = clean_optional(row.get("review_status"))
    if status in {"included", "excluded"}:
        return status == "included"
    return truthy_flag(row.get("include"), default=True)


def _patch_bbox(row: dict) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(row[key]) for key in ("bounds_left", "bounds_bottom", "bounds_right", "bounds_top"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def _patch_bbox_iou(first: dict, second: dict) -> float:
    left_box = _patch_bbox(first)
    right_box = _patch_bbox(second)
    if not left_box or not right_box:
        return 0.0
    left = max(left_box[0], right_box[0])
    bottom = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    top = min(left_box[3], right_box[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    first_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
    second_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class WorkspaceStore:
    """Own Workspace metadata, memberships, source choices, and artifacts."""

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
        self._sample_rows_cache: dict[tuple[str, str], dict[str, dict]] = {}
        self._variant_ids_cache: dict[tuple[str, str, str], list[str]] = {}
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
                "source_workspace_ids": [],
                "created_at": now,
                "updated_at": now,
                "training_defaults": {},
            }
            registry["items"].append(workspace)
            self._write_members(DEFAULT_WORKSPACE_ID, set())
            self._write_sources(DEFAULT_WORKSPACE_ID, {"sites": {}, "conflicts": {}})
            self._write_registry(registry)
            return self._summary(workspace)

    def workspace_samples_dir(self, workspace_id: str, region: str) -> Path:
        self._workspace_record(workspace_id)
        if region not in self.regions:
            raise WorkspaceError(f"Unknown region: {region}")
        return self.root / workspace_id / "samples" / region

    def workspace_training_samples(self, workspace_id: str, region: str) -> Path:
        return self.workspace_samples_dir(workspace_id, region) / "manifest.csv"

    def workspace_training_label_dir(self, workspace_id: str, region: str) -> Path:
        return self.workspace_samples_dir(workspace_id, region) / "labels"

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

    def invalidate_training_sample_cache(self, workspace_id: str, region: str) -> None:
        self._sample_rows_cache.pop((workspace_id, region), None)
        self._variant_ids_cache = {
            key: value
            for key, value in self._variant_ids_cache.items()
            if key[:2] != (workspace_id, region)
        }

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

    def create(self, name: str, mode: str = "empty", source_workspace_ids: Iterable[str] = ()) -> dict:
        name = str(name or "").strip()
        if not name:
            raise WorkspaceError("训练工作区名称不能为空")
        if mode not in {"empty", "union"}:
            raise WorkspaceError("训练工作区创建方式必须为空白或并集")
        source_ids = list(dict.fromkeys(str(value) for value in source_workspace_ids if str(value)))
        if mode == "union" and not source_ids:
            raise WorkspaceError("并集训练工作区至少需要一个来源训练工作区")
        if mode == "empty" and source_ids:
            raise WorkspaceError("空白训练工作区不能指定来源训练工作区")
        with self._lock:
            registry = self._read_registry()
            self._validate_unique_name(registry, name)
            sources = [self._workspace_record(value, registry) for value in source_ids]
            if any(item.get("status") != "active" for item in sources):
                raise WorkspaceError("来源训练工作区必须处于可用状态且没有待解决的来源冲突")
            workspace_id = uuid.uuid4().hex[:12]
            now = _timestamp()
            workspace = {
                "id": workspace_id,
                "name": name,
                "status": "active",
                "source_workspace_ids": source_ids,
                "created_at": now,
                "updated_at": now,
                "training_defaults": {},
            }
            members: set[tuple[str, str]] = set()
            source_state = {"sites": {}, "conflicts": {}}
            if mode == "union":
                for source_id in source_ids:
                    members.update(self.members(source_id))
                sites = set().union(
                    *(self._member_sites(source_id, self.members(source_id)) for source_id in source_ids)
                )
                for site_id in sorted(sites):
                    choices = []
                    for source_id in source_ids:
                        if site_id not in self._member_sites(source_id, self.members(source_id)):
                            continue
                        selected = self._read_sources(source_id)["sites"].get(site_id, {}).get("variant_ids", [])
                        choice = set(selected)
                        if choice and choice not in choices:
                            choices.append(choice)
                    if len(choices) <= 1:
                        if choices:
                            source_state["sites"][site_id] = {"variant_ids": sorted(choices[0])}
                    else:
                        source_state["conflicts"][site_id] = {
                            "candidate_variant_ids": sorted(set().union(*choices)),
                            "source_workspace_ids": source_ids,
                        }
                if source_state["conflicts"]:
                    workspace["status"] = "needs_resolution"
            registry["items"].append(workspace)
            self._write_members(workspace_id, members)
            self._write_sources(workspace_id, source_state)
            self._write_registry(registry)
            if mode == "union":
                self._write_union_logical_patch_manifests(workspace_id, source_ids)
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
            workspace["status"] = "needs_resolution" if self._read_sources(workspace_id)["conflicts"] else "active"
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(workspace)

    def members(self, workspace_id: str, region: str = "") -> set[tuple[str, str]]:
        self._workspace_record(workspace_id)
        manifest_members: set[tuple[str, str]] = set()
        for key in ([region] if region else self.regions):
            path = self.workspace_logical_patch_manifest(workspace_id, key)
            for row in read_csv_records(path):
                if row.get("logical_patch_id") and _patch_is_included(row):
                    manifest_members.add((key, str(row["logical_patch_id"])))
        if manifest_members:
            return manifest_members
        rows = read_csv_records(self._members_path(workspace_id))
        return {
            (str(row.get("region") or ""), str(row.get("logical_patch_id") or ""))
            for row in rows
            if row.get("logical_patch_id") and (not region or row.get("region") == region)
        }

    def patch_counts(self, workspace_id: str, region: str) -> dict[str, int]:
        member_ids = {patch_id for _region, patch_id in self.members(workspace_id, region)}
        counts: dict[str, int] = {}
        for row in self._workspace_logical_index(workspace_id, region).values():
            if row.get("logical_patch_id") not in member_ids or not row.get("site_id"):
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
            members = self.members(workspace_id)
            if operation in {"include", "restore"}:
                current_ids = {patch_id for member_region, patch_id in members if member_region == region} - wanted
                conflicts = []
                for patch_id in wanted:
                    incoming = canonical[patch_id]
                    incoming_image = incoming.get("image_fingerprint") or incoming.get("image_path")
                    for current_id in current_ids:
                        existing = canonical.get(current_id)
                        if not existing:
                            continue
                        existing_image = existing.get("image_fingerprint") or existing.get("image_path")
                        overlap = _patch_bbox_iou(incoming, existing)
                        if incoming_image and incoming_image == existing_image and overlap >= 0.9:
                            conflicts.append({
                                "type": "spatial_overlap",
                                "patch_id": patch_id,
                                "existing_patch_id": current_id,
                                "overlap": round(overlap, 6),
                            })
                            break
                if conflicts and not replace:
                    raise WorkspacePatchConflict(conflicts)
                if replace:
                    members.difference_update((region, item["existing_patch_id"]) for item in conflicts)
            before = set(members)
            if operation in {"include", "restore"}:
                members.update((region, patch_id) for patch_id in wanted)
            else:
                members.difference_update((region, patch_id) for patch_id in wanted)
            self._write_members(workspace_id, members)
            selected = {patch_id for member_region, patch_id in members if member_region == region}
            self._write_workspace_logical_rows(
                workspace_id,
                region,
                [
                    {
                        **row,
                        "include": "true" if patch_id in selected else "false",
                        "review_status": "included" if patch_id in selected else "excluded",
                        "exclude_reason": "" if operation in {"include", "restore"} else row.get("exclude_reason", ""),
                    }
                    for patch_id, row in canonical.items()
                ],
            )
            self._refresh_status(workspace_id)
            return {
                "workspace_id": workspace_id,
                "operation": operation,
                "updated": len(before ^ members),
                "logical_patch_ids": sorted(wanted),
            }

    def conflicts(self, workspace_id: str) -> dict:
        workspace = self.get(workspace_id)
        sources = self._read_sources(workspace_id)
        items = []
        for site_id, conflict in sorted(sources["conflicts"].items()):
            variants = [self.variant(value) for value in conflict.get("candidate_variant_ids", [])]
            items.append({"site_id": site_id, **conflict, "variants": variants})
        return {"workspace": workspace, "items": items, "total": len(items)}

    def resolve_sources(self, workspace_id: str, site_id: str, variant_ids: Iterable[str]) -> dict:
        selected = list(dict.fromkeys(str(value) for value in variant_ids if str(value)))
        if not selected:
            raise WorkspaceError("At least one source variant is required")
        with self._lock:
            self._workspace_record(workspace_id, allow_archived=False)
            sources = self._read_sources(workspace_id)
            conflict = sources["conflicts"].get(site_id)
            candidates = set((conflict or {}).get("candidate_variant_ids", []))
            if conflict and not set(selected).issubset(candidates):
                raise WorkspaceError("Selected source variants are not conflict candidates")
            for variant_id in selected:
                variant = self.variant(variant_id)
                if variant.get("site_id") != site_id:
                    raise WorkspaceError(f"Source variant does not belong to {site_id}: {variant_id}")
            sources["sites"][site_id] = {"variant_ids": selected}
            sources["conflicts"].pop(site_id, None)
            self._write_sources(workspace_id, sources)
            self._refresh_status(workspace_id)
            return {"workspace": self.get(workspace_id), "site_id": site_id, "variant_ids": selected}

    def selected_variants(self, workspace_id: str, site_id: str) -> list[dict]:
        sources = self._read_sources(workspace_id)
        return [self.variant(value) for value in sources["sites"].get(site_id, {}).get("variant_ids", [])]

    def variant(self, variant_id: str) -> dict:
        matches = list((self.root / "source_variants").glob(f"*/{variant_id}.json"))
        if not matches:
            raise KeyError(f"Source variant not found: {variant_id}")
        return json.loads(matches[0].read_text(encoding="utf-8"))

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
        if path.exists():
            self._sync_members_from_workspace_manifest(workspace_id, region, path)
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
                        removed_ids = {
                            row.get("logical_patch_id", "")
                            for row in rows
                            if row.get("sample_id") == sample_id
                        }
                        self._write_workspace_logical_rows(workspace_id, region, kept)
                        members = self.members(workspace_id)
                        members.difference_update((region, patch_id) for patch_id in removed_ids)
                        self._write_members(workspace_id, members)
        return removed

    def workspace_model_dir(self, workspace_id: str, scope: str) -> Path:
        self._workspace_record(workspace_id)
        return self.model_root / "workspaces" / workspace_id / scope

    def source_signature(self, workspace_id: str, site_id: str) -> str:
        variants = self.selected_variants(workspace_id, site_id)
        payload = [(item["id"], item["content_hash"]) for item in variants]
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def assert_trainable(self, workspace_id: str) -> None:
        workspace = self.get(workspace_id, allow_archived=False)
        if workspace["status"] == "archived":
            raise WorkspaceError(f"训练工作区已归档: {workspace_id}")

    def _variants_for_patch(self, workspace_id: str, region_key: str, patch: dict) -> list[str]:
        sample_id = patch.get("sample_id", "")
        cache_key = (workspace_id, region_key, sample_id)
        if cache_key in self._variant_ids_cache:
            return self._variant_ids_cache[cache_key]
        samples_key = (workspace_id, region_key)
        samples = self._sample_rows_cache.get(samples_key)
        if samples is None:
            path = self.ensure_workspace_training_samples(workspace_id, region_key)
            samples = {
                row.get("sample_id", ""): row
                for row in read_csv_records(path)
                if row.get("sample_id")
            }
            if not samples and path != self.regions[region_key].training_samples:
                samples = {
                    row.get("sample_id", ""): row
                    for row in read_csv_records(self.regions[region_key].training_samples)
                    if row.get("sample_id")
                }
            self._sample_rows_cache[samples_key] = samples
        sample = samples.get(sample_id)
        result = self._register_sample_variants(region_key, sample) if sample else []
        self._variant_ids_cache[cache_key] = result
        return result

    def _register_sample_variants(self, region_key: str, sample: dict) -> list[str]:
        region = self.regions[region_key]
        label_path = Path(str(sample.get("label_path") or ""))
        if not label_path.is_absolute():
            label_path = PROJECT_ROOT / label_path
        if not label_path.exists():
            return []
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        features = payload.get("features") if payload.get("type") == "FeatureCollection" else [payload]
        top_properties = payload.get("properties") or {}
        view_state = top_properties.get("view_state") if isinstance(top_properties.get("view_state"), dict) else {}
        if not view_state:
            try:
                view_state = json.loads(sample.get("view_state_json") or "{}")
            except json.JSONDecodeError:
                view_state = {}
        groups: dict[str, list[dict]] = {}
        fallback_sources = [
            value.strip()
            for value in str(sample.get("context_sources") or sample.get("label_source") or "").split(",")
            if value.strip() and value.strip() != "current_view"
        ]
        for feature in features or []:
            if not isinstance(feature, dict) or not feature.get("geometry"):
                continue
            properties = feature.get("properties") or {}
            source = clean_optional(properties.get("training_layer") or properties.get("source"))
            if not source and len(fallback_sources) == 1:
                source = fallback_sources[0]
            if source:
                groups.setdefault(source, []).append(
                    {"type": "Feature", "geometry": feature["geometry"], "properties": properties}
                )
        if not groups and payload.get("geometry") and fallback_sources:
            groups[fallback_sources[0]] = [
                {"type": "Feature", "geometry": payload["geometry"], "properties": payload.get("properties") or {}}
            ]
        local = view_state.get("selected_local_label") if isinstance(view_state.get("selected_local_label"), dict) else {}
        result = []
        for source, source_features in sorted(groups.items()):
            parameters: dict[str, Any] = {}
            if source == "jrc":
                parameters["threshold"] = int(float(top_properties.get("jrc_threshold") or sample.get("label_threshold") or 75))
            if source == "local_label":
                parameters = {
                    "label_id": clean_optional(local.get("id")) or "",
                    "label_path": clean_optional(local.get("path")) or "",
                }
            collection = {"type": "FeatureCollection", "features": source_features}
            content_hash = hashlib.sha256(_canonical_json(collection).encode("utf-8")).hexdigest()
            identity = {
                "site_id": sample.get("site_id", ""),
                "source": source,
                "parameters": parameters,
                "content_hash": content_hash,
            }
            variant_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
            record = {
                "id": variant_id,
                **identity,
                "region": region.key,
                "label": self._variant_label(source, parameters),
                "sample_ids": [sample.get("sample_id", "")],
                "snapshot": collection,
            }
            path = self.root / "source_variants" / str(sample.get("site_id") or "unknown") / f"{variant_id}.json"
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                record["sample_ids"] = sorted(set(existing.get("sample_ids", [])) | set(record["sample_ids"]))
            _atomic_text(path, json.dumps(record, ensure_ascii=False, indent=2))
            result.append(variant_id)
        return result

    @staticmethod
    def _variant_label(source: str, parameters: dict) -> str:
        labels = {
            "osm": "OSM",
            "hydrolakes": "HydroLAKES",
            "esa": "ESA",
            "jrc": "JRC",
            "local_label": "Local Label",
            "context_osm": "Context OSM",
            "context_hydrolakes": "Context HydroLAKES",
        }
        label = labels.get(source, source)
        if source == "jrc":
            return f"{label} @ {parameters.get('threshold', 75)}"
        if source == "local_label" and parameters.get("label_id"):
            return f"{label} @ {parameters['label_id']}"
        return label

    def _merge_patch_sources(self, workspace_id: str, region: str, patches: list[dict]) -> None:
        sources = self._read_sources(workspace_id)
        for patch in patches:
            site_id = clean_optional(patch.get("site_id"))
            if not site_id:
                continue
            incoming = set(self._variants_for_patch(workspace_id, region, patch))
            if not incoming:
                continue
            conflict = sources["conflicts"].get(site_id)
            if conflict:
                conflict["candidate_variant_ids"] = sorted(set(conflict.get("candidate_variant_ids", [])) | incoming)
                continue
            current = set(sources["sites"].get(site_id, {}).get("variant_ids", []))
            if not current:
                sources["sites"][site_id] = {"variant_ids": sorted(incoming)}
            elif current != incoming:
                sources["conflicts"][site_id] = {
                    "candidate_variant_ids": sorted(current | incoming),
                    "source_workspace_ids": [workspace_id],
                }
        self._write_sources(workspace_id, sources)

    def _cleanup_empty_sites(self, workspace_id: str, members: set[tuple[str, str]]) -> None:
        active_sites = self._member_sites(workspace_id, members)
        sources = self._read_sources(workspace_id)
        sources["sites"] = {key: value for key, value in sources["sites"].items() if key in active_sites}
        sources["conflicts"] = {key: value for key, value in sources["conflicts"].items() if key in active_sites}
        self._write_sources(workspace_id, sources)

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
                self._write_sources(workspace_id, {"sites": {}, "conflicts": {}})
            workspace["updated_at"] = _timestamp()
            self._write_registry(registry)

    def _summary(self, workspace: dict) -> dict:
        workspace_id = workspace["id"]
        members = self.members(workspace_id) if self._members_path(workspace_id).exists() else set()
        sources = self._read_sources(workspace_id)
        return {
            **workspace,
            "selected_patch_count": len(members),
            "site_count": len(self._member_sites(workspace_id, members)) if members else 0,
            "conflict_count": len(sources["conflicts"]),
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

    def _members_path(self, workspace_id: str) -> Path:
        return self.root / workspace_id / "patch_members.csv"

    def _write_members(self, workspace_id: str, members: set[tuple[str, str]]) -> None:
        lines = []
        with tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["region", "logical_patch_id"])
            writer.writeheader()
            for region, patch_id in sorted(members):
                writer.writerow({"region": region, "logical_patch_id": patch_id})
            handle.seek(0)
            lines.append(handle.read())
        _atomic_text(self._members_path(workspace_id), "".join(lines))

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

    def _sync_members_from_workspace_manifest(self, workspace_id: str, region: str, path: Path) -> None:
        members = self.members(workspace_id)
        synced = {(key, patch_id) for key, patch_id in members if key != region}
        synced.update(
            (region, row.get("logical_patch_id", ""))
            for row in read_csv_records(path)
            if row.get("logical_patch_id") and truthy_flag(row.get("include"), default=True)
        )
        if synced != members:
            self._write_members(workspace_id, synced)

    def _write_union_logical_patch_manifests(self, workspace_id: str, source_ids: list[str]) -> None:
        for region in self.regions:
            selected_rows: dict[str, dict] = {}
            for source_id in source_ids:
                path = self.ensure_workspace_logical_patch_manifest(source_id, region)
                members = {patch_id for _region, patch_id in self.members(source_id, region)}
                for row in read_csv_records(path):
                    patch_id = row.get("logical_patch_id", "")
                    if patch_id in members:
                        selected_rows.setdefault(patch_id, {**row, "include": "true"})
            if selected_rows:
                self._write_workspace_logical_rows(
                    workspace_id,
                    region,
                    [selected_rows[key] for key in sorted(selected_rows)],
                )

    def _sources_path(self, workspace_id: str) -> Path:
        return self.root / workspace_id / "site_sources.json"

    def _read_sources(self, workspace_id: str) -> dict:
        path = self._sources_path(workspace_id)
        if not path.exists():
            return {"sites": {}, "conflicts": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("sites", {})
        payload.setdefault("conflicts", {})
        return payload

    def _write_sources(self, workspace_id: str, payload: dict) -> None:
        _atomic_text(self._sources_path(workspace_id), json.dumps(payload, ensure_ascii=False, indent=2))
