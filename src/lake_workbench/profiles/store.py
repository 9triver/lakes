"""Global training Profiles over shared samples and user-owned logical patches."""

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


DEFAULT_PROFILE_ID = "default"
PROFILE_FORMAT_VERSION = 1


class ProfileError(ValueError):
    """Raised when a Profile operation violates a domain invariant."""


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


class ProfileStore:
    """Own Profile metadata, memberships, source choices, and migration."""

    def __init__(
        self,
        regions: dict[str, RegionConfig],
        root: Path | None = None,
        model_root: Path | None = None,
    ) -> None:
        self.regions = regions
        self.root = root or PROJECT_ROOT / "data" / "profiles"
        self.model_root = model_root or PROJECT_ROOT / "data" / "models"
        self.registry_path = self.root / "profiles.json"
        self._lock = threading.RLock()
        self._sample_rows_cache: dict[str, dict[str, dict]] = {}
        self._variant_ids_cache: dict[tuple[str, str], list[str]] = {}
        self._logical_rows_cache: dict[tuple[str, str], tuple[int, dict[str, dict]]] = {}

    def ensure_default_profile(self) -> dict:
        with self._lock:
            registry = self._read_registry()
            if registry["items"]:
                self.ensure_profile_logical_patch_manifests()
                return self.get(DEFAULT_PROFILE_ID)
            now = _timestamp()
            profile = {
                "id": DEFAULT_PROFILE_ID,
                "name": "默认",
                "status": "active",
                "source_profile_ids": [],
                "created_at": now,
                "updated_at": now,
                "training_defaults": {},
            }
            registry["items"].append(profile)
            members: set[tuple[str, str]] = set()
            site_source_sets: dict[str, list[set[str]]] = {}
            for region_key, region in self.regions.items():
                for row in self._logical_index(region_key).values():
                    if not truthy_flag(row.get("include"), default=True):
                        continue
                    patch_id = clean_optional(row.get("logical_patch_id"))
                    site_id = clean_optional(row.get("site_id"))
                    if not patch_id:
                        continue
                    members.add((region_key, patch_id))
                    if site_id:
                        variants = set(self._variants_for_patch(region_key, row))
                        if variants and variants not in site_source_sets.setdefault(site_id, []):
                            site_source_sets[site_id].append(variants)
            sources = {"sites": {}, "conflicts": {}}
            for site_id, choices in site_source_sets.items():
                if len(choices) == 1:
                    sources["sites"][site_id] = {"variant_ids": sorted(choices[0])}
                else:
                    candidates = sorted(set().union(*choices))
                    sources["conflicts"][site_id] = {
                        "candidate_variant_ids": candidates,
                        "source_profile_ids": [DEFAULT_PROFILE_ID],
                    }
            if sources["conflicts"]:
                profile["status"] = "needs_resolution"
            self._write_members(DEFAULT_PROFILE_ID, members)
            self._write_sources(DEFAULT_PROFILE_ID, sources)
            self._write_registry(registry)
            self.ensure_profile_logical_patch_manifests()
            return self._summary(profile)

    def ensure_profile_logical_patch_manifests(self) -> None:
        """Migrate legacy logical rows into user-owned manifests once."""
        with self._lock:
            for profile in self._read_registry().get("items", []):
                if profile.get("status") == "archived":
                    continue
                for region in self.regions:
                    self.ensure_profile_logical_patch_manifest(profile["id"], region)

    def list(self, include_archived: bool = False) -> dict:
        with self._lock:
            items = [
                self._summary(item)
                for item in self._read_registry()["items"]
                if include_archived or item.get("status") != "archived"
            ]
            items.sort(key=lambda item: (item["id"] != DEFAULT_PROFILE_ID, item["name"].casefold()))
            return {"default": DEFAULT_PROFILE_ID, "items": items}

    def get(self, profile_id: str, *, allow_archived: bool = True) -> dict:
        with self._lock:
            profile = self._profile_record(profile_id)
            if not allow_archived and profile.get("status") == "archived":
                raise ProfileError(f"用户已归档: {profile_id}")
            return self._summary(profile)

    def create(self, name: str, mode: str = "empty", source_profile_ids: Iterable[str] = ()) -> dict:
        name = str(name or "").strip()
        if not name:
            raise ProfileError("用户名称不能为空")
        if mode not in {"empty", "union"}:
            raise ProfileError("用户创建方式必须为空白或并集")
        source_ids = list(dict.fromkeys(str(value) for value in source_profile_ids if str(value)))
        if mode == "union" and not source_ids:
            raise ProfileError("并集用户至少需要一个来源用户")
        if mode == "empty" and source_ids:
            raise ProfileError("空白用户不能指定来源用户")
        with self._lock:
            registry = self._read_registry()
            self._validate_unique_name(registry, name)
            sources = [self._profile_record(value, registry) for value in source_ids]
            if any(item.get("status") != "active" for item in sources):
                raise ProfileError("来源用户必须处于可用状态且没有待解决的来源冲突")
            profile_id = uuid.uuid4().hex[:12]
            now = _timestamp()
            profile = {
                "id": profile_id,
                "name": name,
                "status": "active",
                "source_profile_ids": source_ids,
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
                            "source_profile_ids": source_ids,
                        }
                if source_state["conflicts"]:
                    profile["status"] = "needs_resolution"
            registry["items"].append(profile)
            self._write_members(profile_id, members)
            self._write_sources(profile_id, source_state)
            self._write_registry(registry)
            if mode == "union":
                self._write_union_logical_patch_manifests(profile_id, source_ids)
            return self._summary(profile)

    def rename(self, profile_id: str, name: str) -> dict:
        name = str(name or "").strip()
        if not name:
            raise ProfileError("用户名称不能为空")
        with self._lock:
            registry = self._read_registry()
            profile = self._profile_record(profile_id, registry)
            self._validate_unique_name(registry, name, exclude_id=profile_id)
            profile["name"] = name
            profile["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(profile)

    def archive(self, profile_id: str) -> dict:
        if profile_id == DEFAULT_PROFILE_ID:
            raise ProfileError("默认用户不能归档")
        with self._lock:
            registry = self._read_registry()
            profile = self._profile_record(profile_id, registry)
            profile["status"] = "archived"
            profile["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(profile)

    def restore(self, profile_id: str) -> dict:
        with self._lock:
            registry = self._read_registry()
            profile = self._profile_record(profile_id, registry)
            profile["status"] = "needs_resolution" if self._read_sources(profile_id)["conflicts"] else "active"
            profile["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(profile)

    def members(self, profile_id: str, region: str = "") -> set[tuple[str, str]]:
        self._profile_record(profile_id)
        rows = read_csv_records(self._members_path(profile_id))
        return {
            (str(row.get("region") or ""), str(row.get("logical_patch_id") or ""))
            for row in rows
            if row.get("logical_patch_id") and (not region or row.get("region") == region)
        }

    def patch_counts(self, profile_id: str, region: str) -> dict[str, int]:
        member_ids = {patch_id for _region, patch_id in self.members(profile_id, region)}
        counts: dict[str, int] = {}
        for row in self._profile_logical_index(profile_id, region).values():
            if row.get("logical_patch_id") not in member_ids or not row.get("site_id"):
                continue
            counts[row["site_id"]] = counts.get(row["site_id"], 0) + 1
        return counts

    def update_members(
        self,
        profile_id: str,
        region: str,
        patch_ids: Iterable[str],
        operation: str,
    ) -> dict:
        wanted = {str(value) for value in patch_ids if str(value)}
        if not wanted:
            raise ProfileError("logical_patch_ids is required")
        if operation not in {"include", "exclude", "restore"}:
            raise ProfileError("operation must be include, restore, or exclude")
        if region not in self.regions:
            raise ProfileError(f"Unknown region: {region}")
        with self._lock:
            profile = self._profile_record(profile_id)
            if profile.get("status") == "archived":
                raise ProfileError(f"用户已归档: {profile_id}")
            self.ensure_profile_logical_patch_manifest(profile_id, region)
            canonical = self._profile_logical_index(profile_id, region)
            missing = sorted(wanted - set(canonical))
            if missing and operation in {"include", "restore"}:
                legacy = self._logical_index(region)
                importable = [legacy[patch_id] for patch_id in missing if patch_id in legacy]
                if importable:
                    self._write_profile_logical_rows(
                        profile_id,
                        region,
                        [*canonical.values(), *importable],
                    )
                    canonical = self._profile_logical_index(profile_id, region)
                    missing = sorted(wanted - set(canonical))
            if missing:
                raise KeyError(f"logical patches not found: {', '.join(missing)}")
            members = self.members(profile_id)
            before = set(members)
            if operation in {"include", "restore"}:
                members.update((region, patch_id) for patch_id in wanted)
                self._merge_patch_sources(profile_id, region, [canonical[value] for value in wanted])
            else:
                members.difference_update((region, patch_id) for patch_id in wanted)
            self._write_members(profile_id, members)
            selected = {patch_id for member_region, patch_id in members if member_region == region}
            self._write_profile_logical_rows(
                profile_id,
                region,
                [
                    {**row, "include": "true" if patch_id in selected else "false"}
                    for patch_id, row in canonical.items()
                ],
            )
            self._cleanup_empty_sites(profile_id, members)
            self._refresh_status(profile_id)
            return {
                "profile_id": profile_id,
                "operation": operation,
                "updated": len(before ^ members),
                "logical_patch_ids": sorted(wanted),
            }

    def conflicts(self, profile_id: str) -> dict:
        profile = self.get(profile_id)
        sources = self._read_sources(profile_id)
        items = []
        for site_id, conflict in sorted(sources["conflicts"].items()):
            variants = [self.variant(value) for value in conflict.get("candidate_variant_ids", [])]
            items.append({"site_id": site_id, **conflict, "variants": variants})
        return {"profile": profile, "items": items, "total": len(items)}

    def resolve_sources(self, profile_id: str, site_id: str, variant_ids: Iterable[str]) -> dict:
        selected = list(dict.fromkeys(str(value) for value in variant_ids if str(value)))
        if not selected:
            raise ProfileError("At least one source variant is required")
        with self._lock:
            self._profile_record(profile_id, allow_archived=False)
            sources = self._read_sources(profile_id)
            conflict = sources["conflicts"].get(site_id)
            candidates = set((conflict or {}).get("candidate_variant_ids", []))
            if conflict and not set(selected).issubset(candidates):
                raise ProfileError("Selected source variants are not conflict candidates")
            for variant_id in selected:
                variant = self.variant(variant_id)
                if variant.get("site_id") != site_id:
                    raise ProfileError(f"Source variant does not belong to {site_id}: {variant_id}")
            sources["sites"][site_id] = {"variant_ids": selected}
            sources["conflicts"].pop(site_id, None)
            self._write_sources(profile_id, sources)
            self._refresh_status(profile_id)
            return {"profile": self.get(profile_id), "site_id": site_id, "variant_ids": selected}

    def selected_variants(self, profile_id: str, site_id: str) -> list[dict]:
        sources = self._read_sources(profile_id)
        return [self.variant(value) for value in sources["sites"].get(site_id, {}).get("variant_ids", [])]

    def variant(self, variant_id: str) -> dict:
        matches = list((self.root / "source_variants").glob(f"*/{variant_id}.json"))
        if not matches:
            raise KeyError(f"Source variant not found: {variant_id}")
        return json.loads(matches[0].read_text(encoding="utf-8"))

    def update_training_defaults(self, profile_id: str, values: dict) -> dict:
        with self._lock:
            registry = self._read_registry()
            profile = self._profile_record(profile_id, registry)
            profile["training_defaults"] = dict(values)
            profile["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(profile)

    def profile_dataset_dir(self, profile_id: str, region: str, config_id: str) -> Path:
        self._profile_record(profile_id)
        return self.root / profile_id / "training_datasets" / region / config_id

    def profile_training_patch_cache_dir(self, profile_id: str, region: str, config_id: str) -> Path:
        self._profile_record(profile_id)
        return self.root / profile_id / "training_patch_cache" / region / config_id

    def profile_logical_patch_dir(self, profile_id: str, region: str) -> Path:
        self._profile_record(profile_id)
        return self.root / profile_id / "logical_patches" / region

    def profile_logical_patch_manifest(self, profile_id: str, region: str) -> Path:
        return self.profile_logical_patch_dir(profile_id, region) / "manifest.csv"

    def ensure_profile_logical_patch_manifest(self, profile_id: str, region: str) -> Path:
        """Create a user-owned logical manifest from legacy data when needed.

        The default Profile receives the complete legacy catalog so its historical
        excluded rows remain reviewable. Other existing Profiles receive only
        their selected rows; a newly-created empty Profile remains empty until it
        generates patches in its own workspace.
        """
        path = self.profile_logical_patch_manifest(profile_id, region)
        if path.exists():
            self._sync_members_from_profile_manifest(profile_id, region, path)
            return path
        if region not in self.regions:
            return path
        source = self.regions[region].logical_patch_manifest
        if not source.exists():
            return path
        rows = read_csv_records(source)
        members = self.members(profile_id, region)
        if profile_id == DEFAULT_PROFILE_ID:
            selected = {patch_id for _region, patch_id in members}
            rows = [
                {**row, "include": "true" if row.get("logical_patch_id") in selected else "false"}
                for row in rows
            ]
        else:
            selected = {patch_id for _region, patch_id in members}
            rows = [
                {**row, "include": "true"}
                for row in rows
                if row.get("logical_patch_id") in selected
            ]
        if rows:
            self._write_profile_logical_rows(profile_id, region, rows)
            self._sync_members_from_profile_manifest(profile_id, region, path)
        return path

    def remove_profile_logical_patches(self, sample_id: str) -> int:
        removed = 0
        with self._lock:
            for profile in self._read_registry().get("items", []):
                profile_id = profile.get("id", "")
                for region in self.regions:
                    path = self.profile_logical_patch_manifest(profile_id, region)
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
                        self._write_profile_logical_rows(profile_id, region, kept)
                        members = self.members(profile_id)
                        members.difference_update((region, patch_id) for patch_id in removed_ids)
                        self._write_members(profile_id, members)
        return removed

    def profile_model_dir(self, profile_id: str, scope: str) -> Path:
        self._profile_record(profile_id)
        return self.model_root / "profiles" / profile_id / scope

    def source_signature(self, profile_id: str, site_id: str) -> str:
        variants = self.selected_variants(profile_id, site_id)
        payload = [(item["id"], item["content_hash"]) for item in variants]
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def assert_trainable(self, profile_id: str) -> None:
        profile = self.get(profile_id, allow_archived=False)
        if profile["status"] != "active":
            raise ProfileError(f"构建或训练前必须解决用户的数据源冲突: {profile_id}")

    def _variants_for_patch(self, region_key: str, patch: dict) -> list[str]:
        sample_id = patch.get("sample_id", "")
        cache_key = (region_key, sample_id)
        if cache_key in self._variant_ids_cache:
            return self._variant_ids_cache[cache_key]
        samples = self._sample_rows_cache.get(region_key)
        if samples is None:
            samples = {
                row.get("sample_id", ""): row
                for row in read_csv_records(self.regions[region_key].training_samples)
                if row.get("sample_id")
            }
            self._sample_rows_cache[region_key] = samples
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

    def _merge_patch_sources(self, profile_id: str, region: str, patches: list[dict]) -> None:
        sources = self._read_sources(profile_id)
        for patch in patches:
            site_id = clean_optional(patch.get("site_id"))
            if not site_id:
                continue
            incoming = set(self._variants_for_patch(region, patch))
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
                    "source_profile_ids": [profile_id],
                }
        self._write_sources(profile_id, sources)

    def _cleanup_empty_sites(self, profile_id: str, members: set[tuple[str, str]]) -> None:
        active_sites = self._member_sites(profile_id, members)
        sources = self._read_sources(profile_id)
        sources["sites"] = {key: value for key, value in sources["sites"].items() if key in active_sites}
        sources["conflicts"] = {key: value for key, value in sources["conflicts"].items() if key in active_sites}
        self._write_sources(profile_id, sources)

    def _member_sites(self, profile_id: str, members: set[tuple[str, str]]) -> set[str]:
        by_region: dict[str, set[str]] = {}
        for region, patch_id in members:
            by_region.setdefault(region, set()).add(patch_id)
        result = set()
        for region, wanted in by_region.items():
            for row in self._profile_logical_index(profile_id, region).values():
                if row.get("logical_patch_id") in wanted and row.get("site_id"):
                    result.add(row["site_id"])
        return result

    def _logical_index(self, region: str) -> dict[str, dict]:
        path = self.regions[region].logical_patch_manifest
        return self._logical_index_for_path(("legacy", region), path)

    def _profile_logical_index(self, profile_id: str, region: str) -> dict[str, dict]:
        path = self.profile_logical_patch_manifest(profile_id, region)
        return self._logical_index_for_path((profile_id, region), path)

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

    def _refresh_status(self, profile_id: str) -> None:
        with self._lock:
            registry = self._read_registry()
            profile = self._profile_record(profile_id, registry)
            if profile.get("status") != "archived":
                profile["status"] = "needs_resolution" if self._read_sources(profile_id)["conflicts"] else "active"
            profile["updated_at"] = _timestamp()
            self._write_registry(registry)

    def _summary(self, profile: dict) -> dict:
        profile_id = profile["id"]
        members = self.members(profile_id) if self._members_path(profile_id).exists() else set()
        sources = self._read_sources(profile_id)
        return {
            **profile,
            "selected_patch_count": len(members),
            "site_count": len(self._member_sites(profile_id, members)) if members else 0,
            "conflict_count": len(sources["conflicts"]),
            "default": profile_id == DEFAULT_PROFILE_ID,
        }

    def _profile_record(
        self,
        profile_id: str,
        registry: dict | None = None,
        *,
        allow_archived: bool = True,
    ) -> dict:
        item = next((value for value in (registry or self._read_registry())["items"] if value.get("id") == profile_id), None)
        if item is None:
            raise KeyError(f"Profile not found: {profile_id}")
        if not allow_archived and item.get("status") == "archived":
            raise ProfileError(f"用户已归档: {profile_id}")
        return item

    def _validate_unique_name(self, registry: dict, name: str, exclude_id: str = "") -> None:
        if any(item.get("id") != exclude_id and str(item.get("name", "")).casefold() == name.casefold() for item in registry["items"]):
            raise ProfileError(f"用户名称已存在: {name}")

    def _read_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"version": PROFILE_FORMAT_VERSION, "default_profile_id": DEFAULT_PROFILE_ID, "items": []}
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        payload.setdefault("items", [])
        return payload

    def _write_registry(self, payload: dict) -> None:
        _atomic_text(self.registry_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _members_path(self, profile_id: str) -> Path:
        return self.root / profile_id / "patch_members.csv"

    def _write_members(self, profile_id: str, members: set[tuple[str, str]]) -> None:
        lines = []
        with tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["region", "logical_patch_id"])
            writer.writeheader()
            for region, patch_id in sorted(members):
                writer.writerow({"region": region, "logical_patch_id": patch_id})
            handle.seek(0)
            lines.append(handle.read())
        _atomic_text(self._members_path(profile_id), "".join(lines))

    def _write_profile_logical_rows(self, profile_id: str, region: str, rows: list[dict]) -> None:
        logical_dir = self.profile_logical_patch_dir(profile_id, region)
        owned_rows = [
            {
                **row,
                "preview_path": str(logical_dir / "preview" / f"{row.get('logical_patch_id', '')}.png"),
            }
            for row in rows
        ]
        write_csv_records(self.profile_logical_patch_manifest(profile_id, region), owned_rows)
        self._logical_rows_cache.pop((profile_id, region), None)

    def _sync_members_from_profile_manifest(self, profile_id: str, region: str, path: Path) -> None:
        members = self.members(profile_id)
        synced = {(key, patch_id) for key, patch_id in members if key != region}
        synced.update(
            (region, row.get("logical_patch_id", ""))
            for row in read_csv_records(path)
            if row.get("logical_patch_id") and truthy_flag(row.get("include"), default=True)
        )
        if synced != members:
            self._write_members(profile_id, synced)

    def _write_union_logical_patch_manifests(self, profile_id: str, source_ids: list[str]) -> None:
        for region in self.regions:
            selected_rows: dict[str, dict] = {}
            for source_id in source_ids:
                path = self.ensure_profile_logical_patch_manifest(source_id, region)
                members = {patch_id for _region, patch_id in self.members(source_id, region)}
                for row in read_csv_records(path):
                    patch_id = row.get("logical_patch_id", "")
                    if patch_id in members:
                        selected_rows.setdefault(patch_id, {**row, "include": "true"})
            if selected_rows:
                self._write_profile_logical_rows(
                    profile_id,
                    region,
                    [selected_rows[key] for key in sorted(selected_rows)],
                )

    def _sources_path(self, profile_id: str) -> Path:
        return self.root / profile_id / "site_sources.json"

    def _read_sources(self, profile_id: str) -> dict:
        path = self._sources_path(profile_id)
        if not path.exists():
            return {"sites": {}, "conflicts": {}}
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("sites", {})
        payload.setdefault("conflicts", {})
        return payload

    def _write_sources(self, profile_id: str, payload: dict) -> None:
        _atomic_text(self._sources_path(profile_id), json.dumps(payload, ensure_ascii=False, indent=2))
