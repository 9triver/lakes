"""Profile-aware model discovery and safe key resolution."""

from __future__ import annotations

from pathlib import Path

from lake_workbench.models.metadata import model_sort_key, model_training_metadata
from lake_workbench.models.runtime import load_model_checkpoint
from lake_workbench.utils import clean_optional, display_path


class ProfileModelRegistry:
    def __init__(self, profile_store, regions: dict, profile_id: str) -> None:
        self.profile_store = profile_store
        self.regions = regions
        self.profile_id = profile_id
        self.model_root = profile_store.model_root

    def list(self, scope: str, visibility: str = "current") -> dict:
        if visibility not in {"current", "all"}:
            raise ValueError("model visibility must be current or all")
        profiles = (
            self.profile_store.list(include_archived=True)["items"]
            if visibility == "all"
            else [self.profile_store.get(self.profile_id)]
        )
        scopes = list(self.regions) + ["all"] if scope == "all" else [scope, "all"]
        items = []
        for profile in profiles:
            for model_scope in scopes:
                for path, legacy in self._paths(profile["id"], model_scope):
                    key = self.key(profile["id"], model_scope, path)
                    prefix = f"{profile['name']} / " if visibility == "all" else ""
                    label = f"{prefix}{'全部区域' if model_scope == 'all' else model_scope} / {path.parent.name}/{path.name}"
                    try:
                        model = load_model_checkpoint(path)
                        item = {
                            "key": key,
                            "label": label,
                            "name": path.parent.name,
                            "weight": path.name,
                            "path": display_path(path),
                            "profile_id": profile["id"],
                            "profile_name": profile["name"],
                            "profile_status": profile["status"],
                            "scope": model_scope,
                            "region": model_scope,
                            "legacy": legacy,
                            "epoch": model.epoch,
                            "in_channels": model.in_channels,
                            "base_channels": model.base_channels,
                            "model_type": model.model_type,
                            "model_options": model.model_options,
                            "architecture_label": model.architecture_label,
                            **model_training_metadata(path, model_scope),
                        }
                    except Exception as exc:  # noqa: BLE001 - broken weights remain discoverable.
                        item = {
                            "key": key,
                            "label": label,
                            "name": path.parent.name,
                            "weight": path.name,
                            "path": display_path(path),
                            "profile_id": profile["id"],
                            "profile_name": profile["name"],
                            "profile_status": profile["status"],
                            "scope": model_scope,
                            "region": model_scope,
                            "legacy": legacy,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    items.append(item)
        unique = {item["key"]: item for item in items}
        items = sorted(unique.values(), key=model_sort_key)
        default = next((item["key"] for item in items if not item.get("error")), items[0]["key"] if items else "")
        return {
            "profile_id": self.profile_id,
            "scope": scope,
            "visibility": visibility,
            "default": default,
            "items": items,
        }

    def resolve(self, model_key: str, requested_scope: str) -> Path:
        key = clean_optional(model_key) or ""
        if not key:
            payload = self.list(requested_scope)
            key = payload["default"]
            if not key:
                raise FileNotFoundError("No model weights found")
        path = Path(key)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"invalid model key: {key}")
        parts = path.parts
        if len(parts) == 5 and parts[0] == "profiles":
            profile_id, scope, run_name, weight = parts[1:]
            self.profile_store.get(profile_id)
            preferred = self.model_root / "profiles" / profile_id / scope / run_name / weight
            if preferred.exists():
                return preferred
            if profile_id == "default":
                legacy = self.model_root / scope / run_name / weight
                if legacy.exists():
                    return legacy
            return preferred
        if len(parts) == 3 and parts[0] in {*self.regions, "all"}:
            return self.model_root / parts[0] / parts[1] / parts[2]
        if len(parts) == 2:
            scope = requested_scope if requested_scope != "all" else "all"
            return self.model_root / scope / parts[0] / parts[1]
        raise ValueError(f"invalid model key: {key}")

    @staticmethod
    def key(profile_id: str, scope: str, path: Path) -> str:
        return f"profiles/{profile_id}/{scope}/{path.parent.name}/{path.name}"

    def _paths(self, profile_id: str, scope: str) -> list[tuple[Path, bool]]:
        result = []
        current = self.model_root / "profiles" / profile_id / scope
        if current.exists():
            result.extend((path, False) for path in current.glob("*/*.pt"))
        if profile_id == "default":
            legacy = self.model_root / scope
            if legacy.exists():
                result.extend((path, True) for path in legacy.glob("*/*.pt"))
        return sorted(set(result), key=lambda item: str(item[0]))
