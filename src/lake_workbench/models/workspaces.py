"""Workspace-aware model discovery and safe key resolution."""

from __future__ import annotations

from pathlib import Path

from lake_workbench.models.metadata import model_sort_key, model_training_metadata
from lake_workbench.models.experiments import checkpoint_description, experimental_models
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.models.routing import routed_models,profile_for as routing_profile_for
from lake_workbench.utils import clean_optional, display_path


class WorkspaceModelRegistry:
    def __init__(self, workspace_store, regions: dict, workspace_id: str, *, allow_foreign: bool = False) -> None:
        self.workspace_store = workspace_store
        self.regions = regions
        self.workspace_id = workspace_id
        self.model_root = workspace_store.model_root
        self.allow_foreign = allow_foreign

    def list(self, scope: str, visibility: str = "current") -> dict:
        if visibility not in {"current", "all"}:
            raise ValueError("model visibility must be current or all")
        if visibility == "all" and not self.allow_foreign:
            raise ValueError("无权查看其他训练工作区的模型")
        workspaces = (
            self.workspace_store.list(include_archived=True)["items"]
            if visibility == "all"
            else [self.workspace_store.get(self.workspace_id)]
        )
        scopes = list(self.regions) + ["all"] if scope == "all" else [scope, "all"]
        items = []
        for workspace in workspaces:
            for model_scope in scopes:
                for path in self._paths(workspace["id"], model_scope):
                    key = self.key(workspace["id"], model_scope, path)
                    prefix = f"{workspace['name']} / " if visibility == "all" else ""
                    label = f"{prefix}{'全部区域' if model_scope == 'all' else model_scope} / {path.parent.name}/{path.name}"
                    try:
                        model = checkpoint_description(path)
                        item = {
                            "key": key,
                            "label": label,
                            "name": path.parent.name,
                            "weight": path.name,
                            "path": display_path(path),
                            "workspace_id": workspace["id"],
                            "workspace_name": workspace["name"],
                            "workspace_status": workspace["status"],
                            "scope": model_scope,
                            "region": model_scope,
                            **{key:model[key] for key in ['epoch','in_channels','base_channels','model_type','model_options','architecture_label']},
                            **model_training_metadata(path, model_scope),
                            "completed": (model_training_metadata(path, model_scope).get('history_count',0) >= int(model['config'].get('epochs',1))),
                        }
                    except Exception as exc:  # noqa: BLE001 - broken weights remain discoverable.
                        item = {
                            "key": key,
                            "label": label,
                            "name": path.parent.name,
                            "weight": path.name,
                            "path": display_path(path),
                            "workspace_id": workspace["id"],
                            "workspace_name": workspace["name"],
                            "workspace_status": workspace["status"],
                            "scope": model_scope,
                            "region": model_scope,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    items.append(item)
            if self.model_root.resolve() == (PROJECT_ROOT / 'data/models').resolve():
                items.extend(experimental_models(workspace['id']))
                items.extend(routed_models(workspace['id']))
        unique = {item["key"]: item for item in items}
        items = sorted(unique.values(), key=model_sort_key)
        default = next((item["key"] for item in items if not item.get("error")), items[0]["key"] if items else "")
        return {
            "workspace_id": self.workspace_id,
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
        if len(parts)==3 and parts[0]=='routing':
            workspace_id=parts[1]
            if workspace_id!=self.workspace_id and not self.allow_foreign:raise ValueError('无权使用其他训练工作区的模型')
            self.workspace_store.get(workspace_id)
            if self.model_root.resolve()!=(PROJECT_ROOT/'data/models').resolve():raise ValueError('invalid routing registry')
            selected=next((item for item in routed_models(workspace_id) if item['key']==key),None)
            if selected is None:raise FileNotFoundError('自动路由模型不存在')
            resolved=PROJECT_ROOT/selected['path']
            if routing_profile_for(resolved) is None:raise ValueError('未知自动路由配置')
            return resolved
        if len(parts) == 4 and parts[0] == 'experiments':
            workspace_id = parts[1]
            if workspace_id != self.workspace_id and not self.allow_foreign:
                raise ValueError('无权使用其他训练工作区的模型')
            self.workspace_store.get(workspace_id)
            if self.model_root.resolve() != (PROJECT_ROOT / 'data/models').resolve():
                raise ValueError('invalid experimental model registry')
            selected = next((item for item in experimental_models(workspace_id) if item['key'] == key), None)
            if selected is None:
                raise FileNotFoundError('实验模型不存在')
            return PROJECT_ROOT / selected['path']
        if len(parts) == 5 and parts[0] == "workspaces":
            workspace_id, scope, run_name, weight = parts[1:]
            if workspace_id != self.workspace_id and not self.allow_foreign:
                raise ValueError("无权使用其他训练工作区的模型")
            self.workspace_store.get(workspace_id)
            return self.model_root / "workspaces" / workspace_id / scope / run_name / weight
        raise ValueError(f"invalid model key: {key}")

    @staticmethod
    def key(workspace_id: str, scope: str, path: Path) -> str:
        return f"workspaces/{workspace_id}/{scope}/{path.parent.name}/{path.name}"

    def _paths(self, workspace_id: str, scope: str) -> list[Path]:
        current = self.model_root / "workspaces" / workspace_id / scope
        # Validation exposes the checkpoint selected by training, not the
        # resumable ``last.pt`` checkpoint from the same experiment.
        return sorted(current.glob("*/best.pt")) if current.exists() else []
