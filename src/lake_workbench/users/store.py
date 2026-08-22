"""Persist users separately from workspace-owned training resources."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from lake_workbench.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from lake_workbench.auth import AuthIdentity


DEFAULT_USER_ID = "default"
USER_FORMAT_VERSION = 1


class UserError(ValueError):
    """Raised when a user operation violates a domain invariant."""


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class UserStore:
    """Own user identity records and the one-way default workspace mapping."""

    def __init__(self, workspace_store, root: Path | None = None) -> None:
        self.workspace_store = workspace_store
        self.root = root or PROJECT_ROOT / "data" / "users"
        self.registry_path = self.root / "users.json"
        self._lock = threading.RLock()

    def ensure_default_user(self) -> dict:
        with self._lock:
            registry = self._read_registry()
            if registry["items"]:
                return self.get(DEFAULT_USER_ID)
            workspace = self.workspace_store.ensure_default_workspace()
            now = _timestamp()
            user = {
                "id": DEFAULT_USER_ID,
                "name": "默认用户",
                "status": "active",
                "default_workspace_id": workspace["id"],
                "created_at": now,
                "updated_at": now,
            }
            registry["items"].append(user)
            self._write_registry(registry)
            return self._summary(user)

    def list(self, include_archived: bool = False) -> dict:
        with self._lock:
            items = [
                self._summary(item)
                for item in self._read_registry()["items"]
                if include_archived or item.get("status") != "archived"
            ]
            items.sort(key=lambda item: (item["id"] != DEFAULT_USER_ID, item["name"].casefold()))
            return {"default": DEFAULT_USER_ID, "items": items}

    def get(self, user_id: str, *, allow_archived: bool = True) -> dict:
        with self._lock:
            user = self._record(user_id)
            if not allow_archived and user.get("status") == "archived":
                raise UserError(f"用户已归档: {user_id}")
            return self._summary(user)

    def create(
        self,
        name: str,
        workspace_mode: str = "empty",
        source_workspace_ids: list[str] | None = None,
    ) -> dict:
        name = str(name or "").strip()
        if not name:
            raise UserError("用户名称不能为空")
        with self._lock:
            registry = self._read_registry()
            self._validate_unique_name(registry, name)
            user_id = uuid.uuid4().hex[:12]
            workspace = self.workspace_store.create(
                f"{name}的训练工作区",
                workspace_mode,
                source_workspace_ids or [],
            )
            now = _timestamp()
            user = {
                "id": user_id,
                "name": name,
                "status": "active",
                "default_workspace_id": workspace["id"],
                "created_at": now,
                "updated_at": now,
            }
            registry["items"].append(user)
            self._write_registry(registry)
            return self._summary(user)

    def resolve_identity(
        self,
        identity: "AuthIdentity",
        *,
        bootstrap_email: str = "",
        admin_emails: set[str] | None = None,
    ) -> dict:
        """Find or provision the User mapped to one trusted external identity."""
        normalized_email = identity.email.strip().lower()
        admins = {value.strip().lower() for value in (admin_emails or set()) if value.strip()}
        with self._lock:
            registry = self._read_registry()
            existing = next(
                (
                    item
                    for item in registry["items"]
                    if item.get("auth_provider") == identity.provider
                    and item.get("auth_subject") == identity.subject
                ),
                None,
            )
            if existing is not None:
                changed = False
                if normalized_email and existing.get("email") != normalized_email:
                    existing["email"] = normalized_email
                    changed = True
                expected_role = "admin" if normalized_email in admins else "user"
                if existing.get("role") != expected_role:
                    existing["role"] = expected_role
                    changed = True
                if changed:
                    existing["updated_at"] = _timestamp()
                    self._write_registry(registry)
                return self._summary(existing)

            default = next((item for item in registry["items"] if item.get("id") == DEFAULT_USER_ID), None)
            bootstrap_matches = bool(
                bootstrap_email
                and normalized_email == bootstrap_email.strip().lower()
            )
            may_bind_default = (
                default is not None
                and (
                    not default.get("auth_subject")
                    or (
                        bootstrap_matches
                        and default.get("auth_provider") == "development"
                    )
                )
                and (
                    identity.provider == "development"
                    or bootstrap_matches
                )
            )
            if may_bind_default:
                default.update(
                    {
                        "name": identity.name or normalized_email or default["name"],
                        "email": normalized_email,
                        "auth_provider": identity.provider,
                        "auth_subject": identity.subject,
                        "role": "admin" if normalized_email in admins else "user",
                        "updated_at": _timestamp(),
                    }
                )
                self._write_registry(registry)
                return self._summary(default)

            base_name = identity.name or normalized_email or "用户"
            name = self._available_name(registry, base_name)
            workspace = self.workspace_store.create(f"{name}的训练工作区")
            now = _timestamp()
            user = {
                "id": uuid.uuid4().hex[:12],
                "name": name,
                "email": normalized_email,
                "status": "active",
                "role": "admin" if normalized_email in admins else "user",
                "auth_provider": identity.provider,
                "auth_subject": identity.subject,
                "default_workspace_id": workspace["id"],
                "created_at": now,
                "updated_at": now,
            }
            registry["items"].append(user)
            self._write_registry(registry)
            return self._summary(user)

    def owns_workspace(self, user_id: str, workspace_id: str) -> bool:
        return self._record(user_id).get("default_workspace_id") == workspace_id

    def rename(self, user_id: str, name: str) -> dict:
        name = str(name or "").strip()
        if not name:
            raise UserError("用户名称不能为空")
        with self._lock:
            registry = self._read_registry()
            user = self._record(user_id, registry)
            self._validate_unique_name(registry, name, exclude_id=user_id)
            user["name"] = name
            user["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(user)

    def archive(self, user_id: str) -> dict:
        if user_id == DEFAULT_USER_ID:
            raise UserError("默认用户不能归档")
        with self._lock:
            registry = self._read_registry()
            user = self._record(user_id, registry)
            user["status"] = "archived"
            user["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(user)

    def restore(self, user_id: str) -> dict:
        with self._lock:
            registry = self._read_registry()
            user = self._record(user_id, registry)
            user["status"] = "active"
            user["updated_at"] = _timestamp()
            self._write_registry(registry)
            return self._summary(user)

    def _summary(self, user: dict) -> dict:
        workspace = self.workspace_store.get(user["default_workspace_id"])
        return {**user, "workspace": workspace, "default": user["id"] == DEFAULT_USER_ID}

    def _record(self, user_id: str, registry: dict | None = None) -> dict:
        item = next(
            (value for value in (registry or self._read_registry())["items"] if value.get("id") == user_id),
            None,
        )
        if item is None:
            raise KeyError(f"User not found: {user_id}")
        return item

    @staticmethod
    def _validate_unique_name(registry: dict, name: str, exclude_id: str = "") -> None:
        if any(
            item.get("id") != exclude_id
            and str(item.get("name", "")).casefold() == name.casefold()
            for item in registry["items"]
        ):
            raise UserError(f"用户名称已存在: {name}")

    @staticmethod
    def _available_name(registry: dict, base_name: str) -> str:
        existing = {str(item.get("name", "")).casefold() for item in registry["items"]}
        if base_name.casefold() not in existing:
            return base_name
        index = 2
        while f"{base_name} {index}".casefold() in existing:
            index += 1
        return f"{base_name} {index}"

    def _read_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"version": USER_FORMAT_VERSION, "default_user_id": DEFAULT_USER_ID, "items": []}
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        payload.setdefault("items", [])
        return payload

    def _write_registry(self, payload: dict) -> None:
        _atomic_json(self.registry_path, payload)
