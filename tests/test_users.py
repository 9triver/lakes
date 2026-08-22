from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lake_workbench.auth import AuthIdentity
from lake_workbench.users import UserError, UserStore


class FakeWorkspaceStore:
    def __init__(self) -> None:
        self.items = {
            "default": {
                "id": "default",
                "name": "默认训练工作区",
                "status": "active",
                "selected_patch_count": 3,
                "site_count": 2,
                "conflict_count": 0,
                "default": True,
                "source_workspace_ids": [],
            }
        }
        self.created = []

    def ensure_default_workspace(self):
        return self.items["default"]

    def get(self, workspace_id):
        if workspace_id not in self.items:
            raise KeyError(workspace_id)
        return self.items[workspace_id]

    def create(self, name, mode="empty", source_workspace_ids=None):
        workspace_id = f"workspace-{len(self.items)}"
        workspace = {
            "id": workspace_id,
            "name": name,
            "status": "active",
            "selected_patch_count": 0,
            "site_count": 0,
            "conflict_count": 0,
            "default": False,
            "source_workspace_ids": list(source_workspace_ids or []),
        }
        self.items[workspace_id] = workspace
        self.created.append((name, mode, list(source_workspace_ids or [])))
        return workspace


class UserStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspaces = FakeWorkspaceStore()
        self.store = UserStore(self.workspaces, root=Path(self.temporary.name) / "users")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_default_user_maps_to_default_workspace(self) -> None:
        user = self.store.ensure_default_user()

        self.assertEqual(user["id"], "default")
        self.assertEqual(user["default_workspace_id"], "default")
        self.assertEqual(user["workspace"]["id"], "default")
        self.assertNotIn("user_id", user["workspace"])

    def test_create_user_creates_exactly_one_workspace_mapping(self) -> None:
        self.store.ensure_default_user()
        user = self.store.create("Alice", "union", ["default"])

        self.assertEqual(len(self.workspaces.created), 1)
        self.assertEqual(user["workspace"]["source_workspace_ids"], ["default"])
        self.assertEqual(user["default_workspace_id"], user["workspace"]["id"])
        self.assertNotEqual(user["id"], user["default_workspace_id"])

    def test_user_lifecycle_does_not_archive_workspace(self) -> None:
        self.store.ensure_default_user()
        user = self.store.create("Alice")

        archived = self.store.archive(user["id"])
        self.assertEqual(archived["status"], "archived")
        self.assertEqual(self.workspaces.get(user["default_workspace_id"])["status"], "active")
        self.assertEqual(self.store.restore(user["id"])["status"], "active")

    def test_default_user_cannot_be_archived(self) -> None:
        self.store.ensure_default_user()
        with self.assertRaises(UserError):
            self.store.archive("default")

    def test_development_identity_binds_default_user(self) -> None:
        self.store.ensure_default_user()
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")

        user = self.store.resolve_identity(identity, admin_emails={"dev@example.com"})

        self.assertEqual(user["id"], "default")
        self.assertEqual(user["auth_subject"], "local")
        self.assertEqual(user["role"], "admin")
        self.assertEqual(user["default_workspace_id"], "default")

    def test_bootstrap_email_binds_default_user_in_cloudflare_mode(self) -> None:
        self.store.ensure_default_user()
        identity = AuthIdentity("cloudflare-access", "owner-sub", "owner@example.com", "Owner")

        user = self.store.resolve_identity(
            identity,
            bootstrap_email="owner@example.com",
            admin_emails={"owner@example.com"},
        )

        self.assertEqual(user["id"], "default")
        self.assertEqual(user["role"], "admin")
        self.assertEqual(self.workspaces.created, [])

    def test_bootstrap_email_replaces_development_binding(self) -> None:
        self.store.ensure_default_user()
        self.store.resolve_identity(
            AuthIdentity("development", "local", "dev@example.com", "Developer")
        )

        user = self.store.resolve_identity(
            AuthIdentity("cloudflare-access", "owner-sub", "owner@example.com", "Owner"),
            bootstrap_email="owner@example.com",
        )

        self.assertEqual(user["id"], "default")
        self.assertEqual(user["auth_provider"], "cloudflare-access")
        self.assertEqual(user["auth_subject"], "owner-sub")
        self.assertEqual(self.workspaces.created, [])

    def test_new_cloudflare_identity_creates_one_user_and_workspace(self) -> None:
        self.store.ensure_default_user()
        identity = AuthIdentity("cloudflare-access", "person-sub", "person@example.com", "Person")

        first = self.store.resolve_identity(identity)
        second = self.store.resolve_identity(identity)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["default_workspace_id"], second["default_workspace_id"])
        self.assertEqual(len(self.workspaces.created), 1)
        self.assertNotIn("user_id", first["workspace"])

    def test_role_follows_current_admin_email_configuration(self) -> None:
        self.store.ensure_default_user()
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")

        self.assertEqual(
            self.store.resolve_identity(identity, admin_emails={"dev@example.com"})["role"],
            "admin",
        )
        self.assertEqual(self.store.resolve_identity(identity, admin_emails=set())["role"], "user")


if __name__ == "__main__":
    unittest.main()
