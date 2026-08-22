from __future__ import annotations

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

from lake_workbench.auth import AuthError, AuthIdentity
from lake_workbench.http_handler import create_site_handler
from lake_workbench.routes.models import handle_model_get
from lake_workbench.routes.sites import handle_site_get
from lake_workbench.routes.training import handle_training_get, handle_training_post


class HandlerFactoryTests(unittest.TestCase):
    def test_each_factory_call_binds_isolated_runtime_dependencies(self) -> None:
        first_catalogs = {"first": object()}
        second_catalogs = {"second": object()}
        first = create_site_handler(
            catalogs=first_catalogs,
            downloads_by_region={"first": "download-1"},
            patch_exports_by_region={"first": "patch-1"},
            all_patch_exports="all-patch-1",
            default_region_key="first",
            region_service="regions-1",
        )
        second = create_site_handler(
            catalogs=second_catalogs,
            downloads_by_region={"second": "download-2"},
            patch_exports_by_region={"second": "patch-2"},
            all_patch_exports="all-patch-2",
            default_region_key="second",
            region_service="regions-2",
        )

        self.assertIsNot(first, second)
        self.assertIs(first.catalog, first_catalogs["first"])
        self.assertIs(second.catalog, second_catalogs["second"])
        self.assertEqual(first.downloads, "download-1")
        self.assertEqual(second.downloads, "download-2")
        self.assertEqual(first.region_service, "regions-1")
        self.assertEqual(second.region_service, "regions-2")

    def test_request_context_binds_shared_region_without_workspace(self) -> None:
        catalog = SimpleNamespace(region=SimpleNamespace(key="first"))
        handler_type = create_site_handler(
            catalogs={"first": catalog},
            downloads_by_region={"first": "download-1"},
            patch_exports_by_region={"first": "patch-1"},
            all_patch_exports="patch-all",
            default_region_key="first",
            region_service="regions",
        )
        handler = handler_type.__new__(handler_type)

        self.assertEqual(handler._bind_request_context("/api/sites"), "/api/sites")
        self.assertEqual(handler._bind_request_context("/api/lakes"), "/api/lakes")
        self.assertIs(handler.catalog, catalog)
        self.assertEqual(handler.downloads, "download-1")
        self.assertEqual(handler.patch_exports, "patch-1")
        self.assertIsNone(handler.workspace_id)
        self.assertIsNone(handler.training_runs)

    def test_request_context_normalizes_region_and_all_paths(self) -> None:
        first = SimpleNamespace(region=SimpleNamespace(key="first"))
        second = SimpleNamespace(region=SimpleNamespace(key="second"))
        handler_type = create_site_handler(
            catalogs={"first": first, "second": second},
            downloads_by_region={"first": "download-1", "second": "download-2"},
            patch_exports_by_region={"first": "patch-1", "second": "patch-2"},
            all_patch_exports="patch-all",
            default_region_key="first",
            region_service="regions",
        )
        handler = handler_type.__new__(handler_type)

        self.assertEqual(handler._bind_request_context("/api/regions/second/sites"), "/api/sites")
        self.assertIs(handler.catalog, second)
        self.assertEqual(handler.downloads, "download-2")
        self.assertIsNone(handler.workspace_id)
        self.assertIsNone(handler.training_runs)
        self.assertEqual(handler._bind_request_context("/api/regions/all/training-runs"), "/api/all/training-runs")
        self.assertIs(handler.catalog, first)
        self.assertEqual(handler.patch_exports, "patch-all")
        self.assertIsNone(handler.training_runs)

    def test_request_context_normalizes_workspace_region_paths(self) -> None:
        first = SimpleNamespace(region=SimpleNamespace(key="first"))
        second = SimpleNamespace(region=SimpleNamespace(key="second"))
        workspaces = SimpleNamespace(get=lambda workspace_id, allow_archived=False: {"id": workspace_id})
        handler_type = create_site_handler(
            catalogs={"first": first, "second": second},
            downloads_by_region={"first": "download-1", "second": "download-2"},
            patch_exports_by_region={"first": "patch-1", "second": "patch-2"},
            all_patch_exports="patch-all",
            default_region_key="first",
            region_service="regions",
            workspace_store=workspaces,
            training_manager_factory=lambda workspace_id, scope: f"{workspace_id}:{scope}",
        )
        handler = handler_type.__new__(handler_type)

        path = handler._bind_request_context("/api/workspaces/alpha/regions/second/logical-patches")

        self.assertEqual(path, "/api/logical-patches")
        self.assertEqual(handler.workspace_id, "alpha")
        self.assertIs(handler.catalog, second)
        self.assertEqual(handler.training_runs, "alpha:second")


class HandlerAuthenticationTests(unittest.TestCase):
    @staticmethod
    def handler(authenticate, user: dict):
        catalog = SimpleNamespace(region=SimpleNamespace(key="first"))
        auth_service = SimpleNamespace(authenticate=authenticate)
        user_store = SimpleNamespace(resolve_identity=lambda *_args, **_kwargs: user)
        handler_type = create_site_handler(
            catalogs={"first": catalog},
            downloads_by_region={"first": "download"},
            patch_exports_by_region={"first": "patch"},
            all_patch_exports="all-patch",
            default_region_key="first",
            region_service="regions",
            auth_service=auth_service,
            user_store=user_store,
        )
        handler = handler_type.__new__(handler_type)
        handler.headers = {}
        handler.error = None
        handler._error = lambda status, message: setattr(handler, "error", (status, message))
        return handler

    def test_invalid_authentication_returns_unauthorized(self) -> None:
        def reject(_headers):
            raise AuthError("login required")

        handler = self.handler(reject, {})

        self.assertFalse(handler._authenticate_api_request("/api/regions"))
        self.assertEqual(handler.error, (HTTPStatus.UNAUTHORIZED, "login required"))

    def test_regular_user_can_access_own_workspace(self) -> None:
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")
        user = {"id": "user-1", "status": "active", "role": "user", "default_workspace_id": "own"}
        handler = self.handler(lambda _headers: identity, user)

        self.assertTrue(handler._authenticate_api_request("/api/workspaces/own/regions/first/sites"))
        self.assertEqual(handler.current_user, user)
        self.assertEqual(handler.auth_identity, identity)

    def test_regular_user_cannot_access_foreign_workspace(self) -> None:
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")
        user = {"id": "user-1", "status": "active", "role": "user", "default_workspace_id": "own"}
        handler = self.handler(lambda _headers: identity, user)

        self.assertFalse(handler._authenticate_api_request("/api/workspaces/foreign/regions/first/sites"))
        self.assertEqual(handler.error[0], HTTPStatus.FORBIDDEN)

    def test_admin_can_access_foreign_workspace(self) -> None:
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")
        user = {"id": "admin", "status": "active", "role": "admin", "default_workspace_id": "own"}
        handler = self.handler(lambda _headers: identity, user)

        self.assertTrue(handler._authenticate_api_request("/api/workspaces/foreign/regions/first/sites"))

    def test_archived_user_is_rejected(self) -> None:
        identity = AuthIdentity("development", "local", "dev@example.com", "Developer")
        user = {"id": "user-1", "status": "archived", "role": "user", "default_workspace_id": "own"}
        handler = self.handler(lambda _headers: identity, user)

        self.assertFalse(handler._authenticate_api_request("/api/regions"))
        self.assertEqual(handler.error, (HTTPStatus.FORBIDDEN, "用户已归档"))

    def test_local_network_uses_configured_user_without_identity_provisioning(self) -> None:
        identity = AuthIdentity("cloudflare-access", "remote", "remote@example.com", "Remote")
        user = {"id": "default", "email": "owner@example.com", "name": "Owner", "status": "active", "role": "admin", "default_workspace_id": "default"}
        resolve_identity = Mock()
        auth_service = SimpleNamespace(
            local_user_id_for=lambda host, _headers: "default" if host == "192.168.30.107" else None,
            authenticate=lambda _headers: identity,
        )
        user_store = SimpleNamespace(get=lambda user_id, allow_archived=False: user, resolve_identity=resolve_identity)
        handler_type = create_site_handler(
            catalogs={"first": SimpleNamespace(region=SimpleNamespace(key="first"))},
            downloads_by_region={"first": "download"},
            patch_exports_by_region={"first": "patch"},
            all_patch_exports="all-patch",
            default_region_key="first",
            region_service="regions",
            auth_service=auth_service,
            user_store=user_store,
        )
        handler = handler_type.__new__(handler_type)
        handler.headers = {}
        handler.client_address = ("192.168.30.107", 12345)
        handler.error = None
        handler._error = lambda status, message: setattr(handler, "error", (status, message))

        self.assertTrue(handler._authenticate_api_request("/api/workspaces/default/regions/first/sites"))
        self.assertEqual(handler.auth_identity.provider, "local-network")
        resolve_identity.assert_not_called()


class SiteRouteTests(unittest.TestCase):
    def test_annotation_route_uses_a_source_independent_envelope(self) -> None:
        site = SimpleNamespace(site_id="gansu_20307")
        annotation = {"geometry": {"type": "Polygon", "coordinates": []}, "properties": {}}
        response = {
            "site_id": "gansu_20307",
            "source": "osm",
            "status": "available",
            "parameters": {},
            "annotation": annotation,
        }
        catalog = SimpleNamespace(
            get_site=lambda _key: site,
            annotation_for_site=lambda _site, _source, _options: response,
        )
        handler = SimpleNamespace(catalog=catalog, payload=None)
        handler._json = lambda payload: setattr(handler, "payload", payload)
        handler._error = lambda *_args: None

        handled = handle_site_get(handler, "/api/sites/gansu_20307/annotations/osm", "")

        self.assertTrue(handled)
        self.assertEqual(
            handler.payload,
            response,
        )

    def test_unknown_annotation_source_returns_bad_request(self) -> None:
        site = SimpleNamespace(site_id="gansu_20307")

        def annotation_for_site(_site, source, _options):
            raise ValueError(f"unknown annotation source: {source}")

        handler = SimpleNamespace(
            catalog=SimpleNamespace(get_site=lambda _key: site, annotation_for_site=annotation_for_site),
            error=None,
        )
        handler._json = lambda _payload: None
        handler._error = lambda status, message: setattr(handler, "error", (status, message))

        handled = handle_site_get(handler, "/api/sites/gansu_20307/annotations/unknown", "")

        self.assertTrue(handled)
        self.assertEqual(handler.error[0].value, 400)

    def test_legacy_annotation_route_is_not_handled(self) -> None:
        handler = SimpleNamespace(catalog=SimpleNamespace())
        self.assertFalse(handle_site_get(handler, "/api/sites/gansu_20307/esa", ""))

    def test_direct_local_label_geometry_route_is_not_handled(self) -> None:
        handler = SimpleNamespace(catalog=SimpleNamespace())
        self.assertFalse(handle_site_get(handler, "/api/sites/gansu_20307/local-labels/label_1", ""))


class WorkspaceScopedRouteTests(unittest.TestCase):
    @staticmethod
    def handler(**values):
        handler = SimpleNamespace(workspace_id=None, error=None, payload=None, **values)
        handler._error = lambda status, message: setattr(handler, "error", (status, message))
        handler._json = lambda payload: setattr(handler, "payload", payload)
        return handler

    def test_logical_patch_route_rejects_missing_workspace(self) -> None:
        handler = self.handler()

        handled = handle_training_get(handler, "/api/logical-patches", "")

        self.assertTrue(handled)
        self.assertEqual(handler.error[0].value, 404)

    def test_training_sample_creation_rejects_missing_workspace(self) -> None:
        handler = self.handler()

        handled = handle_training_post(handler, "/api/sites/site-1/training-samples")

        self.assertTrue(handled)
        self.assertEqual(handler.error[0].value, 404)

    def test_model_route_rejects_missing_workspace(self) -> None:
        handler = self.handler()

        handled = handle_model_get(handler, "/api/model-validation/models", "")

        self.assertTrue(handled)
        self.assertEqual(handler.error[0].value, 404)

    def test_shared_training_samples_remain_available(self) -> None:
        handler = self.handler(catalog=SimpleNamespace(list_training_samples=lambda: {"items": [], "total": 0}))

        handled = handle_training_get(handler, "/api/training-samples", "")

        self.assertTrue(handled)
        self.assertEqual(handler.payload, {"items": [], "total": 0})


if __name__ == "__main__":
    unittest.main()
