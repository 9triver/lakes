from __future__ import annotations

import unittest
from types import SimpleNamespace

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

    def test_request_context_binds_shared_region_without_profile(self) -> None:
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
        self.assertIsNone(handler.profile_id)
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
        self.assertIsNone(handler.profile_id)
        self.assertIsNone(handler.training_runs)
        self.assertEqual(handler._bind_request_context("/api/regions/all/training-runs"), "/api/all/training-runs")
        self.assertIs(handler.catalog, first)
        self.assertEqual(handler.patch_exports, "patch-all")
        self.assertIsNone(handler.training_runs)

    def test_request_context_normalizes_profile_region_paths(self) -> None:
        first = SimpleNamespace(region=SimpleNamespace(key="first"))
        second = SimpleNamespace(region=SimpleNamespace(key="second"))
        profiles = SimpleNamespace(get=lambda profile_id, allow_archived=False: {"id": profile_id})
        handler_type = create_site_handler(
            catalogs={"first": first, "second": second},
            downloads_by_region={"first": "download-1", "second": "download-2"},
            patch_exports_by_region={"first": "patch-1", "second": "patch-2"},
            all_patch_exports="patch-all",
            default_region_key="first",
            region_service="regions",
            profile_store=profiles,
            training_manager_factory=lambda profile_id, scope: f"{profile_id}:{scope}",
        )
        handler = handler_type.__new__(handler_type)

        path = handler._bind_request_context("/api/profiles/alpha/regions/second/logical-patches")

        self.assertEqual(path, "/api/logical-patches")
        self.assertEqual(handler.profile_id, "alpha")
        self.assertIs(handler.catalog, second)
        self.assertEqual(handler.training_runs, "alpha:second")


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


class ProfileScopedRouteTests(unittest.TestCase):
    @staticmethod
    def handler(**values):
        handler = SimpleNamespace(profile_id=None, error=None, payload=None, **values)
        handler._error = lambda status, message: setattr(handler, "error", (status, message))
        handler._json = lambda payload: setattr(handler, "payload", payload)
        return handler

    def test_logical_patch_route_rejects_missing_profile(self) -> None:
        handler = self.handler()

        handled = handle_training_get(handler, "/api/logical-patches", "")

        self.assertTrue(handled)
        self.assertEqual(handler.error[0].value, 404)

    def test_training_sample_creation_rejects_missing_profile(self) -> None:
        handler = self.handler()

        handled = handle_training_post(handler, "/api/sites/site-1/training-samples")

        self.assertTrue(handled)
        self.assertEqual(handler.error[0].value, 404)

    def test_model_route_rejects_missing_profile(self) -> None:
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
