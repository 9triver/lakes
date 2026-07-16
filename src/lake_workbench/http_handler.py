#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP routing and response handling for Lakes Workbench."""

from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from lake_workbench.jobs import DownloadManager, PatchExportManager, TrainingManager
from lake_workbench.catalog import SiteCatalog
from lake_workbench.regions.service import RegionService
from lake_workbench.routes.frontend import handle_frontend_get
from lake_workbench.routes.sites import handle_site_get, handle_site_post
from lake_workbench.routes.models import handle_model_get
from lake_workbench.routes.regions import handle_region_get
from lake_workbench.routes.sentinel import handle_sentinel_get, handle_sentinel_post
from lake_workbench.routes.training import (
    handle_training_delete,
    handle_training_get,
    handle_training_patch,
    handle_training_post,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)




class SiteHandler(BaseHTTPRequestHandler):
    catalogs: dict[str, SiteCatalog]
    downloads_by_region: dict[str, DownloadManager]
    patch_exports_by_region: dict[str, PatchExportManager]
    dataset_builds_by_region: dict[str, PatchExportManager]
    training_runs_by_scope: dict[str, TrainingManager]
    all_patch_exports: PatchExportManager
    all_dataset_builds: PatchExportManager
    catalog: SiteCatalog
    downloads: DownloadManager
    patch_exports: PatchExportManager
    dataset_builds: PatchExportManager
    training_runs: TrainingManager
    region_service: RegionService

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            path = self._bind_request_context(path)
            if path is None:
                return

            if handle_frontend_get(self, path, STATIC_DIR):
                return
            if handle_region_get(self, path, parsed.query):
                return
            if handle_training_get(self, path, parsed.query):
                return
            if handle_model_get(self, path, parsed.query):
                return
            if handle_sentinel_get(self, path, parsed.query):
                return
            if handle_site_get(self, path, parsed.query):
                return
            if path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "Not found")
            else:
                self._serve_file(STATIC_DIR / "dist" / "index.html")
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            path = self._bind_request_context(path)
            if path is None:
                return

            if handle_training_post(self, path):
                return
            if handle_sentinel_post(self, path):
                return
            if handle_site_post(self, path):
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found")
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            path = self._bind_request_context(path)
            if path is None:
                return

            if not handle_training_patch(self, path):
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            path = self._bind_request_context(path)
            if path is None:
                return

            if not handle_training_delete(self, path):
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _bind_request_context(self, path: str) -> str | None:
        region_key = self.__class__.catalog.region.key
        normalized_path = path
        if path.startswith("/api/regions/all/"):
            region_key = "all"
            normalized_path = "/api/all" + path.removeprefix("/api/regions/all")
        elif path.startswith("/api/regions/"):
            match = re.match(r"^/api/regions/([^/]+)(/.*)?$", path)
            requested_region = match.group(1) if match else ""
            if requested_region not in self.__class__.catalogs:
                self._error(HTTPStatus.NOT_FOUND, f"Region not found: {requested_region}")
                return None
            region_key = requested_region
            normalized_path = "/api" + (match.group(2) or "")

        if region_key == "all":
            self.catalog = self.__class__.catalog
            self.downloads = self.__class__.downloads
            self.patch_exports = self.__class__.all_patch_exports
            self.dataset_builds = self.__class__.all_dataset_builds
        else:
            self.catalog = self.__class__.catalogs[region_key]
            self.downloads = self.__class__.downloads_by_region[region_key]
            self.patch_exports = self.__class__.patch_exports_by_region[region_key]
            self.dataset_builds = self.__class__.dataset_builds_by_region[region_key]
        self.training_runs = self.__class__.training_runs_by_scope[region_key]
        return normalized_path


    def _json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def _error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        payload: bytes,
        content_type: str,
        *,
        cache_control: str = "no-store",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists():
            self._error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def create_site_handler(
    *,
    catalogs: dict[str, SiteCatalog],
    downloads_by_region: dict[str, DownloadManager],
    patch_exports_by_region: dict[str, PatchExportManager],
    training_runs_by_scope: dict[str, TrainingManager],
    all_patch_exports: PatchExportManager,
    default_region_key: str,
    region_service: RegionService,
    dataset_builds_by_region: dict[str, PatchExportManager] | None = None,
    all_dataset_builds: PatchExportManager | None = None,
) -> type[SiteHandler]:
    """Bind one server's runtime dependencies to an isolated handler class."""

    class ConfiguredSiteHandler(SiteHandler):
        pass

    ConfiguredSiteHandler.catalogs = catalogs
    ConfiguredSiteHandler.downloads_by_region = downloads_by_region
    ConfiguredSiteHandler.patch_exports_by_region = patch_exports_by_region
    ConfiguredSiteHandler.dataset_builds_by_region = dataset_builds_by_region or patch_exports_by_region
    ConfiguredSiteHandler.training_runs_by_scope = training_runs_by_scope
    ConfiguredSiteHandler.all_patch_exports = all_patch_exports
    ConfiguredSiteHandler.all_dataset_builds = all_dataset_builds or all_patch_exports
    ConfiguredSiteHandler.catalog = catalogs[default_region_key]
    ConfiguredSiteHandler.downloads = downloads_by_region[default_region_key]
    ConfiguredSiteHandler.patch_exports = patch_exports_by_region[default_region_key]
    ConfiguredSiteHandler.dataset_builds = ConfiguredSiteHandler.dataset_builds_by_region[default_region_key]
    ConfiguredSiteHandler.training_runs = training_runs_by_scope[default_region_key]
    ConfiguredSiteHandler.region_service = region_service
    return ConfiguredSiteHandler
