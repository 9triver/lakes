"""Observation-site metadata, imagery, labels, and water-layer routes."""

import json
import os
import re
import threading
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.automatic_labels import (
    DERIVED_LABEL_SOURCES,
    generate_derived_label,
)
from lake_workbench.imagery.display import blank_png
from lake_workbench.routes.regions import site_list_options


TILE_RENDER_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.environ.get("LAKES_TILE_RENDER_WORKERS", "1"))))
DERIVED_LABEL_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(os.environ.get("LAKES_DERIVED_LABEL_WORKERS", "1")))
)


def _site(handler, site_key: str):
    site = handler.catalog.get_site(site_key)
    if site is None:
        handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
    return site


def _workspace_logical_manifest(handler):
    workspace_id = getattr(handler, "workspace_id", None)
    store = getattr(handler.__class__, "workspace_store", None)
    if not workspace_id or store is None:
        handler._error(HTTPStatus.NOT_FOUND, "逻辑 Patch 路径必须包含 workspace")
        return None
    return store.ensure_workspace_logical_patch_manifest(workspace_id, handler.catalog.region.key)


def handle_site_get(handler, path: str, query_string: str) -> bool:
    params = parse_qs(query_string)
    if path == "/api/sites":
        query, limit, offset, filters = site_list_options(query_string)
        payload = handler.catalog.list_sites(query=query, limit=limit, offset=offset, filters=filters)
        store = getattr(handler.__class__, "workspace_store", None)
        if store and getattr(handler, "workspace_id", None):
            counts = store.patch_counts(handler.workspace_id, handler.catalog.region.key)
            payload["items"] = [{**item, "included_logical_patch_count": counts.get(item.get("site_id", ""), 0)} for item in payload["items"]]
        handler._json(payload)
    elif re.fullmatch(r"/api/sites/[^/]+", path):
        site = _site(handler, path.rsplit("/", 1)[-1])
        if site is not None:
            payload = handler.catalog.get_site_detail(site)
            store = getattr(handler.__class__, "workspace_store", None)
            if store and getattr(handler, "workspace_id", None):
                payload["included_logical_patch_count"] = store.patch_counts(handler.workspace_id, handler.catalog.region.key).get(site.site_id, 0)
            handler._json(payload)
    elif re.fullmatch(r"/api/sites/[^/]+/image\.png", path):
        site = _site(handler, path.split("/")[-2])
        if site is not None:
            size = int(params.get("size", ["900"])[0])
            padding = float(params.get("padding", ["0.6"])[0])
            try:
                payload, meta = handler.catalog.image_for_site(site, size=size, padding=padding)
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
            else:
                handler._send_bytes(
                    payload,
                    "image/png",
                    headers={"X-Image-Meta": json.dumps(meta, ensure_ascii=True)},
                )
    elif re.fullmatch(r"/api/sites/[^/]+/tile-meta", path):
        site = _site(handler, path.split("/")[-2])
        if site is not None:
            try:
                handler._json(handler.catalog.tile_meta_for_site(site, padding=float(params.get("padding", ["0.8"])[0])))
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
    elif re.fullmatch(r"/api/sites/[^/]+/tiles/\d+/\d+/\d+\.png", path):
        match = re.fullmatch(r"/api/sites/([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.png", path)
        site = _site(handler, match.group(1))
        if site is not None:
            try:
                with TILE_RENDER_SEMAPHORE:
                    payload, _meta = handler.catalog.tile_png_for_site(
                        site,
                        z=int(match.group(2)),
                        x=int(match.group(3)),
                        y=int(match.group(4)),
                        padding=float(params.get("padding", ["0.8"])[0]),
                    )
            except FileNotFoundError:
                payload = blank_png(256)
            handler._send_bytes(payload, "image/png", cache_control="public, max-age=600")
    elif re.fullmatch(r"/api/sites/[^/]+/logical-patch-source/meta", path):
        manifest = _workspace_logical_manifest(handler)
        if manifest is None:
            return True
        site = _site(handler, path.split("/")[-3])
        patch_id = params.get("patch_id", [""])[0]
        if site is not None:
            try:
                patch = handler.catalog.logical_patch_by_id(patch_id, manifest)
                if patch.get("site_id") != site.site_id:
                    raise KeyError(f"logical patch does not belong to site: {patch_id}")
                payload = handler.catalog.logical_patch_source_meta(patch_id, manifest)
                payload["tile_url"] = (
                    f"/api/workspaces/{handler.workspace_id}/regions/{handler.catalog.region.key}"
                    f"/sites/{site.site_id}/logical-patch-source/tiles/{{z}}/{{x}}/{{y}}.png?patch_id={patch_id}"
                )
                handler._json(payload)
            except (KeyError, FileNotFoundError) as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
    elif re.fullmatch(r"/api/sites/[^/]+/logical-patch-source/tiles/\d+/\d+/\d+\.png", path):
        manifest = _workspace_logical_manifest(handler)
        if manifest is None:
            return True
        match = re.fullmatch(r"/api/sites/([^/]+)/logical-patch-source/tiles/(\d+)/(\d+)/(\d+)\.png", path)
        site = _site(handler, match.group(1))
        patch_id = params.get("patch_id", [""])[0]
        if site is not None:
            try:
                patch = handler.catalog.logical_patch_by_id(patch_id, manifest)
                if patch.get("site_id") != site.site_id:
                    raise KeyError(f"logical patch does not belong to site: {patch_id}")
                with TILE_RENDER_SEMAPHORE:
                    payload = handler.catalog.logical_patch_source_tile(patch_id, int(match.group(2)), int(match.group(3)), int(match.group(4)), manifest)
            except (KeyError, FileNotFoundError):
                payload = blank_png(256)
            handler._send_bytes(payload, "image/png", cache_control="public, max-age=600")
    elif re.fullmatch(r"/api/sites/[^/]+/annotations/[^/]+", path):
        parts = path.split("/")
        site = _site(handler, parts[-3])
        if site is not None:
            source = parts[-1]
            options = {key: values[0] for key, values in params.items() if values}
            try:
                payload = handler.catalog.annotation_for_site(site, source, options)
            except ValueError as exc:
                handler._error(HTTPStatus.BAD_REQUEST, str(exc))
            else:
                handler._json(payload)
    elif re.fullmatch(r"/api/sites/[^/]+/context-water", path):
        site = _site(handler, path.split("/")[-2])
        if site is not None:
            handler._json(
                handler.catalog.context_water_for_site(
                    site,
                    padding=float(params.get("padding", ["0.8"])[0]),
                    min_area_km2=float(params.get("min_area_km2", ["1.0"])[0]),
                    limit=int(params.get("limit", ["500"])[0]),
                )
            )
    elif re.fullmatch(r"/api/sites/[^/]+/local-labels", path):
        site = _site(handler, path.split("/")[-2])
        if site is not None:
            handler._json(handler.catalog.local_label_items(site))
    elif re.fullmatch(r"/api/sites/[^/]+/imagery", path):
        site = _site(handler, path.split("/")[-2])
        if site is not None:
            handler._json(handler.catalog.imagery_for_site(site))
    elif re.fullmatch(r"/api/sites/[^/]+/image-meta", path):
        site = _site(handler, path.split("/")[-2])
        if site is not None:
            try:
                _, meta = handler.catalog.image_for_site(site)
                handler._json(meta)
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
    else:
        return False
    return True


def handle_site_post(handler, path: str) -> bool:
    generated_label_match = re.fullmatch(
        r"/api/sites/([^/]+)/generated-labels/([^/]+)", path
    )
    if generated_label_match:
        site = _site(handler, generated_label_match.group(1))
        if site is None:
            return True
        source = generated_label_match.group(2)
        if source not in DERIVED_LABEL_SOURCES:
            handler._error(
                HTTPStatus.BAD_REQUEST,
                f"Unknown generated annotation source: {source}",
            )
            return True
        workspace_id = getattr(handler, "workspace_id", None)
        store = getattr(handler.__class__, "workspace_store", None)
        if not workspace_id or store is None:
            handler._error(
                HTTPStatus.NOT_FOUND,
                "Generated annotations require a workspace-scoped path",
            )
            return True
        payload = handler._read_json()
        try:
            with DERIVED_LABEL_SEMAPHORE:
                result = generate_derived_label(
                    handler.catalog,
                    site,
                    source,
                    payload,
                    store.workspace_derived_label_dir(
                        workspace_id, handler.catalog.region.key
                    ),
                )
        except FileNotFoundError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
        else:
            handler._json(result)
        return True
    if not re.fullmatch(r"/api/sites/[^/]+/imagery/active", path):
        return False
    site = _site(handler, path.split("/")[-3])
    if site is None:
        return True
    payload = handler._read_json()
    asset_id = payload.get("asset_id")
    if asset_id:
        try:
            result = handler.catalog.set_active_site_imagery(site, str(asset_id), str(payload.get("product") or ""))
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._json(result)
        return True
    tile = payload.get("tile")
    product = payload.get("product")
    if not tile or not product:
        handler._error(HTTPStatus.BAD_REQUEST, "tile and product are required")
        return True
    try:
        result = handler.catalog.set_active_imagery(tile, product, site)
    except KeyError as exc:
        handler._error(HTTPStatus.NOT_FOUND, str(exc))
    else:
        handler._json(result)
    return True
