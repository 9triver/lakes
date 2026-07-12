#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP routing and response handling for Lakes Workbench."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from lake_workbench.sentinel_download import (
    query_copernicus_tile_products,
)
from lake_workbench.jobs import DownloadManager, PatchExportManager, TrainingManager
from lake_workbench.catalog import LakeCatalog
from lake_workbench.model_validation import ModelInferenceBusy
from lake_workbench.imagery import (
    blank_png,
)
from lake_workbench.region_service import RegionService
from lake_workbench.utils import (
    default_sentinel_date_range,
    is_frontend_route,
    parse_float_or_default,
    truthy_flag,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
TILE_RENDER_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.environ.get("LAKES_TILE_RENDER_WORKERS", "1"))))
MODEL_VALIDATION_SEMAPHORE = threading.BoundedSemaphore(1)




class LakeHandler(BaseHTTPRequestHandler):
    catalogs: dict[str, LakeCatalog]
    downloads_by_region: dict[str, DownloadManager]
    patch_exports_by_region: dict[str, PatchExportManager]
    training_runs_by_scope: dict[str, TrainingManager]
    all_patch_exports: PatchExportManager
    catalog: LakeCatalog
    downloads: DownloadManager
    patch_exports: PatchExportManager
    training_runs: TrainingManager
    region_service: RegionService

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if is_frontend_route(path):
                self._serve_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                self._serve_file(STATIC_DIR / path.removeprefix("/static/"))
            elif path == "/api/regions":
                self._json(self.__class__.region_service.regions_payload())
            elif path == "/api/all/lakes":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["200"])[0])
                offset = int(params.get("offset", ["0"])[0])
                filters = {
                    key: params.get(key, [""])[0]
                    for key in [
                        "water_type",
                        "province",
                        "city",
                        "county",
                        "polygon_quality",
                        "metadata_quality",
                        "area_bucket",
                        "has_tci",
                        "has_name",
                        "min_area",
                        "max_area",
                    ]
                }
                self._json(self.__class__.region_service.all_lakes_payload(query=query, limit=limit, offset=offset, filters=filters))
            elif path == "/api/all/training-samples":
                self._json(self.__class__.region_service.all_training_samples_payload())
            elif path == "/api/all/training-patches":
                params = parse_qs(parsed.query)
                include = params.get("include", [""])[0]
                self._json(self.__class__.region_service.all_training_patches_payload(include=include))
            elif path == "/api/all/training-runs":
                self._json(self.training_runs.list())
            elif re.fullmatch(r"/api/all/training-runs/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.training_runs.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Training job not found")
                    return
                self._json(job)
            elif path == "/api/all/model-validation/models":
                self._json(self.__class__.region_service.all_model_validation_models_payload())
            elif path == "/api/all/model-validation/random":
                params = parse_qs(parsed.query)
                threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
                model_key = params.get("model", [""])[0]
                if not MODEL_VALIDATION_SEMAPHORE.acquire(blocking=False):
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, "模型推理正在运行，请稍后再试")
                    return
                try:
                    self._json(self.__class__.region_service.all_model_validation_random(threshold=threshold, model_key=model_key))
                except ModelInferenceBusy as exc:
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
                    return
                except (FileNotFoundError, ValueError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                finally:
                    MODEL_VALIDATION_SEMAPHORE.release()
            elif path == "/api/lakes":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["200"])[0])
                offset = int(params.get("offset", ["0"])[0])
                filters = {
                    key: params.get(key, [""])[0]
                    for key in [
                        "water_type",
                        "province",
                        "city",
                        "county",
                        "polygon_quality",
                        "metadata_quality",
                        "area_bucket",
                        "has_tci",
                        "has_name",
                        "min_area",
                        "max_area",
                    ]
                }
                self._json(self.catalog.list_lakes(query=query, limit=limit, offset=offset, filters=filters))
            elif re.fullmatch(r"/api/lakes/[^/]+", path):
                lake_key = path.rsplit("/", 1)[-1]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.get_lake_detail(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/image.png", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                size = int(params.get("size", ["900"])[0])
                padding = float(params.get("padding", ["0.6"])[0])
                try:
                    payload, meta = self.catalog.image_for_lake(lake, size=size, padding=padding)
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("X-Image-Meta", json.dumps(meta, ensure_ascii=True))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif re.fullmatch(r"/api/lakes/[^/]+/tile-meta", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                try:
                    self._json(self.catalog.tile_meta_for_lake(lake, padding=padding))
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif re.fullmatch(r"/api/lakes/[^/]+/tiles/\d+/\d+/\d+\.png", path):
                match = re.fullmatch(r"/api/lakes/([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.png", path)
                lake = self.catalog.get_lake(match.group(1))
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                try:
                    with TILE_RENDER_SEMAPHORE:
                        payload, _meta = self.catalog.tile_png_for_lake(
                            lake,
                            z=int(match.group(2)),
                            x=int(match.group(3)),
                            y=int(match.group(4)),
                            padding=padding,
                        )
                except FileNotFoundError:
                    payload = blank_png(256)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=600")
                self.end_headers()
                self.wfile.write(payload)
            elif re.fullmatch(r"/api/lakes/[^/]+/esa", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json({"esa": self.catalog._esa_smoothed_layer(lake)})
            elif re.fullmatch(r"/api/lakes/[^/]+/jrc", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                threshold = int(params.get("threshold", ["75"])[0])
                self._json({"jrc": self.catalog._jrc_occurrence_layer(lake, threshold=threshold)})
            elif re.fullmatch(r"/api/lakes/[^/]+/sentinel/tiles", path):
                lake_key = path.split("/")[-3]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.sentinel_tiles_for_lake(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/context-water", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                min_area_km2 = float(params.get("min_area_km2", ["1.0"])[0])
                limit = int(params.get("limit", ["500"])[0])
                self._json(self.catalog.context_water_for_lake(lake, padding=padding, min_area_km2=min_area_km2, limit=limit))
            elif re.fullmatch(r"/api/lakes/[^/]+/local-labels", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.local_label_items(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/local-labels/[^/]+", path):
                parts = path.split("/")
                lake_key = parts[-3]
                label_id = parts[-1]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                try:
                    self._json(self.catalog.local_label_geojson(lake, label_id))
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif re.fullmatch(r"/api/lakes/[^/]+/imagery", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.imagery_for_lake(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/training-samples/readiness", path):
                lake_key = path.split("/")[-3]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                buffer_ratio = float(params.get("buffer_ratio", ["0.8"])[0])
                self._json(self.catalog.training_sample_readiness(lake, buffer_ratio=buffer_ratio))
            elif path == "/api/training-samples":
                self._json(self.catalog.list_training_samples())
            elif path == "/api/training-patches":
                params = parse_qs(parsed.query)
                include = params.get("include", [""])[0]
                self._json(self.catalog.list_training_patches(include=include))
            elif re.fullmatch(r"/api/(?:all/)?training-patches/export-jobs/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.patch_exports.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Patch export job not found")
                    return
                self._json(job)
            elif path == "/api/training-runs":
                self._json(self.training_runs.list())
            elif re.fullmatch(r"/api/training-runs/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.training_runs.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Training job not found")
                    return
                self._json(job)
            elif re.fullmatch(r"/api/training-patches/[^/]+/preview\.png", path):
                patch_id = path.split("/")[-2]
                try:
                    payload, content_type = self.catalog.training_patch_preview(patch_id)
                except (KeyError, FileNotFoundError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif path == "/api/model-validation/models":
                self._json(self.catalog.model_validation_models())
            elif path == "/api/model-validation/random":
                params = parse_qs(parsed.query)
                threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
                model_key = params.get("model", [""])[0]
                if not MODEL_VALIDATION_SEMAPHORE.acquire(blocking=False):
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, "模型推理正在运行，请稍后再试")
                    return
                try:
                    self._json(self.catalog.model_validation_random(threshold=threshold, model_key=model_key))
                except ModelInferenceBusy as exc:
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
                    return
                except (FileNotFoundError, ValueError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                finally:
                    MODEL_VALIDATION_SEMAPHORE.release()
            elif re.fullmatch(r"/api/lakes/[^/]+/model-prediction", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
                model_key = params.get("model", [""])[0]
                if not MODEL_VALIDATION_SEMAPHORE.acquire(blocking=False):
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, "模型推理正在运行，请稍后再试")
                    return
                try:
                    self._json(self.catalog.model_prediction_for_lake(lake, threshold=threshold, model_key=model_key))
                except ModelInferenceBusy as exc:
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
                    return
                except (FileNotFoundError, ValueError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                finally:
                    MODEL_VALIDATION_SEMAPHORE.release()
            elif path == "/api/sentinel/products":
                params = parse_qs(parsed.query)
                tile = params.get("tile", [""])[0]
                if not tile:
                    self._error(HTTPStatus.BAD_REQUEST, "tile is required")
                    return
                lake = None
                lake_key = params.get("lake_id", [""])[0]
                if lake_key:
                    lake = self.catalog.get_lake(lake_key)
                default_start, default_end = default_sentinel_date_range()
                start = params.get("start", [default_start])[0]
                end = params.get("end", [default_end])[0]
                cloud = float(params.get("cloud", ["50"])[0])
                product_type = params.get("product_type", ["MSIL1C"])[0]
                limit = int(params.get("limit", ["50"])[0])
                products = query_copernicus_tile_products(tile, start, end, cloud, product_type, limit)
                products = self.catalog.enrich_products_for_lake(lake, products)
                enriched = []
                for product in products:
                    local = self.catalog.local_product_status(product.get("product_id"), product.get("name"))
                    enriched.append({**product, **local})
                products = enriched
                self._json({
                    "tile": str(tile).upper().removeprefix("T"),
                    "start": start,
                    "end": end,
                    "cloud": cloud,
                    "product_type": product_type,
                    "lake_id": lake.object_id if lake else None,
                    "products": products,
                })
            elif re.fullmatch(r"/api/sentinel/downloads/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.downloads.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Download job not found")
                    return
                self._json(job)
            elif re.fullmatch(r"/api/lakes/[^/]+/image-meta", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                try:
                    _, meta = self.catalog.image_for_lake(lake)
                    self._json(meta)
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "Not found")
            else:
                self._serve_file(STATIC_DIR / "index.html")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if path == "/api/sentinel/downloads":
                payload = self._read_json()
                product = payload.get("product") or payload
                if not product.get("product_id") or not product.get("name"):
                    self._error(HTTPStatus.BAD_REQUEST, "product_id and name are required")
                    return
                status = self.catalog.local_product_status(product.get("product_id"), product.get("name"))
                if status.get("downloaded"):
                    self._json({
                        "job_id": None,
                        "status": "completed",
                        "message": "产品已在本地",
                        "progress": 100,
                        "result": status,
                        "product": product,
                    })
                    return
                self._json(self.downloads.create(product))
            elif re.fullmatch(r"/api/lakes/[^/]+/imagery/active", path):
                lake_key = path.split("/")[-3]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                payload = self._read_json()
                tile = payload.get("tile")
                product = payload.get("product")
                if not tile or not product:
                    self._error(HTTPStatus.BAD_REQUEST, "tile and product are required")
                    return
                try:
                    result = self.catalog.set_active_imagery(tile, product, lake)
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json(result)
            elif re.fullmatch(r"/api/lakes/[^/]+/training-samples", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                payload = self._read_json()
                try:
                    result = self.catalog.create_training_sample(lake, payload)
                except ValueError as exc:
                    self._error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                patch_job = None
                auto_patch = truthy_flag(payload.get("auto_patch"), default=True)
                if auto_patch and result.get("action") != "updated_existing":
                    patch_job = self.patch_exports.create({
                        "sample_id": result["sample_id"],
                        "patch_size": 256,
                        "stride": 128,
                        "preview_scale": 2,
                        "overwrite": False,
                    })
                self._json({"sample": result, "patch_job": patch_job})
            elif path in {"/api/training-patches/export-jobs", "/api/all/training-patches/export-jobs"}:
                self._json(self.patch_exports.create(self._read_json()))
            elif path in {"/api/training-runs", "/api/all/training-runs"}:
                self._json(self.training_runs.create(self._read_json()))
            elif re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+/cancel", path):
                job_id = path.split("/")[-2]
                job = self.training_runs.cancel(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Training job not found")
                    return
                self._json(job)
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if re.fullmatch(r"/api/training-samples/[^/]+", path):
                sample_id = path.rsplit("/", 1)[-1]
                try:
                    result = self.catalog.update_training_sample(sample_id, self._read_json())
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json({"sample": result})
            elif re.fullmatch(r"/api/training-patches/[^/]+", path):
                patch_id = path.rsplit("/", 1)[-1]
                try:
                    result = self.catalog.update_training_patch(patch_id, self._read_json())
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json({"patch": result})
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if re.fullmatch(r"/api/training-samples/[^/]+", path):
                sample_id = path.rsplit("/", 1)[-1]
                try:
                    result = self.catalog.delete_training_sample(sample_id)
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json(result)
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _route_region_path(self, path: str) -> tuple[str, LakeCatalog, DownloadManager] | None:
        match = re.match(r"^/api/regions/([^/]+)(/.*)?$", path)
        if not match:
            return path, self.__class__.catalog, self.__class__.downloads
        region_key = match.group(1)
        if region_key not in self.__class__.catalogs:
            self._error(HTTPStatus.NOT_FOUND, f"Region not found: {region_key}")
            return None
        subpath = "/api" + (match.group(2) or "")
        return subpath, self.__class__.catalogs[region_key], self.__class__.downloads_by_region[region_key]


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


def create_lake_handler(
    *,
    catalogs: dict[str, LakeCatalog],
    downloads_by_region: dict[str, DownloadManager],
    patch_exports_by_region: dict[str, PatchExportManager],
    training_runs_by_scope: dict[str, TrainingManager],
    all_patch_exports: PatchExportManager,
    default_region_key: str,
    region_service: RegionService,
) -> type[LakeHandler]:
    """Bind one server's runtime dependencies to an isolated handler class."""

    class ConfiguredLakeHandler(LakeHandler):
        pass

    ConfiguredLakeHandler.catalogs = catalogs
    ConfiguredLakeHandler.downloads_by_region = downloads_by_region
    ConfiguredLakeHandler.patch_exports_by_region = patch_exports_by_region
    ConfiguredLakeHandler.training_runs_by_scope = training_runs_by_scope
    ConfiguredLakeHandler.all_patch_exports = all_patch_exports
    ConfiguredLakeHandler.catalog = catalogs[default_region_key]
    ConfiguredLakeHandler.downloads = downloads_by_region[default_region_key]
    ConfiguredLakeHandler.patch_exports = patch_exports_by_region[default_region_key]
    ConfiguredLakeHandler.training_runs = training_runs_by_scope[default_region_key]
    ConfiguredLakeHandler.region_service = region_service
    return ConfiguredLakeHandler
