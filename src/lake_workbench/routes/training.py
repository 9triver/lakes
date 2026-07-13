"""Training sample, patch export, and model training routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.utils import truthy_flag


def handle_training_get(handler, path: str, query_string: str) -> bool:
    if path == "/api/all/training-samples":
        handler._json(handler.__class__.region_service.all_training_samples_payload())
    elif path == "/api/all/training-patches":
        include = parse_qs(query_string).get("include", [""])[0]
        handler._json(handler.__class__.region_service.all_training_patches_payload(include=include))
    elif path in {"/api/training-runs", "/api/all/training-runs"}:
        handler._json(handler.training_runs.list())
    elif re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+", path):
        job_id = path.rsplit("/", 1)[-1]
        job = handler.training_runs.get(job_id)
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Training job not found")
        else:
            handler._json(job)
    elif re.fullmatch(r"/api/lakes/[^/]+/training-samples/readiness", path):
        lake_key = path.split("/")[-3]
        lake = handler.catalog.get_lake(lake_key)
        if lake is None:
            handler._error(HTTPStatus.NOT_FOUND, "Lake not found")
        else:
            params = parse_qs(query_string)
            buffer_ratio = float(params.get("buffer_ratio", ["0.8"])[0])
            handler._json(handler.catalog.training_sample_readiness(lake, buffer_ratio=buffer_ratio))
    elif path == "/api/training-samples":
        handler._json(handler.catalog.list_training_samples())
    elif path == "/api/training-patches":
        include = parse_qs(query_string).get("include", [""])[0]
        handler._json(handler.catalog.list_training_patches(include=include))
    elif re.fullmatch(r"/api/(?:all/)?training-patches/export-jobs/[^/]+", path):
        job_id = path.rsplit("/", 1)[-1]
        job = handler.patch_exports.get(job_id)
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Patch export job not found")
        else:
            handler._json(job)
    elif re.fullmatch(r"/api/training-patches/[^/]+/preview\.png", path):
        patch_id = path.split("/")[-2]
        try:
            payload, content_type = handler.catalog.training_patch_preview(patch_id)
        except (KeyError, FileNotFoundError) as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._send_bytes(payload, content_type, cache_control="no-store")
    else:
        return False
    return True


def handle_training_post(handler, path: str) -> bool:
    if re.fullmatch(r"/api/lakes/[^/]+/training-samples", path):
        lake_key = path.split("/")[-2]
        lake = handler.catalog.get_lake(lake_key)
        if lake is None:
            handler._error(HTTPStatus.NOT_FOUND, "Lake not found")
            return True
        payload = handler._read_json()
        try:
            result = handler.catalog.create_training_sample(lake, payload)
        except ValueError as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
            return True
        patch_job = None
        if truthy_flag(payload.get("auto_patch"), default=True) and result.get("action") != "updated_existing":
            patch_job = handler.patch_exports.create(
                {
                    "sample_id": result["sample_id"],
                    "patch_size": 256,
                    "stride": 128,
                    "preview_scale": 2,
                    "overwrite": False,
                }
            )
        handler._json({"sample": result, "patch_job": patch_job})
    elif path in {"/api/training-patches/export-jobs", "/api/all/training-patches/export-jobs"}:
        handler._json(handler.patch_exports.create(handler._read_json()))
    elif path in {"/api/training-runs", "/api/all/training-runs"}:
        handler._json(handler.training_runs.create(handler._read_json()))
    elif re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+/cancel", path):
        job_id = path.split("/")[-2]
        job = handler.training_runs.cancel(job_id)
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Training job not found")
        else:
            handler._json(job)
    else:
        return False
    return True


def handle_training_patch(handler, path: str) -> bool:
    if re.fullmatch(r"/api/training-samples/[^/]+", path):
        sample_id = path.rsplit("/", 1)[-1]
        try:
            result = handler.catalog.update_training_sample(sample_id, handler._read_json())
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._json({"sample": result})
    elif re.fullmatch(r"/api/training-patches/[^/]+", path):
        patch_id = path.rsplit("/", 1)[-1]
        try:
            result = handler.catalog.update_training_patch(patch_id, handler._read_json())
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._json({"patch": result})
    else:
        return False
    return True


def handle_training_delete(handler, path: str) -> bool:
    if not re.fullmatch(r"/api/training-samples/[^/]+", path):
        return False
    sample_id = path.rsplit("/", 1)[-1]
    try:
        result = handler.catalog.delete_training_sample(sample_id)
    except KeyError as exc:
        handler._error(HTTPStatus.NOT_FOUND, str(exc))
    else:
        handler._json(result)
    return True
