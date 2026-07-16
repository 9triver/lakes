"""Training sample, patch export, and model training routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.utils import truthy_flag


def handle_training_get(handler, path: str, query_string: str) -> bool:
    params = parse_qs(query_string)
    if path == "/api/all/training-samples":
        handler._json(handler.__class__.region_service.all_training_samples_payload())
    elif path == "/api/all/logical-patches":
        handler._json(
            handler.__class__.region_service.all_logical_patches_payload(
                include=params.get("include", [""])[0],
                site_id=params.get("site_id", [""])[0],
            )
        )
    elif path == "/api/logical-patches":
        handler._json(
            handler.catalog.list_logical_patches(
                include=params.get("include", [""])[0],
                site_id=params.get("site_id", [""])[0],
                sample_id=params.get("sample_id", [""])[0],
                image_index=params.get("image_index", [""])[0],
            )
        )
    elif path == "/api/all/training-datasets":
        handler._json(handler.__class__.region_service.all_training_dataset_statuses())
    elif path == "/api/training-datasets":
        handler._json(handler.catalog.training_dataset_statuses())
    elif path == "/api/all/training-patches":
        include = params.get("include", [""])[0]
        handler._json(handler.__class__.region_service.all_training_patches_payload(include=include))
    elif path in {"/api/training-runs", "/api/all/training-runs"}:
        handler._json(handler.training_runs.list(params.get("dataset_config_id", ["resize256_v1"])[0]))
    elif re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+", path):
        job_id = path.rsplit("/", 1)[-1]
        job = handler.training_runs.get(job_id)
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Training job not found")
        else:
            handler._json(job)
    elif re.fullmatch(r"/api/sites/[^/]+/training-samples/readiness", path):
        site_key = path.split("/")[-3]
        site = handler.catalog.get_site(site_key)
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
        else:
            buffer_ratio = float(params.get("buffer_ratio", ["0.8"])[0])
            handler._json(handler.catalog.training_sample_readiness(site, buffer_ratio=buffer_ratio))
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
    elif re.fullmatch(r"/api/(?:all/)?training-datasets/[^/]+/build-jobs/[^/]+", path):
        job_id = path.rsplit("/", 1)[-1]
        job = handler.dataset_builds.get(job_id)
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Training dataset build job not found")
        else:
            handler._json(job)
    elif re.fullmatch(r"/api/(?:logical-patches|training-patches)/[^/]+/preview\.png", path):
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
    if re.fullmatch(r"/api/sites/[^/]+/training-samples", path):
        site_key = path.split("/")[-2]
        site = handler.catalog.get_site(site_key)
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
            return True
        payload = handler._read_json()
        try:
            result = handler.catalog.create_training_sample(site, payload)
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
    elif path in {"/api/logical-patches/build-jobs", "/api/all/logical-patches/build-jobs", "/api/training-patches/export-jobs", "/api/all/training-patches/export-jobs"}:
        handler._json(handler.patch_exports.create(handler._read_json()))
    elif re.fullmatch(r"/api/(?:all/)?training-datasets/[^/]+/build-jobs", path):
        config_id = path.split("/")[-2]
        handler._json(handler.dataset_builds.create({**handler._read_json(), "config_id": config_id}))
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
    elif path == "/api/logical-patches":
        payload = handler._read_json()
        try:
            result = handler.catalog.update_logical_patches(payload.get("logical_patch_ids") or [], str(payload.get("operation") or ""))
        except ValueError as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._json(result)
    elif re.fullmatch(r"/api/(?:logical-patches|training-patches)/[^/]+", path):
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
