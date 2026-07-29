"""Training sample, patch export, and model training routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.utils import truthy_flag


def _profile_store(handler):
    return getattr(handler.__class__, "profile_store", None)


def _has_profile(handler) -> bool:
    return bool(getattr(handler, "profile_id", None))


def _require_profile(handler) -> bool:
    if _has_profile(handler):
        return True
    handler._error(HTTPStatus.NOT_FOUND, "用户工作区路径必须包含用户")
    return False


def _profile_path(path: str) -> bool:
    return (
        path in {
            "/api/logical-patches",
            "/api/all/logical-patches",
            "/api/training-datasets",
            "/api/all/training-datasets",
            "/api/training-patches",
            "/api/all/training-patches",
            "/api/training-runs",
            "/api/all/training-runs",
            "/api/logical-patches/build-jobs",
            "/api/all/logical-patches/build-jobs",
            "/api/training-patches/export-jobs",
            "/api/all/training-patches/export-jobs",
        }
        or re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+", path) is not None
        or re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+/cancel", path) is not None
        or re.fullmatch(r"/api/(?:all/)?training-patches/export-jobs/[^/]+", path) is not None
        or re.fullmatch(r"/api/(?:all/)?training-datasets/[^/]+/build-jobs(?:/[^/]+)?", path) is not None
        or re.fullmatch(r"/api/(?:logical-patches|training-patches)/[^/]+", path) is not None
    )


def _profile_logical_payload(handler, include: str = "", site_id: str = "", sample_id: str = "", image_index: str = "") -> dict:
    store = _profile_store(handler)
    if store is None:
        return handler.catalog.list_logical_patches(include=include, site_id=site_id, sample_id=sample_id, image_index=image_index)
    manifest = store.ensure_profile_logical_patch_manifest(handler.profile_id, handler.catalog.region.key)
    payload = handler.catalog.list_logical_patches(
        include=include,
        site_id=site_id,
        sample_id=sample_id,
        image_index=image_index,
        manifest_path=manifest,
    )
    items = []
    for item in payload["items"]:
        included = bool(item.get("included"))
        variants = store.selected_variants(handler.profile_id, item.get("site_id", ""))
        items.append({
            **item,
            "profile_id": handler.profile_id,
            "included": included,
            "include": "true" if included else "false",
            "source_variants": [{key: value for key, value in variant.items() if key != "snapshot"} for variant in variants],
            "preview_url": f"/api/profiles/{handler.profile_id}/regions/{handler.catalog.region.key}/logical-patches/{item.get('logical_patch_id', '')}/preview.png",
        })
    included_count = sum(1 for item in items if item["included"])
    return {"profile_id": handler.profile_id, "total": len(items), "included_count": included_count, "excluded_count": len(items) - included_count, "items": items}


def handle_training_get(handler, path: str, query_string: str) -> bool:
    if _profile_path(path) and not _require_profile(handler):
        return True
    params = parse_qs(query_string)
    if path == "/api/all/training-samples":
        handler._json(handler.__class__.region_service.all_training_samples_payload())
    elif path == "/api/all/logical-patches":
        store = _profile_store(handler)
        handler._json(
            handler.__class__.region_service.all_profile_logical_patches_payload(
                store,
                handler.profile_id,
                include=params.get("include", [""])[0],
                site_id=params.get("site_id", [""])[0],
            )
            if store
            else handler.__class__.region_service.all_logical_patches_payload(
                include=params.get("include", [""])[0],
                site_id=params.get("site_id", [""])[0],
            )
        )
    elif path == "/api/logical-patches":
        handler._json(_profile_logical_payload(
            handler,
            include=params.get("include", [""])[0],
            site_id=params.get("site_id", [""])[0],
            sample_id=params.get("sample_id", [""])[0],
            image_index=params.get("image_index", [""])[0],
        ))
    elif path == "/api/all/training-datasets":
        store = _profile_store(handler)
        handler._json(handler.__class__.region_service.all_profile_training_dataset_statuses(store, handler.profile_id) if store else handler.__class__.region_service.all_training_dataset_statuses())
    elif path == "/api/training-datasets":
        store = _profile_store(handler)
        if store:
            from lake_workbench.training.logical_patches import dataset_configs, profile_training_dataset_status

            handler._json({"profile_id": handler.profile_id, "region": handler.catalog.region.key, "items": [profile_training_dataset_status(handler.catalog.region, value, store, handler.profile_id) for value in dataset_configs()]})
        else:
            handler._json(handler.catalog.training_dataset_statuses())
    elif path == "/api/all/training-patches":
        include = params.get("include", [""])[0]
        store = _profile_store(handler)
        handler._json(
            handler.__class__.region_service.all_profile_logical_patches_payload(
                store,
                handler.profile_id,
                include=include,
            )
        )
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
        handler._json(_profile_logical_payload(handler, include=include))
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
        if not _require_profile(handler):
            return True
        patch_id = path.split("/")[-2]
        try:
            store = _profile_store(handler)
            if store:
                from lake_workbench.training.logical_patches import profile_logical_patch_preview

                manifest = store.ensure_profile_logical_patch_manifest(handler.profile_id, handler.catalog.region.key)
                patch = handler.catalog.logical_patch_by_id(patch_id, manifest)
                payload = profile_logical_patch_preview(handler.catalog.region, patch, store, handler.profile_id)
                content_type = "image/png"
            else:
                payload, content_type = handler.catalog.training_patch_preview(patch_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._send_bytes(payload, content_type, cache_control="no-store")
    else:
        return False
    return True


def handle_training_post(handler, path: str) -> bool:
    if (
        re.fullmatch(r"/api/sites/[^/]+/training-samples", path)
        or _profile_path(path)
    ) and not _require_profile(handler):
        return True
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
        store = _profile_store(handler)
        manifest = store.ensure_profile_logical_patch_manifest(handler.profile_id, handler.catalog.region.key) if store else None
        profile_owns_sample = bool(
            manifest
            and any(
                row.get("sample_id") == result.get("sample_id")
                for row in handler.catalog.list_logical_patches(manifest_path=manifest)["items"]
            )
        )
        if truthy_flag(payload.get("auto_patch"), default=True) and (
            result.get("action") != "updated_existing" or not profile_owns_sample
        ):
            patch_job = handler.patch_exports.create(
                {
                    "sample_id": result["sample_id"],
                    "patch_size": 256,
                    "stride": 128,
                    "preview_scale": 2,
                    "overwrite": False,
                    "profile_id": handler.profile_id,
                }
            )
        handler._json({"sample": result, "patch_job": patch_job})
    elif path in {"/api/logical-patches/build-jobs", "/api/all/logical-patches/build-jobs", "/api/training-patches/export-jobs", "/api/all/training-patches/export-jobs"}:
        handler._json(handler.patch_exports.create({**handler._read_json(), "profile_id": handler.profile_id}))
    elif re.fullmatch(r"/api/(?:all/)?training-datasets/[^/]+/build-jobs", path):
        config_id = path.split("/")[-2]
        handler._json(handler.dataset_builds.create({**handler._read_json(), "config_id": config_id, "profile_id": handler.profile_id}))
    elif path in {"/api/training-runs", "/api/all/training-runs"}:
        options = handler._read_json()
        store = _profile_store(handler)
        if store:
            defaults = {key: value for key, value in options.items() if key != "run_name"}
            store.update_training_defaults(handler.profile_id, defaults)
        handler._json(handler.training_runs.create(options))
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
    if _profile_path(path) and not _require_profile(handler):
        return True
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
            store = _profile_store(handler)
            result = store.update_members(
                handler.profile_id,
                handler.catalog.region.key,
                payload.get("logical_patch_ids") or [],
                str(payload.get("operation") or ""),
            ) if store else handler.catalog.update_logical_patches(payload.get("logical_patch_ids") or [], str(payload.get("operation") or ""))
        except ValueError as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._json(result)
    elif re.fullmatch(r"/api/(?:logical-patches|training-patches)/[^/]+", path):
        patch_id = path.rsplit("/", 1)[-1]
        try:
            payload = handler._read_json()
            store = _profile_store(handler)
            if store:
                value = payload.get("include") if "include" in payload else payload.get("included")
                store.update_members(handler.profile_id, handler.catalog.region.key, [patch_id], "include" if truthy_flag(value, default=False) else "exclude")
                result = _profile_logical_payload(handler)["items"]
                result = next(item for item in result if item.get("logical_patch_id") == patch_id)
            else:
                result = handler.catalog.update_training_patch(patch_id, payload)
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
        store = _profile_store(handler)
        if store:
            result["profile_logical_patches_deleted"] = store.remove_profile_logical_patches(sample_id)
        handler._json(result)
    return True
