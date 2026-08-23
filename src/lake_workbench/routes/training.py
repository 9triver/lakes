"""Training sample, patch export, and model training routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.jobs import TrainingRunActiveError
from lake_workbench.utils import truthy_flag
from lake_workbench.training.registry import DatasetConflict
from lake_workbench.workspaces import WorkspacePatchConflict


def _workspace_store(handler):
    return getattr(handler.__class__, "workspace_store", None)


def _has_workspace(handler) -> bool:
    return bool(getattr(handler, "workspace_id", None))


def _require_workspace(handler) -> bool:
    if _has_workspace(handler):
        return True
    handler._error(HTTPStatus.NOT_FOUND, "训练资源路径必须包含 workspace")
    return False


def _workspace_path(path: str) -> bool:
    return (
        path in {
            "/api/logical-patches",
            "/api/all/logical-patches",
            "/api/training-datasets",
            "/api/all/training-datasets",
            "/api/training-patches",
            "/api/all/training-patches",
            "/api/global-dataset",
            "/api/all/global-dataset",
            "/api/global-dataset/build-jobs",
            "/api/all/global-dataset/build-jobs",
            "/api/training-runs",
            "/api/all/training-runs",
            "/api/logical-patches/build-jobs",
            "/api/all/logical-patches/build-jobs",
            "/api/training-patches/export-jobs",
            "/api/all/training-patches/export-jobs",
            "/api/global-dataset/patches/withdraw",
            "/api/all/global-dataset/patches/withdraw",
        }
        or re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+", path) is not None
        or re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+/cancel", path) is not None
        or re.fullmatch(r"/api/(?:all/)?training-patches/export-jobs/[^/]+", path) is not None
        or re.fullmatch(r"/api/(?:all/)?training-datasets/[^/]+/build-jobs(?:/[^/]+)?", path) is not None
        or re.fullmatch(r"/api/(?:logical-patches|training-patches)/[^/]+", path) is not None
        or path in {"/api/global-dataset/patches", "/api/all/global-dataset/patches"}
        or re.fullmatch(r"/api/(?:all/)?global-dataset/build-jobs/[^/]+", path) is not None
    )


def _workspace_logical_payload(handler, include: str = "", site_id: str = "", sample_id: str = "", image_index: str = "") -> dict:
    store = _workspace_store(handler)
    manifest = store.ensure_workspace_logical_patch_manifest(handler.workspace_id, handler.catalog.region.key)
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
        items.append({
            **item,
            "workspace_id": handler.workspace_id,
            "included": included,
            "include": "true" if included else "false",
            "preview_url": f"/api/workspaces/{handler.workspace_id}/regions/{handler.catalog.region.key}/logical-patches/{item.get('logical_patch_id', '')}/preview.png",
        })
    _annotate_contributions(handler, items)
    included_count = sum(1 for item in items if item["included"])
    return {"workspace_id": handler.workspace_id, "total": len(items), "included_count": included_count, "excluded_count": len(items) - included_count, "items": items}


def _annotate_contributions(handler, items: list[dict]) -> None:
    patch_ids = [item.get("logical_patch_id", "") for item in items if item.get("logical_patch_id")]
    scopes = handler.__class__.dataset_registry.contribution_scopes(handler.workspace_id, patch_ids)
    for item in items:
        contribution_scopes = scopes.get(item.get("logical_patch_id", ""), [])
        item["contribution_scopes"] = contribution_scopes
        item["contributed"] = bool(contribution_scopes)


def handle_training_get(handler, path: str, query_string: str) -> bool:
    if _workspace_path(path) and not _require_workspace(handler):
        return True
    params = parse_qs(query_string)
    if path == "/api/all/training-samples":
        store = _workspace_store(handler)
        handler._json(handler.__class__.region_service.all_training_samples_payload(store, handler.workspace_id))
    elif path in {"/api/global-dataset", "/api/all/global-dataset"}:
        handler._json(handler.__class__.dataset_registry.list("all" if path.startswith("/api/all/") else handler.catalog.region.key))
    elif re.fullmatch(r"/api/(?:all/)?global-dataset/build-jobs/[^/]+", path):
        job = handler.__class__.global_dataset_builds.get(path.rsplit("/", 1)[-1])
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Global Dataset build job not found")
        else:
            handler._json(job)
    elif path == "/api/all/logical-patches":
        store = _workspace_store(handler)
        payload = handler.__class__.region_service.all_workspace_logical_patches_payload(
            store,
            handler.workspace_id,
            include=params.get("include", [""])[0],
            site_id=params.get("site_id", [""])[0],
            sample_id=params.get("sample_id", [""])[0],
        )
        _annotate_contributions(handler, payload["items"])
        handler._json(payload)
    elif path == "/api/logical-patches":
        handler._json(_workspace_logical_payload(
            handler,
            include=params.get("include", [""])[0],
            site_id=params.get("site_id", [""])[0],
            sample_id=params.get("sample_id", [""])[0],
            image_index=params.get("image_index", [""])[0],
        ))
    elif path == "/api/all/training-datasets":
        store = _workspace_store(handler)
        handler._json(handler.__class__.region_service.all_workspace_training_dataset_statuses(store, handler.workspace_id))
    elif path == "/api/training-datasets":
        store = _workspace_store(handler)
        from lake_workbench.training.logical_patches import dataset_configs, workspace_training_dataset_status

        handler._json({"workspace_id": handler.workspace_id, "region": handler.catalog.region.key, "items": [workspace_training_dataset_status(handler.catalog.region, value, store, handler.workspace_id) for value in dataset_configs()]})
    elif path == "/api/all/training-patches":
        include = params.get("include", [""])[0]
        store = _workspace_store(handler)
        handler._json(
            handler.__class__.region_service.all_workspace_logical_patches_payload(
                store,
                handler.workspace_id,
                include=include,
            )
        )
    elif path in {"/api/training-runs", "/api/all/training-runs"}:
        handler._json(handler.training_runs.list(
            params.get("dataset_config_id", ["resize256_v1"])[0],
            params.get("dataset_source", ["workspace"])[0],
        ))
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
        store = _workspace_store(handler)
        samples_path = store.ensure_workspace_training_samples(handler.workspace_id, handler.catalog.region.key) if store and handler.workspace_id else None
        handler._json(handler.catalog.list_training_samples() if samples_path is None else handler.catalog.list_training_samples(samples_path))
    elif path == "/api/training-patches":
        include = parse_qs(query_string).get("include", [""])[0]
        handler._json(_workspace_logical_payload(handler, include=include))
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
        if not _require_workspace(handler):
            return True
        patch_id = path.split("/")[-2]
        try:
            store = _workspace_store(handler)
            from lake_workbench.training.logical_patches import workspace_logical_patch_preview

            manifest = store.ensure_workspace_logical_patch_manifest(handler.workspace_id, handler.catalog.region.key)
            patch = handler.catalog.logical_patch_by_id(patch_id, manifest)
            overlay = truthy_flag(params.get("overlay", ["1"])[0], default=True)
            payload = workspace_logical_patch_preview(handler.catalog.region, patch, store, handler.workspace_id, overlay=overlay)
            content_type = "image/png"
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
        or _workspace_path(path)
    ) and not _require_workspace(handler):
        return True
    if re.fullmatch(r"/api/sites/[^/]+/training-samples", path):
        site_key = path.split("/")[-2]
        site = handler.catalog.get_site(site_key)
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
            return True
        payload = handler._read_json()
        try:
            store = _workspace_store(handler)
            samples_path = store.ensure_workspace_training_samples(handler.workspace_id, handler.catalog.region.key)
            label_dir = store.workspace_training_label_dir(handler.workspace_id, handler.catalog.region.key)
            result = handler.catalog.create_training_sample(site, payload, samples_path=samples_path, label_dir=label_dir)
        except ValueError as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
            return True
        patch_job = None
        store = _workspace_store(handler)
        if store:
            store.invalidate_training_sample_cache(handler.workspace_id, handler.catalog.region.key)
        manifest = store.ensure_workspace_logical_patch_manifest(handler.workspace_id, handler.catalog.region.key)
        workspace_owns_sample = bool(
            any(
                row.get("sample_id") == result.get("sample_id")
                for row in handler.catalog.list_logical_patches(manifest_path=manifest)["items"]
            )
        )
        if truthy_flag(payload.get("auto_patch"), default=True) and (
            result.get("action") != "updated_existing" or not workspace_owns_sample
        ):
            patch_job = handler.patch_exports.create(
                {
                    "sample_id": result["sample_id"],
                    "patch_size": 256,
                    "stride": 128,
                    "preview_scale": 2,
                    "overwrite": False,
                    "workspace_id": handler.workspace_id,
                }
            )
        handler._json({"sample": result, "patch_job": patch_job})
    elif path in {"/api/logical-patches/build-jobs", "/api/all/logical-patches/build-jobs", "/api/training-patches/export-jobs", "/api/all/training-patches/export-jobs"}:
        handler._json(handler.patch_exports.create({**handler._read_json(), "workspace_id": handler.workspace_id}))
    elif path in {"/api/global-dataset/patches", "/api/all/global-dataset/patches"}:
        payload = handler._read_json()
        target_scope = "all" if path.startswith("/api/all/") else handler.catalog.region.key
        try:
            groups = payload.get("source_groups")
            if isinstance(groups, dict):
                result = handler.__class__.dataset_registry.contribute_many(
                    _workspace_store(handler),
                    handler.workspace_id,
                    target_scope,
                    {str(region): [str(value) for value in values or []] for region, values in groups.items()},
                    replace=truthy_flag(payload.get("replace"), default=False),
                )
            else:
                result = handler.__class__.dataset_registry.contribute(
                    _workspace_store(handler),
                    handler.workspace_id,
                    target_scope,
                    str(payload.get("source_region") or handler.catalog.region.key),
                    [str(value) for value in payload.get("patch_ids") or []],
                    replace=truthy_flag(payload.get("replace"), default=False),
                )
        except DatasetConflict as exc:
            handler._error(HTTPStatus.CONFLICT, str(exc), {"conflicts": exc.conflicts})
        except (KeyError, ValueError) as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
        else:
            handler._json(result)
    elif path in {"/api/global-dataset/patches/withdraw", "/api/all/global-dataset/patches/withdraw"}:
        payload = handler._read_json()
        target_scope = "all" if path.startswith("/api/all/") else handler.catalog.region.key
        handler._json(
            handler.__class__.dataset_registry.withdraw(
                handler.workspace_id,
                target_scope,
                [str(value) for value in payload.get("patch_ids") or []],
            )
        )
    elif path in {"/api/global-dataset/build-jobs", "/api/all/global-dataset/build-jobs"}:
        payload = handler._read_json()
        target_scope = "all" if path.startswith("/api/all/") else handler.catalog.region.key
        handler._json(handler.__class__.global_dataset_builds.create({**payload, "scope": target_scope}))
    elif re.fullmatch(r"/api/(?:all/)?training-datasets/[^/]+/build-jobs", path):
        config_id = path.split("/")[-2]
        handler._json(handler.dataset_builds.create({**handler._read_json(), "config_id": config_id, "workspace_id": handler.workspace_id}))
    elif path in {"/api/training-runs", "/api/all/training-runs"}:
        options = handler._read_json()
        store = _workspace_store(handler)
        defaults = {key: value for key, value in options.items() if key != "run_name"}
        store.update_training_defaults(handler.workspace_id, defaults)
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
    if _workspace_path(path) and not _require_workspace(handler):
        return True
    if re.fullmatch(r"/api/training-samples/[^/]+", path):
        sample_id = path.rsplit("/", 1)[-1]
        try:
            store = _workspace_store(handler)
            samples_path = store.ensure_workspace_training_samples(handler.workspace_id, handler.catalog.region.key) if store and handler.workspace_id else None
            result = handler.catalog.update_training_sample(sample_id, handler._read_json(), samples_path)
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            store = _workspace_store(handler)
            if store:
                store.invalidate_training_sample_cache(handler.workspace_id, handler.catalog.region.key)
            handler._json({"sample": result})
    elif path == "/api/logical-patches":
        payload = handler._read_json()
        try:
            store = _workspace_store(handler)
            result = store.update_members(
                handler.workspace_id,
                handler.catalog.region.key,
                payload.get("logical_patch_ids") or [],
                str(payload.get("operation") or ""),
                replace=truthy_flag(payload.get("replace"), default=False),
            )
        except WorkspacePatchConflict as exc:
            handler._error(HTTPStatus.CONFLICT, str(exc), {"conflicts": exc.conflicts})
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
            store = _workspace_store(handler)
            value = payload.get("include") if "include" in payload else payload.get("included")
            store.update_members(handler.workspace_id, handler.catalog.region.key, [patch_id], "include" if truthy_flag(value, default=False) else "exclude", replace=truthy_flag(payload.get("replace"), default=False))
            result = _workspace_logical_payload(handler)["items"]
            result = next(item for item in result if item.get("logical_patch_id") == patch_id)
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            handler._json({"patch": result})
    else:
        return False
    return True


def handle_training_delete(handler, path: str) -> bool:
    if re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+", path):
        if not _require_workspace(handler):
            return True
        job_id = path.rsplit("/", 1)[-1]
        try:
            result = handler.training_runs.delete(job_id)
        except TrainingRunActiveError as exc:
            handler._error(HTTPStatus.CONFLICT, str(exc))
        except ValueError as exc:
            handler._error(HTTPStatus.BAD_REQUEST, str(exc))
        else:
            if result is None:
                handler._error(HTTPStatus.NOT_FOUND, "Training job not found")
            else:
                handler._json(result)
        return True

    if re.fullmatch(r"/api/training-samples/[^/]+", path):
        sample_id = path.rsplit("/", 1)[-1]
        try:
            store = _workspace_store(handler)
            samples_path = store.ensure_workspace_training_samples(handler.workspace_id, handler.catalog.region.key) if store and handler.workspace_id else None
            result = handler.catalog.delete_training_sample(sample_id, samples_path)
        except KeyError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            store = _workspace_store(handler)
            result["workspace_logical_patches_deleted"] = store.remove_workspace_logical_patches(sample_id)
            handler._json(result)
        return True

    return False
