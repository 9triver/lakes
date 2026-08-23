"""Model discovery and inference routes."""

import re
import threading
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.models.validation import ModelInferenceBusy
from lake_workbench.models.workspaces import WorkspaceModelRegistry
from lake_workbench.models.runtime import load_model_checkpoint
from lake_workbench.utils import parse_float_or_default


MODEL_VALIDATION_SEMAPHORE = threading.BoundedSemaphore(1)


def _run_inference(handler, callback) -> None:
    if not MODEL_VALIDATION_SEMAPHORE.acquire(blocking=False):
        handler._error(HTTPStatus.TOO_MANY_REQUESTS, "模型推理正在运行，请稍后再试")
        return
    try:
        handler._json(callback())
    except ModelInferenceBusy as exc:
        handler._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
    except (FileNotFoundError, ValueError) as exc:
        handler._error(HTTPStatus.NOT_FOUND, str(exc))
    finally:
        MODEL_VALIDATION_SEMAPHORE.release()


def handle_model_get(handler, path: str, query_string: str) -> bool:
    if getattr(handler, "workspace_id", None) is None and (
        path in {"/api/model-validation/models", "/api/model-validation/random", "/api/all/model-validation/models", "/api/all/model-validation/random"}
        or re.fullmatch(r"/api/sites/[^/]+/model-prediction", path)
    ):
        handler._error(HTTPStatus.NOT_FOUND, "模型资源路径必须包含 workspace")
        return True
    params = parse_qs(query_string)
    threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
    model_key = params.get("model", [""])[0]
    workspace_store = getattr(handler.__class__, "workspace_store", None)
    allow_foreign = getattr(handler, "current_user", {}).get("role") == "admin"
    workspace_registry = WorkspaceModelRegistry(
        workspace_store,
        handler.__class__.catalogs,
        handler.workspace_id,
        allow_foreign=allow_foreign,
    )
    visibility = params.get("visibility", ["current"])[0]
    if path == "/api/all/model-validation/models":
        handler._json(workspace_registry.list("all", visibility))
    elif path == "/api/all/model-validation/random":
        def run_all_workspace_model():
            selected_key = model_key or workspace_registry.list("all")["default"]
            path_value = workspace_registry.resolve(selected_key, "all")
            if not path_value.exists():
                raise FileNotFoundError(f"Model not found: {selected_key}")
            return handler.__class__.region_service.all_model_validation_random_loaded(load_model_checkpoint(path_value), threshold, selected_key)

        _run_inference(handler, run_all_workspace_model)
    elif path == "/api/model-validation/models":
        handler._json(workspace_registry.list(handler.catalog.region.key, visibility))
    elif path == "/api/model-validation/random":
        def run_region_workspace_model():
            selected_key = model_key or workspace_registry.list(handler.catalog.region.key)["default"]
            path_value = workspace_registry.resolve(selected_key, handler.catalog.region.key)
            if not path_value.exists():
                raise FileNotFoundError(f"Model not found: {selected_key}")
            result = handler.catalog.model_validation_random(threshold=threshold, model=load_model_checkpoint(path_value))
            result["model"]["key"] = selected_key
            return result

        _run_inference(handler, run_region_workspace_model)
    elif re.fullmatch(r"/api/sites/[^/]+/model-prediction", path):
        site_key = path.split("/")[-2]
        site = handler.catalog.get_site(site_key)
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
        else:
            def run_site_workspace_model():
                selected_key = model_key or workspace_registry.list(handler.catalog.region.key)["default"]
                path_value = workspace_registry.resolve(selected_key, handler.catalog.region.key)
                if not path_value.exists():
                    raise FileNotFoundError(f"Model not found: {selected_key}")
                result = handler.catalog.model_prediction_for_site(site, threshold=threshold, model=load_model_checkpoint(path_value))
                result["model"]["key"] = selected_key
                return result

            _run_inference(handler, run_site_workspace_model)
    else:
        return False
    return True
