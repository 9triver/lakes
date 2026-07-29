"""Model discovery and inference routes."""

import re
import threading
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.models.validation import ModelInferenceBusy
from lake_workbench.models.profiles import ProfileModelRegistry
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
    if getattr(handler, "profile_id", None) is None and (
        path in {"/api/model-validation/models", "/api/model-validation/random", "/api/all/model-validation/models", "/api/all/model-validation/random"}
        or re.fullmatch(r"/api/sites/[^/]+/model-prediction", path)
    ):
        handler._error(HTTPStatus.NOT_FOUND, "用户工作区路径必须包含用户")
        return True
    params = parse_qs(query_string)
    threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
    model_key = params.get("model", [""])[0]
    profile_store = getattr(handler.__class__, "profile_store", None)
    profile_registry = ProfileModelRegistry(profile_store, handler.__class__.catalogs, handler.profile_id) if profile_store else None
    visibility = params.get("visibility", ["current"])[0]
    if path == "/api/all/model-validation/models":
        handler._json(profile_registry.list("all", visibility) if profile_registry else handler.__class__.region_service.all_model_validation_models_payload())
    elif path == "/api/all/model-validation/random":
        if profile_registry:
            def run_all_profile_model():
                selected_key = model_key or profile_registry.list("all")["default"]
                path_value = profile_registry.resolve(selected_key, "all")
                if not path_value.exists():
                    raise FileNotFoundError(f"Model not found: {selected_key}")
                return handler.__class__.region_service.all_model_validation_random_loaded(load_model_checkpoint(path_value), threshold, selected_key)

            _run_inference(handler, run_all_profile_model)
        else:
            _run_inference(handler, lambda: handler.__class__.region_service.all_model_validation_random(threshold=threshold, model_key=model_key))
    elif path == "/api/model-validation/models":
        handler._json(profile_registry.list(handler.catalog.region.key, visibility) if profile_registry else handler.catalog.model_validation_models())
    elif path == "/api/model-validation/random":
        if profile_registry:
            def run_region_profile_model():
                selected_key = model_key or profile_registry.list(handler.catalog.region.key)["default"]
                path_value = profile_registry.resolve(selected_key, handler.catalog.region.key)
                if not path_value.exists():
                    raise FileNotFoundError(f"Model not found: {selected_key}")
                result = handler.catalog.model_validation_random(threshold=threshold, model=load_model_checkpoint(path_value))
                result["model"]["key"] = selected_key
                return result

            _run_inference(handler, run_region_profile_model)
        else:
            _run_inference(handler, lambda: handler.catalog.model_validation_random(threshold=threshold, model_key=model_key))
    elif re.fullmatch(r"/api/sites/[^/]+/model-prediction", path):
        site_key = path.split("/")[-2]
        site = handler.catalog.get_site(site_key)
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
        else:
            if profile_registry:
                def run_site_profile_model():
                    selected_key = model_key or profile_registry.list(handler.catalog.region.key)["default"]
                    path_value = profile_registry.resolve(selected_key, handler.catalog.region.key)
                    if not path_value.exists():
                        raise FileNotFoundError(f"Model not found: {selected_key}")
                    result = handler.catalog.model_prediction_for_site(site, threshold=threshold, model=load_model_checkpoint(path_value))
                    result["model"]["key"] = selected_key
                    return result

                _run_inference(handler, run_site_profile_model)
            else:
                _run_inference(handler, lambda: handler.catalog.model_prediction_for_site(site, threshold=threshold, model_key=model_key))
    else:
        return False
    return True
