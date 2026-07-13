"""Model discovery and inference routes."""

import re
import threading
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.models.validation import ModelInferenceBusy
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
    params = parse_qs(query_string)
    threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
    model_key = params.get("model", [""])[0]
    if path == "/api/all/model-validation/models":
        handler._json(handler.__class__.region_service.all_model_validation_models_payload())
    elif path == "/api/all/model-validation/random":
        _run_inference(
            handler,
            lambda: handler.__class__.region_service.all_model_validation_random(
                threshold=threshold,
                model_key=model_key,
            ),
        )
    elif path == "/api/model-validation/models":
        handler._json(handler.catalog.model_validation_models())
    elif path == "/api/model-validation/random":
        _run_inference(
            handler,
            lambda: handler.catalog.model_validation_random(
                threshold=threshold,
                model_key=model_key,
            ),
        )
    elif re.fullmatch(r"/api/lakes/[^/]+/model-prediction", path):
        lake_key = path.split("/")[-2]
        lake = handler.catalog.get_lake(lake_key)
        if lake is None:
            handler._error(HTTPStatus.NOT_FOUND, "Lake not found")
        else:
            _run_inference(
                handler,
                lambda: handler.catalog.model_prediction_for_lake(
                    lake,
                    threshold=threshold,
                    model_key=model_key,
                ),
            )
    else:
        return False
    return True
