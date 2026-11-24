"""Model discovery and inference routes."""

import json
import re
import threading
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs

from lake_workbench.models.validation import ModelInferenceBusy
from lake_workbench.models.workspaces import WorkspaceModelRegistry
from lake_workbench.models.runtime import load_model_checkpoint
from lake_workbench.utils import parse_float_or_default
from lake_workbench.models.comparison import comparison_context, comparison_preview, compare_one, saved_comparisons, iter_saved_comparisons
from lake_workbench.models.preparation import comparison_sites, iter_comparison_sites, prepare
from lake_workbench.paths import PROJECT_ROOT


MODEL_VALIDATION_SEMAPHORE = threading.BoundedSemaphore(1)


def _run_inference(handler, callback, *, wait: bool = False, serialize: bool = True) -> None:
    acquired = False
    if serialize:
        acquired=MODEL_VALIDATION_SEMAPHORE.acquire(timeout=120) if wait else MODEL_VALIDATION_SEMAPHORE.acquire(blocking=False)
        if not acquired:
            handler._error(HTTPStatus.TOO_MANY_REQUESTS, "模型推理正在运行，请稍后再试")
            return
    try:
        handler._json(callback())
    except ModelInferenceBusy as exc:
        handler._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
    except (FileNotFoundError, ValueError) as exc:
        handler._error(HTTPStatus.NOT_FOUND, str(exc))
    finally:
        if acquired:
            MODEL_VALIDATION_SEMAPHORE.release()


def _stream_events(handler, events):
    handler.send_response(HTTPStatus.OK)
    handler.send_header('Content-Type','application/x-ndjson; charset=utf-8')
    handler.send_header('Cache-Control','no-store')
    handler.send_header('X-Accel-Buffering','no')
    handler.send_header('Connection','close')
    handler.end_headers();handler.close_connection=True
    def emit(event):
        handler.wfile.write((json.dumps(event,ensure_ascii=False)+'\n').encode('utf-8'))
        handler.wfile.flush()
    try:
        for event in events:emit(event)
    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
        return
    except Exception as error:
        emit(dict(type='error',message=str(error)))


def _stream_saved_results(handler, registry, site, asset, reference, threshold, threshold_strategy):
    def events():
        yield dict(type='progress',stage='models',message='正在读取模型目录',completed=0,total=None,loaded=0)
        available=registry.list('all')['items']
        paths=[(item['key'],PROJECT_ROOT / Path(item['path'])) for item in available if not item.get('error')]
        yield dict(type='progress',stage='context',message='正在核对共同影像、参考和阈值',completed=0,total=None,loaded=0)
        yield from iter_saved_comparisons(handler.catalog,site,paths,asset,reference,threshold,threshold_strategy)
    _stream_events(handler,events())


def handle_model_get(handler, path: str, query_string: str) -> bool:
    if getattr(handler, "workspace_id", None) is None and (
        path in {"/api/model-comparison/sites", "/api/model-validation/models", "/api/model-validation/random", "/api/all/model-validation/models", "/api/all/model-validation/random", "/api/all/model-validation/random-target"}
        or re.fullmatch(r"/api/sites/[^/]+/(?:model-prediction|model-comparison(?:/(?:context|saved|preparation|preview\.png))?)", path)
    ):
        handler._error(HTTPStatus.NOT_FOUND, "模型资源路径必须包含 workspace")
        return True
    params = parse_qs(query_string)
    threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
    threshold_strategy=params.get("threshold_strategy",["manual"])[0]
    strategy_options={"threshold_strategy":threshold_strategy} if threshold_strategy!="manual" else {}
    model_key = params.get("model", [""])[0]
    workspace_store = getattr(handler.__class__, "workspace_store", None)
    allow_foreign = (getattr(handler, "current_user", None) or {}).get("role") == "admin"
    workspace_registry = WorkspaceModelRegistry(
        workspace_store,
        handler.__class__.catalogs,
        handler.workspace_id,
        allow_foreign=allow_foreign,
    )
    visibility = params.get("visibility", ["current"])[0]
    if path == '/api/model-comparison/sites' or re.fullmatch(r'/api/sites/[^/]+/model-comparison/preparation',path):
        derived=workspace_store.workspace_derived_label_dir(handler.workspace_id,handler.catalog.region.key)
        if path == '/api/model-comparison/sites':
            if params.get('stream', ['0'])[0] == '1':
                _stream_events(handler, iter_comparison_sites(handler.catalog, handler.workspace_id, derived))
            else:
                handler._json(comparison_sites(handler.catalog, handler.workspace_id, derived))
        else:
            site=handler.catalog.get_site(path.split('/')[3])
            if site is None:handler._error(HTTPStatus.NOT_FOUND,'Observation site not found')
            else:handler._json(prepare(handler.catalog,site,handler.workspace_id,derived,
                params.get('asset',[''])[0],params.get('reference',['spectral_osm_intersection'])[0],params.get('retry',['0'])[0]))
        return True
    comparison_match = re.fullmatch(r'/api/sites/([^/]+)/model-comparison(/context|/saved|/preview\.png)?',path)
    if comparison_match:
        site=handler.catalog.get_site(comparison_match.group(1))
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND,'Observation site not found')
            return True
        asset=params.get('asset',[''])[0]
        reference=params.get('reference',['spectral_osm_intersection'])[0]
        if comparison_match.group(2)=='/preview.png':
            try:handler._send_bytes(comparison_preview(handler.catalog,site,asset,params.get('style',['rgb'])[0]),'image/png',cache_control='private, max-age=600')
            except (ValueError,FileNotFoundError) as exc:handler._error(HTTPStatus.BAD_REQUEST,str(exc))
        elif comparison_match.group(2)=='/context':
            handler._json(comparison_context(handler.catalog,site,asset,reference))
        elif comparison_match.group(2)=='/saved':
            if params.get('stream', ['0'])[0] == '1':
                _stream_saved_results(handler, workspace_registry, site, asset, reference, threshold, threshold_strategy)
                return True
            available=workspace_registry.list('all')['items']
            # This snapshot already contains authorized, server-generated paths.
            # Resolving each key again rescans the entire experiment tree per model.
            paths=[(item['key'],PROJECT_ROOT / Path(item['path'])) for item in available if not item.get('error')]
            handler._json(saved_comparisons(handler.catalog,site,paths,asset,reference,threshold,**strategy_options))
        else:
            def compare_site_model():
                selected=workspace_registry.resolve(model_key,handler.catalog.region.key)
                return compare_one(handler.catalog,site,selected,model_key,asset,reference,threshold,**strategy_options)
            _run_inference(handler,compare_site_model,wait=True)
    elif path == "/api/all/model-validation/models":
        handler._json(workspace_registry.list("all", visibility))
    elif path == "/api/all/model-validation/random-target":
        try:
            handler._json(handler.__class__.region_service.random_model_validation_target())
        except FileNotFoundError as exc:
            handler._error(HTTPStatus.NOT_FOUND, str(exc))
    elif path == "/api/all/model-validation/random":
        def run_all_workspace_model():
            selected_key = model_key or workspace_registry.list("all")["default"]
            path_value = workspace_registry.resolve(selected_key, "all")
            if not path_value.exists():
                raise FileNotFoundError(f"Model not found: {selected_key}")
            return handler.__class__.region_service.all_model_validation_random_loaded(load_model_checkpoint(path_value), threshold, selected_key,**strategy_options)

        _run_inference(handler, run_all_workspace_model, serialize=False)
    elif path == "/api/model-validation/models":
        handler._json(workspace_registry.list(handler.catalog.region.key, visibility))
    elif path == "/api/model-validation/random":
        def run_region_workspace_model():
            selected_key = model_key or workspace_registry.list(handler.catalog.region.key)["default"]
            path_value = workspace_registry.resolve(selected_key, handler.catalog.region.key)
            if not path_value.exists():
                raise FileNotFoundError(f"Model not found: {selected_key}")
            result = handler.catalog.model_validation_random(threshold=threshold, model=load_model_checkpoint(path_value),**strategy_options)
            result["model"]["key"] = selected_key
            return result

        _run_inference(handler, run_region_workspace_model, serialize=False)
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
                result = handler.catalog.model_prediction_for_site(site, threshold=threshold, model=load_model_checkpoint(path_value),**strategy_options)
                result["model"]["key"] = selected_key
                return result

            _run_inference(handler, run_site_workspace_model, serialize=False)
    else:
        return False
    return True
