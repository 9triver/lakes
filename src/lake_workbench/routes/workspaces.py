"""Training Workspace management routes."""

import re
from http import HTTPStatus
from lake_workbench.workspaces import WorkspaceError


def _handle_error(handler, exc: Exception) -> None:
    status = HTTPStatus.NOT_FOUND if isinstance(exc, KeyError) else HTTPStatus.BAD_REQUEST
    handler._error(status, str(exc))


def handle_workspace_get(handler, path: str, query_string: str) -> bool:
    store = getattr(handler.__class__, "workspace_store", None)
    if store is None:
        return False
    try:
        if re.fullmatch(r"/api/workspaces/[^/]+", path):
            handler._json({"workspace": store.get(path.rsplit("/", 1)[-1])})
        elif re.fullmatch(r"/api/workspaces/[^/]+/source-conflicts", path):
            handler._json(store.conflicts(path.split("/")[-2]))
        else:
            return False
    except (KeyError, WorkspaceError) as exc:
        _handle_error(handler, exc)
    return True


def handle_workspace_patch(handler, path: str) -> bool:
    store = getattr(handler.__class__, "workspace_store", None)
    if store is None:
        return False
    try:
        if re.fullmatch(r"/api/workspaces/[^/]+/sites/[^/]+/source-variants", path):
            parts = path.split("/")
            workspace_id, site_id = parts[3], parts[5]
            result = store.resolve_sources(workspace_id, site_id, handler._read_json().get("variant_ids") or [])
            handler._json(result)
        else:
            return False
    except (KeyError, WorkspaceError) as exc:
        _handle_error(handler, exc)
    return True
