"""User management routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.users import UserError
from lake_workbench.utils import truthy_flag


def _handle_error(handler, exc: Exception) -> None:
    status = HTTPStatus.NOT_FOUND if isinstance(exc, KeyError) else HTTPStatus.BAD_REQUEST
    handler._error(status, str(exc))


def _is_admin(handler) -> bool:
    return bool(getattr(handler, "current_user", {}).get("role") == "admin")


def _can_manage(handler, user_id: str) -> bool:
    return _is_admin(handler) or getattr(handler, "current_user", {}).get("id") == user_id


def handle_user_get(handler, path: str, query_string: str) -> bool:
    store = getattr(handler.__class__, "user_store", None)
    if store is None:
        return False
    try:
        if path == "/api/users":
            if not _is_admin(handler):
                handler._error(HTTPStatus.FORBIDDEN, "仅管理员可以列出用户")
                return True
            params = parse_qs(query_string)
            include_archived = truthy_flag(params.get("include_archived", [""])[0], default=False)
            handler._json(store.list(include_archived=include_archived))
        elif re.fullmatch(r"/api/users/[^/]+", path):
            user_id = path.rsplit("/", 1)[-1]
            if not _can_manage(handler, user_id):
                handler._error(HTTPStatus.FORBIDDEN, "无权访问该用户")
            else:
                handler._json({"user": store.get(user_id)})
        else:
            return False
    except (KeyError, UserError) as exc:
        _handle_error(handler, exc)
    return True


def handle_user_post(handler, path: str) -> bool:
    store = getattr(handler.__class__, "user_store", None)
    if store is None:
        return False
    try:
        if path == "/api/users":
            if not _is_admin(handler):
                handler._error(HTTPStatus.FORBIDDEN, "仅管理员可以创建用户")
                return True
            payload = handler._read_json()
            user = store.create(
                payload.get("name", ""),
                payload.get("workspace_mode", "empty"),
                payload.get("source_workspace_ids") or [],
            )
            handler._json({"user": user})
        elif re.fullmatch(r"/api/users/[^/]+/(?:archive|restore)", path):
            user_id, operation = path.split("/")[-2:]
            if not _is_admin(handler):
                handler._error(HTTPStatus.FORBIDDEN, "仅管理员可以归档用户")
                return True
            user = store.archive(user_id) if operation == "archive" else store.restore(user_id)
            handler._json({"user": user})
        else:
            return False
    except (KeyError, UserError, ValueError) as exc:
        _handle_error(handler, exc)
    return True


def handle_user_patch(handler, path: str) -> bool:
    store = getattr(handler.__class__, "user_store", None)
    if store is None or not re.fullmatch(r"/api/users/[^/]+", path):
        return False
    try:
        user_id = path.rsplit("/", 1)[-1]
        if not _can_manage(handler, user_id):
            handler._error(HTTPStatus.FORBIDDEN, "无权修改该用户")
            return True
        handler._json({"user": store.rename(user_id, handler._read_json().get("name", ""))})
    except (KeyError, UserError) as exc:
        _handle_error(handler, exc)
    return True
