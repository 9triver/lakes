"""Training Profile management routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.profiles import ProfileError
from lake_workbench.utils import truthy_flag


def _handle_error(handler, exc: Exception) -> None:
    status = HTTPStatus.NOT_FOUND if isinstance(exc, KeyError) else HTTPStatus.BAD_REQUEST
    handler._error(status, str(exc))


def handle_profile_get(handler, path: str, query_string: str) -> bool:
    store = getattr(handler.__class__, "profile_store", None)
    if store is None:
        return False
    try:
        if path == "/api/profiles":
            params = parse_qs(query_string)
            include_archived = truthy_flag(params.get("include_archived", [""])[0], default=False)
            handler._json(store.list(include_archived=include_archived))
        elif re.fullmatch(r"/api/profiles/[^/]+", path):
            handler._json({"profile": store.get(path.rsplit("/", 1)[-1])})
        elif re.fullmatch(r"/api/profiles/[^/]+/source-conflicts", path):
            handler._json(store.conflicts(path.split("/")[-2]))
        else:
            return False
    except (KeyError, ProfileError) as exc:
        _handle_error(handler, exc)
    return True


def handle_profile_post(handler, path: str) -> bool:
    store = getattr(handler.__class__, "profile_store", None)
    if store is None:
        return False
    try:
        if path == "/api/profiles":
            payload = handler._read_json()
            profile = store.create(
                payload.get("name", ""),
                payload.get("mode", "empty"),
                payload.get("source_profile_ids") or [],
            )
            handler._json({"profile": profile})
        elif re.fullmatch(r"/api/profiles/[^/]+/(?:archive|restore)", path):
            profile_id, operation = path.split("/")[-2:]
            profile = store.archive(profile_id) if operation == "archive" else store.restore(profile_id)
            handler._json({"profile": profile})
        else:
            return False
    except (KeyError, ProfileError) as exc:
        _handle_error(handler, exc)
    return True


def handle_profile_patch(handler, path: str) -> bool:
    store = getattr(handler.__class__, "profile_store", None)
    if store is None:
        return False
    try:
        if re.fullmatch(r"/api/profiles/[^/]+", path):
            profile_id = path.rsplit("/", 1)[-1]
            handler._json({"profile": store.rename(profile_id, handler._read_json().get("name", ""))})
        elif re.fullmatch(r"/api/profiles/[^/]+/sites/[^/]+/source-variants", path):
            parts = path.split("/")
            profile_id, site_id = parts[3], parts[5]
            result = store.resolve_sources(profile_id, site_id, handler._read_json().get("variant_ids") or [])
            handler._json(result)
        else:
            return False
    except (KeyError, ProfileError) as exc:
        _handle_error(handler, exc)
    return True
