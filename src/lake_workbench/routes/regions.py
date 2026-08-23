"""Region discovery and cross-region observation-site routes."""

from urllib.parse import parse_qs


SITE_FILTER_KEYS = (
    "area_bucket",
    "has_tci",
    "has_name",
    "has_osm",
    "has_hydrolakes",
    "has_local_labels",
    "min_area",
    "max_area",
)


def site_list_options(query_string: str) -> tuple[str, int, int, dict[str, str]]:
    params = parse_qs(query_string)
    query = params.get("q", [""])[0]
    limit = int(params.get("limit", ["200"])[0])
    offset = int(params.get("offset", ["0"])[0])
    filters = {key: params.get(key, [""])[0] for key in SITE_FILTER_KEYS}
    return query, limit, offset, filters


def handle_region_get(handler, path: str, query_string: str) -> bool:
    if path == "/api/regions":
        handler._json(handler.__class__.region_service.regions_payload())
    elif path == "/api/all/sites":
        query, limit, offset, filters = site_list_options(query_string)
        handler._json(
            handler.__class__.region_service.all_sites_payload(
                query=query,
                limit=limit,
                offset=offset,
                filters=filters,
                workspace_store=getattr(handler.__class__, "workspace_store", None) if getattr(handler, "workspace_id", None) else None,
                workspace_id=getattr(handler, "workspace_id", None),
            )
        )
    else:
        return False
    return True
