"""Region discovery and cross-region lake routes."""

from urllib.parse import parse_qs


LAKE_FILTER_KEYS = (
    "water_type",
    "province",
    "city",
    "county",
    "polygon_quality",
    "metadata_quality",
    "area_bucket",
    "has_tci",
    "has_name",
    "min_area",
    "max_area",
)


def lake_list_options(query_string: str) -> tuple[str, int, int, dict[str, str]]:
    params = parse_qs(query_string)
    query = params.get("q", [""])[0]
    limit = int(params.get("limit", ["200"])[0])
    offset = int(params.get("offset", ["0"])[0])
    filters = {key: params.get(key, [""])[0] for key in LAKE_FILTER_KEYS}
    return query, limit, offset, filters


def handle_region_get(handler, path: str, query_string: str) -> bool:
    if path == "/api/regions":
        handler._json(handler.__class__.region_service.regions_payload())
    elif path == "/api/all/lakes":
        query, limit, offset, filters = lake_list_options(query_string)
        handler._json(
            handler.__class__.region_service.all_lakes_payload(
                query=query,
                limit=limit,
                offset=offset,
                filters=filters,
            )
        )
    else:
        return False
    return True
