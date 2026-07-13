"""Lake metadata, imagery, labels, and water-layer routes."""

import json
import os
import re
import threading
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.imagery import blank_png
from lake_workbench.routes.regions import lake_list_options


TILE_RENDER_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.environ.get("LAKES_TILE_RENDER_WORKERS", "1"))))


def _lake(handler, lake_key: str):
    lake = handler.catalog.get_lake(lake_key)
    if lake is None:
        handler._error(HTTPStatus.NOT_FOUND, "Lake not found")
    return lake


def handle_lake_get(handler, path: str, query_string: str) -> bool:
    params = parse_qs(query_string)
    if path == "/api/lakes":
        query, limit, offset, filters = lake_list_options(query_string)
        handler._json(handler.catalog.list_lakes(query=query, limit=limit, offset=offset, filters=filters))
    elif re.fullmatch(r"/api/lakes/[^/]+", path):
        lake = _lake(handler, path.rsplit("/", 1)[-1])
        if lake is not None:
            handler._json(handler.catalog.get_lake_detail(lake))
    elif re.fullmatch(r"/api/lakes/[^/]+/image\.png", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            size = int(params.get("size", ["900"])[0])
            padding = float(params.get("padding", ["0.6"])[0])
            try:
                payload, meta = handler.catalog.image_for_lake(lake, size=size, padding=padding)
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
            else:
                handler._send_bytes(
                    payload,
                    "image/png",
                    headers={"X-Image-Meta": json.dumps(meta, ensure_ascii=True)},
                )
    elif re.fullmatch(r"/api/lakes/[^/]+/tile-meta", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            try:
                handler._json(handler.catalog.tile_meta_for_lake(lake, padding=float(params.get("padding", ["0.8"])[0])))
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
    elif re.fullmatch(r"/api/lakes/[^/]+/tiles/\d+/\d+/\d+\.png", path):
        match = re.fullmatch(r"/api/lakes/([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.png", path)
        lake = _lake(handler, match.group(1))
        if lake is not None:
            try:
                with TILE_RENDER_SEMAPHORE:
                    payload, _meta = handler.catalog.tile_png_for_lake(
                        lake,
                        z=int(match.group(2)),
                        x=int(match.group(3)),
                        y=int(match.group(4)),
                        padding=float(params.get("padding", ["0.8"])[0]),
                    )
            except FileNotFoundError:
                payload = blank_png(256)
            handler._send_bytes(payload, "image/png", cache_control="public, max-age=600")
    elif re.fullmatch(r"/api/lakes/[^/]+/esa", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            handler._json({"esa": handler.catalog._esa_smoothed_layer(lake)})
    elif re.fullmatch(r"/api/lakes/[^/]+/jrc", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            threshold = int(params.get("threshold", ["75"])[0])
            handler._json({"jrc": handler.catalog._jrc_occurrence_layer(lake, threshold=threshold)})
    elif re.fullmatch(r"/api/lakes/[^/]+/context-water", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            handler._json(
                handler.catalog.context_water_for_lake(
                    lake,
                    padding=float(params.get("padding", ["0.8"])[0]),
                    min_area_km2=float(params.get("min_area_km2", ["1.0"])[0]),
                    limit=int(params.get("limit", ["500"])[0]),
                )
            )
    elif re.fullmatch(r"/api/lakes/[^/]+/local-labels", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            handler._json(handler.catalog.local_label_items(lake))
    elif re.fullmatch(r"/api/lakes/[^/]+/local-labels/[^/]+", path):
        parts = path.split("/")
        lake = _lake(handler, parts[-3])
        if lake is not None:
            try:
                handler._json(handler.catalog.local_label_geojson(lake, parts[-1]))
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
    elif re.fullmatch(r"/api/lakes/[^/]+/imagery", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            handler._json(handler.catalog.imagery_for_lake(lake))
    elif re.fullmatch(r"/api/lakes/[^/]+/image-meta", path):
        lake = _lake(handler, path.split("/")[-2])
        if lake is not None:
            try:
                _, meta = handler.catalog.image_for_lake(lake)
                handler._json(meta)
            except FileNotFoundError as exc:
                handler._error(HTTPStatus.NOT_FOUND, str(exc))
    else:
        return False
    return True


def handle_lake_post(handler, path: str) -> bool:
    if not re.fullmatch(r"/api/lakes/[^/]+/imagery/active", path):
        return False
    lake = _lake(handler, path.split("/")[-3])
    if lake is None:
        return True
    payload = handler._read_json()
    tile = payload.get("tile")
    product = payload.get("product")
    if not tile or not product:
        handler._error(HTTPStatus.BAD_REQUEST, "tile and product are required")
        return True
    try:
        result = handler.catalog.set_active_imagery(tile, product, lake)
    except KeyError as exc:
        handler._error(HTTPStatus.NOT_FOUND, str(exc))
    else:
        handler._json(result)
    return True
