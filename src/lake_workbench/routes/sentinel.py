"""Sentinel tile discovery, product search, and download routes."""

import re
from http import HTTPStatus
from urllib.parse import parse_qs

from lake_workbench.sentinel.download import query_copernicus_tile_products
from lake_workbench.utils import default_sentinel_date_range


def handle_sentinel_get(handler, path: str, query_string: str) -> bool:
    if re.fullmatch(r"/api/sites/[^/]+/sentinel/tiles", path):
        site_key = path.split("/")[-3]
        site = handler.catalog.get_site(site_key)
        if site is None:
            handler._error(HTTPStatus.NOT_FOUND, "Observation site not found")
        else:
            handler._json(handler.catalog.sentinel_tiles_for_site(site))
    elif path == "/api/sentinel/products":
        params = parse_qs(query_string)
        tile = params.get("tile", [""])[0]
        if not tile:
            handler._error(HTTPStatus.BAD_REQUEST, "tile is required")
            return True
        site = None
        site_key = params.get("site_id", [""])[0]
        if site_key:
            site = handler.catalog.get_site(site_key)
        default_start, default_end = default_sentinel_date_range()
        start = params.get("start", [default_start])[0]
        end = params.get("end", [default_end])[0]
        cloud = float(params.get("cloud", ["50"])[0])
        product_type = params.get("product_type", ["MSIL1C"])[0]
        limit = int(params.get("limit", ["50"])[0])
        products = query_copernicus_tile_products(tile, start, end, cloud, product_type, limit)
        products = handler.catalog.enrich_products_for_site(site, products)
        products = [
            {
                **product,
                **handler.catalog.local_product_status(product.get("product_id"), product.get("name")),
            }
            for product in products
        ]
        handler._json(
            {
                "tile": str(tile).upper().removeprefix("T"),
                "start": start,
                "end": end,
                "cloud": cloud,
                "product_type": product_type,
                "site_id": site.site_id if site else None,
                "products": products,
            }
        )
    elif re.fullmatch(r"/api/sentinel/downloads/[^/]+", path):
        job_id = path.rsplit("/", 1)[-1]
        job = handler.downloads.get(job_id)
        if job is None:
            handler._error(HTTPStatus.NOT_FOUND, "Download job not found")
        else:
            handler._json(job)
    else:
        return False
    return True


def handle_sentinel_post(handler, path: str) -> bool:
    if path != "/api/sentinel/downloads":
        return False
    payload = handler._read_json()
    product = payload.get("product") or payload
    if not product.get("product_id") or not product.get("name"):
        handler._error(HTTPStatus.BAD_REQUEST, "product_id and name are required")
        return True
    status = handler.catalog.local_product_status(product.get("product_id"), product.get("name"))
    if status.get("downloaded"):
        handler._json(
            {
                "job_id": None,
                "status": "completed",
                "message": "产品已在本地",
                "progress": 100,
                "result": status,
                "product": product,
            }
        )
    else:
        handler._json(handler.downloads.create(product))
    return True
