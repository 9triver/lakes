#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query and download Sentinel-2 products for the Lakes browser."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lakes_browser.sentinel_download import (
    download_copernicus_product,
    product_date,
    product_tile_name,
    product_type_from_name,
    query_copernicus_tile_products,
    upsert_csv_row,
    valid_ratio_for_tci,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = DATA_DIR / "hunan_single_tiles" / "products"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "processed" / "sentinel_products.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="query Copernicus products by Sentinel tile")
    query.add_argument("--tile", required=True, help="MGRS tile, e.g. 49RFN or T49RFN")
    query.add_argument("--start", default="2025-06-01", help="start date, YYYY-MM-DD")
    query.add_argument("--end", default="2025-08-31", help="end date, YYYY-MM-DD")
    query.add_argument("--cloud", type=float, default=20, help="maximum cloud cover percentage")
    query.add_argument("--product-type", default="MSIL1C", help="product name filter, default MSIL1C")
    query.add_argument("--limit", type=int, default=50, help="maximum result count")
    query.add_argument("--json", action="store_true", help="print raw JSON results")

    download = subparsers.add_parser("download", help="download one Copernicus product")
    download.add_argument("--product-id", required=True, help="Copernicus product UUID")
    download.add_argument("--name", required=True, help="product SAFE name")
    download.add_argument("--cloud-cover", default="", help="cloud cover value to record in local index")
    download.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="SAFE output directory")
    download.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="local downloaded product CSV index")
    download.add_argument("--env", type=Path, default=None, help="optional .env file with Copernicus credentials")

    args = parser.parse_args()
    if args.command == "query":
        products = query_copernicus_tile_products(
            tile=args.tile,
            start=args.start,
            end=args.end,
            cloud=args.cloud,
            product_type=args.product_type,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(products, ensure_ascii=False, indent=2))
        else:
            print_products(products)
    elif args.command == "download":
        product = {
            "product_id": args.product_id,
            "name": args.name,
            "cloud_cover": args.cloud_cover,
        }

        last_pct = -1

        def progress(done: int, total: int) -> None:
            nonlocal last_pct
            pct = int(done * 100 / total) if total else 0
            if pct != last_pct:
                print(f"downloading {pct}% ({done}/{total or '?'})", flush=True)
                last_pct = pct

        safe_dir, tci_path = download_copernicus_product(product, args.out_dir, progress=progress, env_path=args.env)
        row = {
            "product_id": args.product_id,
            "product_name": args.name,
            "tile": product_tile_name(args.name),
            "date": product_date(args.name),
            "cloud_cover": args.cloud_cover,
            "product_type": product_type_from_name(args.name),
            "source": "script_download",
            "safe_path": display_path(safe_dir),
            "tci_path": display_path(tci_path),
            "download_status": "downloaded",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "valid_ratio": valid_ratio_for_tci(tci_path),
        }
        upsert_csv_row(args.index, row, key="product_name")
        print(json.dumps(row, ensure_ascii=False, indent=2))


def print_products(products: list[dict]) -> None:
    if not products:
        print("No products found.")
        return
    for product in products:
        cloud = product.get("cloud_cover")
        cloud_text = "unknown" if cloud is None else f"{float(cloud):.1f}%"
        size = product.get("content_length")
        size_text = human_size(size) if size else "unknown size"
        print(
            "\t".join(
                [
                    str(product.get("date") or ""),
                    str(product.get("tile") or ""),
                    cloud_text,
                    size_text,
                    str(product.get("product_id") or ""),
                    str(product.get("name") or ""),
                ]
            )
        )


def human_size(value) -> str:
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
