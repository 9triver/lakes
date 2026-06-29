#!/usr/bin/env python3
"""Pre-generate ESA WorldCover water polygons for large lakes."""

from __future__ import annotations

import argparse
from lakes_browser.server import (
    REGIONS,
    DEFAULT_REGION_KEY,
    LakeCatalog,
    build_esa_smoothed_layer,
    write_esa_polygon_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--lake", action="append", default=None, help="Lake id/key to precompute.")
    parser.add_argument("--all", action="store_true", help="Precompute every lake in the selected region.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = LakeCatalog(REGIONS[args.region])
    lakes = [lake.object_id for lake in catalog.lakes] if args.all else args.lake or ([catalog.lakes[0].object_id] if catalog.lakes else [])
    for lake_key in lakes:
        lake = catalog.get_lake(lake_key)
        if lake is None:
            print(f"missing lake: {lake_key}")
            continue
        print(f"lake {lake.object_id} {lake.name} area={lake.area_km2:.2f} km2")
        layer = build_esa_smoothed_layer(catalog.region, lake)
        if layer is None:
            print("  failed")
            continue
        layer["properties"]["pre_generated"] = True
        path = write_esa_polygon_cache(catalog.region, lake.object_id, layer)
        status = "empty" if layer.get("geometry") is None else "ok"
        print(f"  {status}: {path}")


if __name__ == "__main__":
    main()
