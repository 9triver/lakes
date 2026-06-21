#!/usr/bin/env python3
"""Pre-generate ESA WorldCover water polygons for large lakes."""

from __future__ import annotations

import argparse
from lakes_browser.server import (
    LakeCatalog,
    build_esa_smoothed_layer,
    write_esa_polygon_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", action="append", default=None, help="Lake id/key to precompute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lakes = args.lake or ["hn_osm_1462005"]
    catalog = LakeCatalog()
    for lake_key in lakes:
        lake = catalog.get_lake(lake_key)
        if lake is None:
            print(f"missing lake: {lake_key}")
            continue
        print(f"lake {lake.object_id} {lake.name} area={lake.area_km2:.2f} km2")
        layer = build_esa_smoothed_layer(lake)
        if layer is None:
            print("  failed")
            continue
        layer["properties"]["pre_generated"] = True
        path = write_esa_polygon_cache(lake.object_id, layer)
        status = "empty" if layer.get("geometry") is None else "ok"
        print(f"  {status}: {path}")


if __name__ == "__main__":
    main()
