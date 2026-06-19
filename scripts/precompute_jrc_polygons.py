#!/usr/bin/env python3
"""Pre-generate JRC occurrence polygons for large lakes."""

from __future__ import annotations

import argparse
from lakes_browser.server import (
    LakeCatalog,
    build_jrc_occurrence_layer,
    write_jrc_polygon_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", action="append", default=None, help="Lake id/key to precompute.")
    parser.add_argument("--thresholds", default="50,75,90", help="Comma-separated occurrence thresholds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lakes = args.lake or ["hn_osm_1462005"]
    thresholds = [int(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    catalog = LakeCatalog()
    for lake_key in lakes:
        lake = catalog.get_lake(lake_key)
        if lake is None:
            print(f"missing lake: {lake_key}")
            continue
        print(f"lake {lake.object_id} {lake.name} area={lake.area_km2:.2f} km2")
        for threshold in thresholds:
            print(f"  threshold {threshold}: generating")
            layer = build_jrc_occurrence_layer(lake, threshold)
            if layer is None:
                print("    failed")
                continue
            layer["properties"]["pre_generated"] = True
            path = write_jrc_polygon_cache(lake.object_id, threshold, layer)
            status = "empty" if layer.get("geometry") is None else "ok"
            print(f"    {status}: {path}")


if __name__ == "__main__":
    main()
