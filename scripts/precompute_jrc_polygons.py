#!/usr/bin/env python3
"""Pre-generate JRC occurrence polygons for observation sites."""

from __future__ import annotations

import argparse

from lake_workbench.catalog import SiteCatalog
from lake_workbench.regions.config import load_region_configs
from lake_workbench.water.layers import build_jrc_occurrence_layer, write_jrc_polygon_cache


REGIONS, DEFAULT_REGION_KEY = load_region_configs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--site", action="append", default=None, help="Observation site id/key to precompute.")
    parser.add_argument("--lake", action="append", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--all", action="store_true", help="Precompute every site in the selected region.")
    parser.add_argument("--thresholds", default="50,75,90", help="Comma-separated occurrence thresholds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [int(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    catalog = SiteCatalog(REGIONS[args.region], DEFAULT_REGION_KEY)
    requested = args.site or args.lake
    site_ids = [site.site_id for site in catalog.sites] if args.all else requested or ([catalog.sites[0].site_id] if catalog.sites else [])
    for site_id in site_ids:
        site = catalog.get_site(site_id)
        if site is None:
            print(f"missing site: {site_id}")
            continue
        print(f"site {site.site_id} {site.display_name} coverage={site.area_km2:.2f} km2")
        for threshold in thresholds:
            print(f"  threshold {threshold}: generating")
            layer = build_jrc_occurrence_layer(catalog.region, site, threshold)
            if layer is None:
                print("    failed")
                continue
            layer["properties"]["pre_generated"] = True
            path = write_jrc_polygon_cache(catalog.region, site.site_id, threshold, layer)
            status = "empty" if layer.get("geometry") is None else "ok"
            print(f"    {status}: {path}")


if __name__ == "__main__":
    main()
