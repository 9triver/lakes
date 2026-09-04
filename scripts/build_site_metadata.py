#!/usr/bin/env python3
"""Build observation-site metadata from local imagery directories."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lake_workbench.metadata.geometry import (  # noqa: E402
    geo_frame,
    geometry_area_km2,
    majority_coverage_geometry,
    polygonal_geometry,
)
from lake_workbench.metadata.imagery import (  # noqa: E402
    scan_site_imagery,
    sentinel_tiles_for_geometry,
)
from lake_workbench.metadata.candidates import (  # noqa: E402
    build_vector_candidates,
    mark_suggested_candidate,
    site_display_name,
    suggested_site_name,
)
from lake_workbench.metadata.labels import (  # noqa: E402
    load_local_label_features,
)
from lake_workbench.metadata.writer import BuildResult, write_result  # noqa: E402
from lake_workbench.regions.config import RegionConfig, load_region_configs  # noqa: E402
from lake_workbench.utils import display_region_path  # noqa: E402
from lake_workbench.water.layers import (  # noqa: E402
    build_esa_smoothed_layer,
    build_jrc_occurrence_layer,
    write_esa_polygon_cache,
    write_jrc_polygon_cache,
)

from lake_workbench.metadata.sources import (  # noqa: E402
    load_external_hydrolakes,
    load_external_osm_water,
    load_sentinel_tile_index,
)


REGIONS, DEFAULT_REGION_KEY = load_region_configs()
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--footprint-size", type=int, default=256)
    parser.add_argument("--jrc-threshold", type=int, default=75)
    parser.add_argument("--skip-raster-water", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers for local imagery scanning")
    parser.add_argument(
        "--reuse-label-cache",
        action="store_true",
        help="reuse a completed temporary local-label GPKG from an interrupted build",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = REGIONS[args.region]
    output_dir = args.output_dir.resolve() if args.output_dir else region.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_site_metadata(
        region,
        footprint_size=max(64, min(1024, args.footprint_size)),
        jrc_threshold=max(1, min(100, args.jrc_threshold)),
        include_raster_water=not args.skip_raster_water,
        workers=max(1, min(16, args.workers)),
        label_output=output_dir / ".local_labels.building.gpkg",
        reuse_label_cache=args.reuse_label_cache,
    )
    write_result(region, output_dir, result)


def build_site_metadata(
    region: RegionConfig,
    footprint_size: int = 256,
    jrc_threshold: int = 75,
    include_raster_water: bool = True,
    workers: int = 4,
    label_output: Path | None = None,
    reuse_label_cache: bool = False,
) -> BuildResult:
    image_root = region.local_imagery_root
    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"local imagery root not found for {region.key}: {image_root}")
    sentinel_index = load_sentinel_tile_index(region.sentinel_tile_index_paths)
    if sentinel_index is not None and not sentinel_index.empty:
        # Build the shared STRtree before worker threads start querying it.
        sentinel_index.sindex
    site_names = sorted(
        {
            path.name
            for path in image_root.iterdir()
            if path.is_dir()
        }
    )
    if not site_names:
        raise RuntimeError(f"no site directories under {image_root}")
    site_dirs = [image_root / name for name in site_names]

    site_work: list[dict] = []
    imagery_rows: list[dict] = []
    product_rows: list[dict] = []
    default_imagery: dict[str, str] = {}
    def scan(site_dir: Path):
        return scan_site_imagery(region, site_dir, sentinel_index, footprint_size)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        scanned_sites = executor.map(scan, site_dirs)
        for index, scanned in enumerate(scanned_sites, start=1):
            if scanned is None:
                continue
            site_dir, site_id, assets = scanned
            if not assets:
                continue
            assets.sort(key=lambda item: (item["acquisition_date"], item["filename"]))
            default_asset = assets[-1]
            default_asset["is_default"] = 1
            union = polygonal_geometry(unary_union([item["geometry"] for item in assets]))
            core = majority_coverage_geometry([item["geometry"] for item in assets], union)
            tiles = sentinel_tiles_for_geometry(union, sentinel_index)
            if not tiles:
                tiles = sorted({item["mgrs_tile"] for item in assets if item["mgrs_tile"]})
            for asset in assets:
                imagery_rows.append(asset)
                product_rows.append(asset_product_row(site_id, site_dir, asset))
            if default_asset["mgrs_tile"]:
                default_imagery[f"site:{site_id}"] = default_asset["asset_id"]
            site_work.append(
                {
                    "site_id": site_id,
                    "local_directory_id": site_dir.name,
                    "source_path": display_region_path(site_dir, region),
                    "geometry": union,
                    "core_geometry": core,
                    "imagery": assets,
                    "sentinel_tiles": tiles,
                }
            )
            print(f"[{region.key}] imagery {index}/{len(site_dirs)} {site_id}: {len(assets)} assets")

    if not site_work:
        raise RuntimeError(f"no readable imagery under {region.local_imagery_root}")

    labels, label_asset_counts, label_feature_counts = load_local_label_features(
        region, site_work, label_output, reuse_existing=reuse_label_cache
    )
    # Only the enclosing bbox is needed by the external readers.  Unioning all
    # site footprints here creates a very large temporary geometry for regions
    # with thousands of sites.
    bounds = [item["geometry"].bounds for item in site_work]
    total_bounds = (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )
    osm = load_external_osm_water(region.osm_water, total_bounds)
    hydro = load_external_hydrolakes(region.hydrolakes, total_bounds)
    external_rows: list[dict] = []
    site_candidate_counts: dict[str, Counter] = {}
    suggested_names: dict[str, str] = {}
    for item in site_work:
        site_id = item["site_id"]
        candidates = build_vector_candidates(site_id, item["geometry"], osm, hydro)
        mark_suggested_candidate(candidates)
        external_rows.extend(candidates)
        site_candidate_counts[site_id] = Counter(row["source"] for row in candidates)
        suggested_names[site_id] = suggested_site_name(candidates)

    if include_raster_water:
        for index, item in enumerate(site_work, start=1):
            site_id = item["site_id"]
            site_like = site_object(site_id, item["geometry"])
            esa = build_esa_smoothed_layer(region, site_like)
            if esa is not None:
                write_esa_polygon_cache(region, site_id, esa)
                if esa.get("geometry"):
                    site_candidate_counts[site_id]["esa"] += 1
            jrc = build_jrc_occurrence_layer(region, site_like, jrc_threshold)
            if jrc is not None:
                write_jrc_polygon_cache(region, site_id, jrc_threshold, jrc)
                if jrc.get("geometry"):
                    site_candidate_counts[site_id]["jrc"] += 1
            print(f"[{region.key}] raster water {index}/{len(site_work)} {site_id}")

    site_rows = []
    core_rows = []
    for item in site_work:
        site_id = item["site_id"]
        assets = item["imagery"]
        dates = [asset["acquisition_date"] for asset in assets if asset["acquisition_date"]]
        bounds = item["geometry"].bounds
        suggested = suggested_names.get(site_id, "")
        display_name = site_display_name(item["local_directory_id"], suggested)
        counts = site_candidate_counts.get(site_id, Counter())
        site_rows.append(
            {
                "site_id": site_id,
                "region": region.key,
                "local_directory_id": item["local_directory_id"],
                "display_name": display_name,
                "suggested_name": suggested or None,
                "identity_source": "local_imagery",
                "source_path": item["source_path"],
                "image_count": len(assets),
                "label_asset_count": label_asset_counts.get(site_id, 0),
                "label_feature_count": label_feature_counts.get(site_id, 0),
                "external_feature_count": sum(counts.values()),
                "osm_feature_count": counts.get("osm", 0),
                "hydrolakes_feature_count": counts.get("hydrolakes", 0),
                "esa_feature_count": counts.get("esa", 0),
                "jrc_feature_count": counts.get("jrc", 0),
                "first_acquisition_date": min(dates) if dates else None,
                "last_acquisition_date": max(dates) if dates else None,
                "sentinel_tiles": ",".join(item["sentinel_tiles"]),
                "coverage_area_km2": geometry_area_km2(item["geometry"]),
                "core_area_km2": geometry_area_km2(item["core_geometry"]),
                "core_coverage_fraction": 0.8,
                "bbox_west": bounds[0],
                "bbox_south": bounds[1],
                "bbox_east": bounds[2],
                "bbox_north": bounds[3],
                "center_lon": (bounds[0] + bounds[2]) / 2,
                "center_lat": (bounds[1] + bounds[3]) / 2,
                "geometry": item["geometry"],
            }
        )
        core_rows.append(
            {
                "site_id": site_id,
                "core_area_km2": geometry_area_km2(item["core_geometry"]),
                "geometry": item["core_geometry"],
            }
        )

    return BuildResult(
        sites=geo_frame(site_rows),
        cores=geo_frame(core_rows),
        imagery=geo_frame(imagery_rows),
        labels=geo_frame(labels) if label_output is None else None,
        external=geo_frame(external_rows),
        product_rows=product_rows,
        default_imagery=default_imagery,
        label_cache=label_output if label_output and label_output.exists() else None,
    )


def site_object(site_id: str, geometry) -> Any:
    bounds = geometry.bounds
    return SimpleNamespace(
        site_id=site_id,
        geometry=geometry,
        bbox=bounds,
        area_km2=geometry_area_km2(geometry),
    )


def asset_product_row(site_id: str, site_dir: Path, asset: dict) -> dict:
    return {
        "site_id": site_id,
        "product_id": asset["asset_id"],
        "product_name": asset["asset_id"],
        "tile": asset["mgrs_tile"] or "",
        "date": asset["acquisition_date"] or "",
        "cloud_cover": "",
        "product_type": f"MSIL1C_{asset['storage_format'].upper()}",
        "source": "local_imagery",
        "safe_path": asset["path"].rsplit("/", 1)[0] if "/" in asset["path"] else "",
        "tci_path": asset["path"],
        "label_path": asset.get("label_path", ""),
        "download_status": "downloaded",
        "downloaded_at": "",
        "valid_ratio": asset["valid_ratio"],
    }


if __name__ == "__main__":
    main()
