#!/usr/bin/env python3
"""Build Jiangsu's first-five logical patches and retain one region per tile.

All source imagery and annotations are read-only. Candidate patches are built
for every region so the tile representative can follow the existing rule of
maximizing included patches, then included water pixels, then site ID.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "dataset_builds"))

import rebuild_hubei_first5_fullgrid as fullgrid  # noqa: E402


REGION = "jiangsu"
SOURCE_ROOT = Path("/mnt/data/lakes-data/regions") / REGION
WORKSPACE = Path("/home/lake/lakes/data/workspaces/default/logical_patches") / REGION
OBSERVATION_FIELDS = [
    "site_id", "tile", "sample_id", "product_name", "date", "selection_rank",
    "image_path", "source_image_path", "source_label_path", "catalog_valid_ratio",
]
TILE_FIELDS = [
    "tile", "site_id", "observations", "patches", "included", "excluded",
    "included_water_pixels", "dates",
]


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_tile(value: str) -> str:
    return str(value or "").strip().upper().removeprefix("T")


def inventory() -> tuple[list[dict], dict, list[dict], list[dict], list[dict], list[dict]]:
    metadata_path = SOURCE_ROOT / "processed" / "site_metadata.csv"
    products_path = SOURCE_ROOT / "processed" / "sentinel_products.csv"
    metadata_rows = read_rows(metadata_path)
    metadata = {row["site_id"]: row for row in metadata_rows if row.get("site_id")}
    products = read_rows(products_path)

    tile_counts: dict[str, Counter] = defaultdict(Counter)
    available: dict[str, list[dict]] = defaultdict(list)
    missing_sources = []
    for row in products:
        site_id = row.get("site_id", "")
        tile = normalized_tile(row.get("tile", ""))
        if site_id and tile:
            tile_counts[site_id][tile] += 1
        if row.get("source") not in {"local_img", "local_imagery"} or not row.get("date"):
            continue
        source_image = Path(row.get("tci_path", ""))
        if not source_image.name.startswith(("S2A_", "S2B_", "S2C_")):
            continue
        if not source_image.is_file():
            missing_sources.append({
                "site_id": site_id,
                "tile": tile,
                "product_id": row.get("product_id", ""),
                "date": row.get("date", ""),
                "image_path": str(source_image),
            })
            continue
        image_path = source_image.with_suffix(".tif") if source_image.with_suffix(".tif").is_file() else source_image
        label_path = next((path for path in (
            source_image.with_name(source_image.stem + ".Swater.shp"),
            source_image.with_name(source_image.stem + "_Swater.shp"),
        ) if path.is_file()), None)
        available[site_id].append({
            "site_id": site_id,
            "sample_id": row.get("product_id") or source_image.stem,
            "product_name": row.get("product_name") or row.get("product_id", ""),
            "date": row["date"],
            "image_path": str(image_path),
            "source_image_path": str(source_image),
            "source_label_path": str(label_path) if label_path else "",
            "catalog_valid_ratio": row.get("valid_ratio", ""),
        })

    site_tiles = {}
    tile_conflicts = []
    all_sites = sorted(set(metadata) | set(available) | set(tile_counts))
    for site_id in all_sites:
        counts = tile_counts.get(site_id, Counter())
        fallback = [normalized_tile(value) for value in metadata.get(site_id, {}).get("sentinel_tiles", "").split(",")]
        fallback = [value for value in fallback if value]
        choices = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        tile = choices[0][0] if choices else (fallback[0] if len(fallback) == 1 else "")
        site_tiles[site_id] = tile
        if len(choices) > 1:
            tile_conflicts.append({
                "site_id": site_id,
                "tile": tile,
                "tile_counts": json.dumps(dict(choices), ensure_ascii=False, sort_keys=True),
            })

    observations = []
    incomplete = {}
    missing_labels = []
    candidate_sites = sorted(available)
    for site_id in candidate_sites:
        tile = site_tiles.get(site_id, "")
        by_date = {}
        for row in sorted(available[site_id], key=lambda item: (item["date"], item["sample_id"])):
            by_date.setdefault(row["date"], row)
        dates = sorted(by_date)
        chosen = [by_date[date] for date in dates[:5]]
        if len(chosen) != 5:
            incomplete[site_id] = len(chosen)
        for rank, item in enumerate(chosen, start=1):
            item = {**item, "tile": tile, "selection_rank": rank}
            observations.append(item)
            if not item["source_label_path"]:
                missing_labels.append(item)

    unavailable = [
        {"site_id": site_id, "tile": site_tiles.get(site_id, ""),
         "image_count": metadata.get(site_id, {}).get("image_count", "0"),
         "reason": "no_available_local_sentinel_image"}
        for site_id in all_sites if site_id not in available
    ]
    diagnostics = {
        "metadata_regions": len(metadata),
        "product_catalog_regions": len({row.get("site_id") for row in products if row.get("site_id")}),
        "available_regions": len(candidate_sites),
        "unique_tiles": len({site_tiles[site] for site in candidate_sites if site_tiles.get(site)}),
        "regions_without_tile": sum(not site_tiles.get(site) for site in candidate_sites),
        "tile_conflicts": len(tile_conflicts),
        "missing_source_rows": len(missing_sources),
        "unavailable_regions": len(unavailable),
        "selected_observations": len(observations),
        "incomplete_regions": incomplete,
        "missing_label_observations": missing_labels,
        "selection_rule": "Earliest five distinct dates with a readable local Sentinel image per region.",
        "tile_rule": "Most frequent catalog tile per region; ties use lexicographically smallest tile.",
        "representative_rule": "Per tile: most included patches, then most water pixels in included patches, then lexicographically smallest site ID.",
        "patch_size": fullgrid.SIZE,
        "stride": fullgrid.STRIDE,
        "thresholds": {"min_water_ratio_valid": 0.05, "min_valid_ratio": 0.80},
        "source_files_modified": False,
    }
    return observations, diagnostics, unavailable, missing_sources, tile_conflicts, missing_labels


def add_tile_column(path: Path, tile: str) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "tile" not in fields:
        fields.append("tile")
    for row in rows:
        row["tile"] = tile
    write_rows(path, rows, fields)


def candidate_metrics(observations: list[dict], stats: list[dict], output: Path) -> dict[str, dict]:
    by_sample = {row["sample_id"]: row for row in observations}
    metrics: dict[str, dict] = {}
    for stat in stats:
        site_id = stat["site_id"]
        item = by_sample[stat["sample_id"]]
        record = metrics.setdefault(site_id, {
            "site_id": site_id,
            "tile": item["tile"],
            "observations": 0,
            "patches": 0,
            "included": 0,
            "excluded": 0,
            "included_water_pixels": 0,
            "dates": [],
        })
        record["observations"] += 1
        record["patches"] += int(stat["patches"])
        record["included"] += int(stat["included"])
        record["excluded"] += int(stat["excluded"])
        record["dates"].append(item["date"])
        scene_csv = output / "scenes" / f"{item['sample_id']}.csv"
        with scene_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("include") == "true":
                    record["included_water_pixels"] += int(row.get("water_pixels") or 0)
    for record in metrics.values():
        record["dates"] = ",".join(sorted(record["dates"]))
    return metrics


def choose_representatives(metrics: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for record in metrics.values():
        if record["tile"]:
            by_tile[record["tile"]].append(record)
    representatives = []
    duplicates = []
    for tile, candidates in sorted(by_tile.items()):
        candidates.sort(key=lambda row: (
            -row["included"],
            -row["included_water_pixels"],
            row["site_id"],
        ))
        winner = dict(candidates[0])
        representatives.append(winner)
        for duplicate in candidates[1:]:
            duplicates.append({
                "tile": tile,
                "site_id": duplicate["site_id"],
                "retained_site_id": winner["site_id"],
                "patches": duplicate["patches"],
                "included": duplicate["included"],
                "excluded": duplicate["excluded"],
                "included_water_pixels": duplicate["included_water_pixels"],
                "reason": "duplicate_sentinel_tile",
            })
    return representatives, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-regions", type=int, default=1576)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    source = SOURCE_ROOT.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("output must be outside the lakes source data tree")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("scenes", "labels"):
        (output / name).mkdir(exist_ok=True)

    if args.publish:
        occupied = [WORKSPACE / name for name in (
            "manifest.csv", "source_dataset.json", "source_dataset_fullgrid.json",
        ) if (WORKSPACE / name).exists()]
        if occupied:
            raise FileExistsError("Refusing to replace existing Jiangsu workspace data: " + ", ".join(map(str, occupied)))

    observations, plan, unavailable, missing_sources, tile_conflicts, missing_labels = inventory()
    plan.update({
        "dataset_id": output.name,
        "expected_regions_from_request": args.expected_regions,
        "difference_from_expected_regions": plan["metadata_regions"] - args.expected_regions,
        "source_root": str(source),
        "output": str(output),
    })
    write_rows(output / "observations.csv", observations, OBSERVATION_FIELDS)
    write_rows(output / "unavailable_regions.csv", unavailable,
               ["site_id", "tile", "image_count", "reason"])
    write_rows(output / "missing_sources.csv", missing_sources,
               ["site_id", "tile", "product_id", "date", "image_path"])
    write_rows(output / "tile_conflicts.csv", tile_conflicts,
               ["site_id", "tile", "tile_counts"])
    fullgrid.atomic_json(output / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"status": "planned", **plan}, ensure_ascii=False), flush=True)
        return 0

    fullgrid.REGION = REGION
    fullgrid.ROOT = source
    fullgrid.WORKSPACE = WORKSPACE
    selected_stats = []
    failures = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        tasks = {pool.submit(fullgrid.process_scene, (item, str(output))): item for item in observations}
        for future in concurrent.futures.as_completed(tasks):
            item = tasks[future]
            try:
                selected_stats.append(future.result())
            except Exception as error:
                failures.append({**item, "error": repr(error)})
            completed = len(selected_stats) + len(failures)
            if completed % 25 == 0 or completed == len(observations):
                status = {
                    "status": "building",
                    "completed_observations": completed,
                    "total_observations": len(observations),
                    "completed_regions": len({row["site_id"] for row in selected_stats}),
                    "candidate_patches": sum(int(row["patches"]) for row in selected_stats),
                    "candidate_included": sum(int(row["included"]) for row in selected_stats),
                    "candidate_excluded": sum(int(row["excluded"]) for row in selected_stats),
                    "errors": len(failures),
                    "elapsed_seconds": round(time.monotonic() - started),
                }
                fullgrid.atomic_json(output / "status.json", status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
    if failures:
        fullgrid.atomic_json(output / "failures.json", failures)
        raise SystemExit(f"{len(failures)} observations failed; no workspace manifest was published")

    metrics = candidate_metrics(observations, selected_stats, output)
    representatives, duplicates = choose_representatives(metrics)
    representative_ids = {row["site_id"] for row in representatives}
    representative_items = [row for row in observations if row["site_id"] in representative_ids]
    representative_stats = [row for row in selected_stats if row["site_id"] in representative_ids]
    incomplete = {row["site_id"]: row["observations"] for row in representatives if row["observations"] != 5}
    write_rows(output / "tile_representatives.csv", representatives, TILE_FIELDS)
    write_rows(output / "duplicate_regions.csv", duplicates, [
        "tile", "site_id", "retained_site_id", "patches", "included", "excluded",
        "included_water_pixels", "reason",
    ])
    if incomplete:
        fullgrid.atomic_json(output / "incomplete_representatives.json", incomplete)
        raise SystemExit("At least one tile has no representative with five usable observations; no workspace manifest was published")

    for item in observations:
        add_tile_column(output / "scenes" / f"{item['sample_id']}.csv", item["tile"])
    fullgrid.FIELDS = [*fullgrid.FIELDS, "tile"]
    validation = fullgrid.validate_publish(
        output, representative_items, representative_stats, args.publish
    )
    validation.update({
        "candidate_regions": len(metrics),
        "candidate_observations": len(observations),
        "candidate_patches": sum(row["patches"] for row in metrics.values()),
        "candidate_included": sum(row["included"] for row in metrics.values()),
        "candidate_excluded": sum(row["excluded"] for row in metrics.values()),
        "unique_tiles": len(representatives),
        "duplicate_regions_excluded_by_tile": len(duplicates),
        "tile_representatives": str(output / "tile_representatives.csv"),
        "duplicate_regions": str(output / "duplicate_regions.csv"),
        "tile_column_present": True,
        "source_files_modified": False,
    })
    fullgrid.atomic_json(output / "validation.json", validation)
    dataset = {
        "dataset_id": output.name,
        "status": "complete",
        "requested_regions": args.expected_regions,
        "catalog_regions": plan["metadata_regions"],
        "catalog_regions_minus_requested": plan["difference_from_expected_regions"],
        "candidate_regions": len(metrics),
        "unique_tiles": len(representatives),
        "retained_regions": len(representatives),
        "duplicate_regions_excluded_by_tile": len(duplicates),
        "candidate_observations": len(observations),
        "retained_observations": len(representative_items),
        "retained_patches": validation["patches"],
        "included": validation["included"],
        "excluded_by_quality_threshold": validation["excluded"],
        "thresholds": {"min_water_ratio_valid": 0.05, "min_valid_ratio": 0.80},
        "selection_rule": plan["selection_rule"],
        "representative_rule": plan["representative_rule"],
        "patch_size": fullgrid.SIZE,
        "stride": fullgrid.STRIDE,
        "source_root": str(source),
        "output": str(output),
        "workspace_manifest": str(WORKSPACE / "manifest.csv") if args.publish else "",
        "all_candidate_metrics": str(output / "tile_representatives.csv"),
        "source_data_policy": "Source imagery and annotations are read-only; only generated workspace files are published.",
    }
    fullgrid.atomic_json(output / "dataset.json", dataset)
    if args.publish:
        descriptor_path = WORKSPACE / "source_dataset.json"
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor.update({
            "tile_representatives": str(output / "tile_representatives.csv"),
            "duplicate_regions": str(output / "duplicate_regions.csv"),
            "dataset_summary": str(output / "dataset.json"),
        })
        fullgrid.atomic_json(descriptor_path, descriptor)
    status = {**validation, "status": "complete", "elapsed_seconds": round(time.monotonic() - started), "output": str(output)}
    fullgrid.atomic_json(output / "status.json", status)
    print(json.dumps({**dataset, "validation": validation, "elapsed_seconds": status["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
