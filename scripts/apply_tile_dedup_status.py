#!/usr/bin/env python3
"""Apply Sentinel tile deduplication by changing membership only.

All logical patch rows remain in each manifest. For each region, the primary
Sentinel MGRS tile is selected from the product catalog (most frequent tile;
lexicographic tie break). One site is retained per primary tile. Included rows
from the other sites are changed to excluded and are never deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REGIONS = ("hubei", "guangdong", "guangxi", "jiangsu", "qinghai", "yunnan", "henan", "zhejiang", "gansu", "shaanxi", "shandong", "sichuan", "jiangxi")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def normalized_tile(value: object) -> str:
    return str(value or "").strip().upper().removeprefix("T")


def included(row: dict[str, str]) -> bool:
    status = str(row.get("review_status") or "").strip().lower()
    if status in {"included", "excluded"}:
        return status == "included"
    return str(row.get("include") or "").strip().lower() in {"true", "1", "yes"}


def update_reason(value: str, reason: str) -> str:
    existing = [part.strip() for part in str(value or "").split(";") if part.strip()]
    if reason not in existing:
        existing.append(reason)
    return ";".join(existing)


def choose_tiles(
    products: list[dict[str, str]], metadata: list[dict[str, str]],
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    product_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in products:
        site_id = str(row.get("site_id") or "").strip()
        tile = normalized_tile(row.get("tile"))
        if site_id and tile:
            product_counts[site_id][tile] += 1

    metadata_tiles: dict[str, list[str]] = {}
    for row in metadata:
        site_id = str(row.get("site_id") or "").strip()
        values = sorted({normalized_tile(item) for item in str(row.get("sentinel_tiles") or "").split(",") if normalized_tile(item)})
        if site_id:
            metadata_tiles[site_id] = values

    primary: dict[str, set[str]] = {}
    all_sites = set(product_counts) | set(metadata_tiles)
    for site_id in all_sites:
        choices = product_counts.get(site_id, Counter())
        # The geometry-derived metadata is the complete intersection set. The
        # product catalog is only a fallback for sites whose metadata is empty.
        if metadata_tiles.get(site_id):
            primary[site_id] = set(metadata_tiles[site_id])
        else:
            primary[site_id] = set(choices)
    return primary, metadata_tiles


def update_json_metadata(path: Path, summary: dict[str, object]) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(value, dict) or not {"included", "excluded"}.issubset(value):
        return
    value["included"] = summary["included_after"]
    value["excluded"] = summary["excluded_after"]
    value["tile_deduplication"] = True
    value["tile_deduplication_rule"] = (
        "Select non-overlapping observation regions using their complete Sentinel MGRS tile sets; "
        "duplicate sites retain all rows but included rows become excluded."
    )
    value["tile_deduplication_changed_included"] = summary["changed_included"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def process_region(workspace_root: Path, source_root: Path, region: str, backup_root: Path, *, dry_run: bool = False) -> dict[str, object]:
    workspace = workspace_root / region
    manifest = workspace / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    fields, rows = read_csv(manifest)
    product_fields, products = read_csv(source_root / region / "processed" / "sentinel_products.csv")
    metadata_fields, metadata = read_csv(source_root / region / "processed" / "site_metadata.csv")
    del product_fields, metadata_fields

    primary_tiles, metadata_tiles = choose_tiles(products, metadata)
    if "tile" not in fields:
        fields.append("tile")
    for row in rows:
        row["tile"] = sorted(primary_tiles.get(str(row.get("site_id") or ""), set()))[0] if primary_tiles.get(str(row.get("site_id") or ""), set()) else ""

    metrics: dict[str, dict[str, object]] = {}
    for row in rows:
        site_id = str(row.get("site_id") or "")
        record = metrics.setdefault(site_id, {
            "site_id": site_id,
            "tile": row.get("tile", ""),
            "tile_set": set(primary_tiles.get(site_id, set())),
            "sentinel_tiles": ",".join(metadata_tiles.get(site_id, [])),
            "patches": 0,
            "included_before": 0,
            "included_water_pixels": 0,
        })
        record["patches"] = int(record["patches"]) + 1
        if included(row):
            record["included_before"] = int(record["included_before"]) + 1
            try:
                record["included_water_pixels"] = int(record["included_water_pixels"]) + int(float(row.get("water_pixels") or 0))
            except (TypeError, ValueError):
                pass

    by_tile: dict[str, list[str]] = defaultdict(list)
    for record in metrics.values():
        for tile in record["tile_set"]:
            by_tile[tile].append(str(record["site_id"]))

    # Greedily keep the highest-quality regions first, while requiring every
    # retained pair to have disjoint Sentinel tile sets. This avoids collapsing
    # an entire connected chain into one site when its endpoints do not overlap.
    ranked_sites = sorted(
        (item for item in metrics.values() if item["tile_set"]),
        key=lambda item: (-int(item["included_before"]), -int(item["included_water_pixels"]), str(item["site_id"])),
    )
    retained: list[str] = []
    duplicate_sites: dict[str, str] = {}
    for candidate in ranked_sites:
        site_id = str(candidate["site_id"])
        tile_set = set(candidate["tile_set"])
        conflicts = [
            kept_id for kept_id in retained
            if tile_set & set(metrics[kept_id]["tile_set"])
        ]
        if conflicts:
            duplicate_sites[site_id] = conflicts[0]
        else:
            retained.append(site_id)

    changed = 0
    changed_by_site: Counter[str] = Counter()
    for row in rows:
        site_id = str(row.get("site_id") or "")
        retained_site = duplicate_sites.get(site_id)
        if retained_site and included(row):
            row["include"] = "false"
            row["review_status"] = "excluded"
            row["exclude_reason"] = update_reason(row.get("exclude_reason", ""), "duplicate_sentinel_tile")
            changed += 1
            changed_by_site[site_id] += 1

    included_after = sum(included(row) for row in rows)
    backup = None
    if not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = backup_root / region / f"before_tile_dedup_{timestamp}"
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest, backup / "manifest.csv")
        for path in sorted(workspace.glob("*.json")):
            shutil.copy2(path, backup / path.name)
        write_csv(manifest, fields, rows)
    summary = {
        "region": region,
        "sites": len(metrics),
        "tiles_with_sites": len(by_tile),
        "retained_sites": len(retained),
        "duplicate_sites": len(duplicate_sites),
        "patches": len(rows),
        "included_before": sum(int(item["included_before"]) for item in metrics.values()),
        "included_after": included_after,
        "excluded_after": len(rows) - included_after,
        "changed_included": changed,
        "backup": str(backup) if backup else None,
    }

    audit_fields = ["site_id", "primary_tile", "sentinel_tiles", "patches", "included_before", "changed_to_excluded", "included_after", "retained_site_id", "role"]
    audit_rows = []
    for site_id in sorted(metrics):
        item = metrics[site_id]
        tile = str(item["tile"])
        changed_site = int(changed_by_site[site_id])
        role = "retained" if site_id in retained else ("duplicate" if site_id in duplicate_sites else "no_tile")
        audit_rows.append({
            "site_id": site_id,
            "primary_tile": tile,
            "sentinel_tiles": str(item["sentinel_tiles"]),
            "patches": str(item["patches"]),
            "included_before": str(item["included_before"]),
            "changed_to_excluded": str(changed_site),
            "included_after": str(int(item["included_before"]) - changed_site),
            "retained_site_id": site_id if site_id in retained else duplicate_sites.get(site_id, ""),
            "role": role,
        })
    if not dry_run:
        write_csv(workspace / "tile_deduplication.csv", audit_fields, audit_rows)
        (workspace / "tile_deduplication_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in sorted(workspace.glob("*.json")):
            if path.name == "tile_deduplication_summary.json":
                continue
            update_json_metadata(path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/lake/lakes/data/workspaces/default/logical_patches"))
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/data/lakes-data/regions"))
    parser.add_argument("--backup-root", type=Path, default=None)
    parser.add_argument("--region", action="append", choices=REGIONS, dest="regions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    regions = args.regions or list(REGIONS)
    backup_root = args.backup_root or args.workspace_root
    summaries = [process_region(args.workspace_root, args.source_root, region, backup_root, dry_run=args.dry_run) for region in regions]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
