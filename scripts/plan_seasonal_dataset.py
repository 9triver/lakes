#!/usr/bin/env python3
"""Plan a compact three-period local imagery dataset.

The planner does not copy or delete imagery. It reads the regional imagery
inventory and writes a deterministic manifest that can be used for transfer.
Each site contributes at most one image to each hydrological period:
wet (June-August), normal (March-May and September-October), and dry
(November-February).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


PERIODS = {
    "wet": {6, 7, 8},
    "normal": {3, 4, 5, 9, 10},
    "dry": {11, 12, 1, 2},
}
PERIOD_ORDER = ("wet", "normal", "dry")


def hydrological_period(value: str) -> str | None:
    try:
        month = date.fromisoformat(value).month
    except (TypeError, ValueError):
        return None
    return next((period for period, months in PERIODS.items() if month in months), None)


def number(value: str | None, fallback: float) -> float:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def label_candidates(image: Path) -> list[Path]:
    return [
        image.with_name(f"{image.stem}.Swater.shp"),
        image.with_name(f"{image.stem}_Swater.shp"),
    ]


def label_family(label: Path) -> list[Path]:
    return [
        label.with_suffix(suffix)
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix")
        if label.with_suffix(suffix).is_file()
    ]


def source_relative(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is outside source root {source_root}") from exc


def source_root_for(path: Path, roots: list[tuple[str, Path]]) -> tuple[str, Path] | None:
    for name, root in roots:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        return name, root
    return None


def companion_files(image: Path, source_root: Path) -> list[str]:
    candidates = [
        image,
        image.with_suffix(".hdr"),
        image.with_name(f"{image.name}.aux.xml"),
        image.with_name(f"{image.name}.ovr"),
    ]
    files = [source_relative(path, source_root) for path in candidates if path.is_file()]
    labels = next((label for label in label_candidates(image) if label.is_file()), None)
    if labels is not None:
        files.extend(source_relative(path, source_root) for path in label_family(labels))
    return list(dict.fromkeys(files))


def choose_candidate(candidates: list[dict]) -> dict:
    # Labels and valid pixels are stronger signals than acquisition recency.
    return max(
        candidates,
        key=lambda row: (
            int(row["has_label"]),
            number(row["valid_ratio"], 0.0),
            -number(row["cloud_cover"], 101.0),
            row["date"],
            row["product_id"],
        ),
    )


def plan_region(region_root: Path, output_dir: Path, max_per_period: int) -> dict:
    source_root = region_root / "raw" / "local_imagery"
    source_roots = [("local_imagery", source_root)] if source_root.is_dir() else []
    inventory_path = region_root / "processed" / "sentinel_products.csv"
    if not source_roots or not inventory_path.is_file():
        return {"region": region_root.name, "status": "skipped", "reason": "missing input"}

    by_site_period: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") not in {"local_img", "local_imagery"}:
                continue
            period = hydrological_period(row.get("date", ""))
            image = Path(row.get("tci_path", ""))
            storage = source_root_for(image, source_roots)
            if period is None or storage is None or not image.is_file() or image.stat().st_size <= 0:
                continue
            storage_name, image_root = storage
            labels = next((path for path in label_candidates(image) if path.is_file()), None)
            candidate = {
                "region": region_root.name,
                "site_id": row.get("site_id", ""),
                "period": period,
                "date": row.get("date", ""),
                "product_id": row.get("product_id", ""),
                "product_name": row.get("product_name", ""),
                "tile": row.get("tile", ""),
                "source_root": storage_name,
                "source_relpath": source_relative(image, image_root),
                "label_relpath": source_relative(labels, image_root) if labels else "",
                "has_label": bool(labels),
                "valid_ratio": row.get("valid_ratio", ""),
                "cloud_cover": row.get("cloud_cover", ""),
                "companion_files": companion_files(image, image_root),
            }
            by_site_period[(candidate["site_id"], period)].append(candidate)

    selected: list[dict] = []
    for (site_id, period), candidates in sorted(by_site_period.items()):
        candidates.sort(key=lambda row: row["date"], reverse=True)
        for rank in range(max_per_period):
            if not candidates:
                break
            candidate = choose_candidate(candidates)
            candidate["selection_rank"] = rank + 1
            selected.append(candidate)
            candidates.remove(candidate)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fields = [
        "region", "site_id", "period", "selection_rank", "date", "product_id",
        "product_name", "tile", "source_root", "source_relpath", "label_relpath", "has_label",
        "valid_ratio", "cloud_cover", "companion_files",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            row = row.copy()
            row["companion_files"] = json.dumps(row["companion_files"], ensure_ascii=False)
            writer.writerow({field: row.get(field, "") for field in fields})

    coverage = Counter(row["period"] for row in selected)
    site_periods = defaultdict(set)
    for row in selected:
        site_periods[row["site_id"]].add(row["period"])
    summary = {
        "region": region_root.name,
        "status": "planned",
        "site_count": len(site_periods),
        "selected_count": len(selected),
        "period_counts": {period: coverage[period] for period in PERIOD_ORDER},
        "sites_with_all_periods": sum(len(periods) == 3 for periods in site_periods.values()),
        "sites_with_two_periods": sum(len(periods) == 2 for periods in site_periods.values()),
        "sites_with_one_period": sum(len(periods) == 1 for periods in site_periods.values()),
        "manifest": str(manifest_path),
        "file_count": len({path for row in selected for path in row["companion_files"]}),
        "bytes": sum(
            path.stat().st_size
            for row in selected
            for path in (
                next(root for name, root in source_roots if name == row["source_root"]) / rel
                for rel in row["companion_files"]
            )
            if path.is_file()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--regions", default="", help="comma-separated region keys; default: all")
    parser.add_argument("--max-per-period", type=int, default=1)
    args = parser.parse_args()
    if args.max_per_period < 1:
        parser.error("--max-per-period must be at least 1")

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    requested = {item.strip() for item in args.regions.split(",") if item.strip()}
    regions = [path for path in sorted((data_root / "regions").iterdir()) if path.is_dir()]
    if requested:
        regions = [path for path in regions if path.name in requested]
    summaries = [
        plan_region(region, output_root / region.name, args.max_per_period)
        for region in regions
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for summary in summaries:
        print(
            f"{summary['region']}: {summary['status']} "
            f"sites={summary.get('site_count', 0)} selected={summary.get('selected_count', 0)} "
            f"bytes={summary.get('bytes', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
