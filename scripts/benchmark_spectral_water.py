#!/usr/bin/env python3
"""Benchmark spectral water methods against date-matched local shapefiles."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lake_workbench.automatic_labels.spectral import (  # noqa: E402
    IGNORE_LABEL,
    WATER_LABEL,
    read_spectral_evidence,
)
from lake_workbench.regions.config import load_region_configs  # noqa: E402


IMAGE_SUFFIXES = {".img", ".tif", ".tiff"}
METHODS = ("consensus_v2", "old_fixed_v1", "ndwi_zero", "mndwi_zero", "otsu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        action="append",
        dest="regions",
        help="region key; may be repeated (default: gansu and yunnan when available)",
    )
    parser.add_argument("--site", action="append", default=[], help="optional site directory ID")
    parser.add_argument("--limit", type=int, default=20, help="maximum image/label pairs per region")
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks" / "spectral_water.csv",
    )
    args = parser.parse_args()

    configured, _default = load_region_configs()
    requested = args.regions or [
        key for key in ("gansu", "yunnan") if key in configured
    ]
    unknown = sorted(set(requested) - set(configured))
    if unknown:
        parser.error(f"unknown regions: {', '.join(unknown)}")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for region_key in requested:
        region = configured[region_key]
        root = region.local_imagery_root or region.data_dir / "local_imagery"
        pairs = find_pairs(root, set(args.site))
        rng.shuffle(pairs)
        if args.limit > 0:
            pairs = pairs[: args.limit]
        print(f"{region_key}: benchmarking {len(pairs)} pairs", flush=True)
        for index, (image_path, label_path) in enumerate(pairs, start=1):
            try:
                result = benchmark_pair(
                    region_key,
                    image_path,
                    label_path,
                    max_dimension=max(128, min(2048, args.max_dimension)),
                )
                rows.extend(result)
                best = next(row for row in result if row["method"] == "consensus_v2")
                print(
                    f"  {index}/{len(pairs)} {image_path.parent.name}/{image_path.name} "
                    f"IoU={best['iou']:.3f} F1={best['f1']:.3f} "
                    f"coverage={best['confident_coverage']:.3f}",
                    flush=True,
                )
            except Exception as exc:  # A bad source pair should not stop the benchmark.
                failures.append(
                    {
                        "region": region_key,
                        "image": str(image_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  FAIL {image_path}: {exc}", flush=True)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output, rows)
    summary = summarize(rows)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reference_notice": (
            "Date-matched local shapefiles are treated as proxy reference labels; "
            "their omissions and geometry errors remain part of these metrics."
        ),
        "parameters": {
            "regions": requested,
            "sites": args.site,
            "limit_per_region": args.limit,
            "max_dimension": args.max_dimension,
            "seed": args.seed,
        },
        "pair_count": len({(row["region"], row["image_path"]) for row in rows}),
        "failure_count": len(failures),
        "summary": summary,
        "failures": failures,
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {output}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0 if rows else 1


def find_pairs(root: Path, site_ids: set[str]) -> list[tuple[Path, Path]]:
    if not root.exists():
        return []
    pairs = []
    for image_path in root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem.endswith("_Swater"):
            continue
        if site_ids and image_path.parent.name not in site_ids:
            continue
        label_path = image_path.with_name(f"{image_path.stem}_Swater.shp")
        if label_path.exists():
            pairs.append((image_path, label_path))
    return sorted(pairs)


def benchmark_pair(
    region: str,
    image_path: Path,
    label_path: Path,
    *,
    max_dimension: int,
) -> list[dict[str, Any]]:
    with rasterio.open(image_path) as src:
        if not src.crs:
            raise ValueError("image has no CRS")
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    evidence = read_spectral_evidence(
        image_path, tuple(bounds), max_dimension=max_dimension
    )
    reference = reference_mask(
        label_path, evidence.shape, evidence.transform, evidence.crs
    )
    valid = evidence.valid
    ndwi = evidence.indexes["ndwi"]
    mndwi = evidence.indexes["mndwi"]
    ndvi = evidence.indexes["ndvi"]
    old_probability = np.clip(
        (
            0.62 * sigmoid((mndwi - 0.05) / 0.09)
            + 0.38 * sigmoid(ndwi / 0.10)
        )
        * (1.0 - 0.45 * sigmoid((ndvi - 0.25) / 0.08)),
        0.0,
        1.0,
    )
    predictions = {
        "consensus_v2": evidence.labels == WATER_LABEL,
        "old_fixed_v1": old_probability >= 0.62,
        "ndwi_zero": ndwi >= 0.0,
        "mndwi_zero": mndwi >= 0.0,
        "otsu": evidence.methods["bounded_mndwi_otsu"],
    }
    confident_coverage = float(
        np.count_nonzero(valid & (evidence.labels != IGNORE_LABEL))
        / max(np.count_nonzero(valid), 1)
    )
    rows = []
    for method in METHODS:
        metrics = binary_metrics(predictions[method], reference, valid)
        rows.append(
            {
                "region": region,
                "site_id": image_path.parent.name,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "processing_level": evidence.diagnostics["processing_level"],
                "method": method,
                **metrics,
                "confident_coverage": confident_coverage
                if method == "consensus_v2"
                else 1.0,
                "valid_pixels": int(np.count_nonzero(valid)),
                "reference_water_pixels": int(np.count_nonzero(reference & valid)),
                "predicted_water_pixels": int(
                    np.count_nonzero(predictions[method] & valid)
                ),
                "cluster_status": evidence.diagnostics["cluster_status"],
                "cluster_best_k": evidence.diagnostics["cluster_best_k"],
                "otsu_threshold": evidence.diagnostics["otsu_mndwi_threshold"],
            }
        )
    return rows


def reference_mask(
    path: Path, shape: tuple[int, int], transform: Any, crs: Any
) -> np.ndarray:
    labels = gpd.read_file(path)
    if labels.crs is None:
        raise ValueError("reference shapefile has no CRS")
    labels = labels.to_crs(crs)
    geometries = [
        geometry
        for geometry in labels.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        return np.zeros(shape, dtype=bool)
    mask = rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    if mask is None:
        raise RuntimeError("Rasterio did not return a reference mask")
    return mask.astype(bool)


def binary_metrics(
    prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    predicted = prediction & valid
    expected = reference & valid
    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected & valid))
    false_negative = int(np.count_nonzero(~predicted & expected & valid))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["region"]), str(row["method"]))].append(row)
        grouped[("all", str(row["method"]))].append(row)
    summary: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (region, method), items in grouped.items():
        summary[region][method] = {
            "pairs": len(items),
            **{
                metric: float(np.mean([float(item[metric]) for item in items]))
                for metric in (
                    "iou",
                    "precision",
                    "recall",
                    "f1",
                    "confident_coverage",
                )
            },
        }
    return dict(summary)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else ["region", "site_id", "image_path", "method"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


if __name__ == "__main__":
    raise SystemExit(main())
