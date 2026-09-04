#!/usr/bin/env python3
"""Convert regional local imagery from ENVI IMG to verified GeoTIFF storage.

Each region is converted and verified in place. The original IMG files and
their sidecars remain next to the generated TIFF files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyogrio
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from convert_local_imagery import convert_region  # noqa: E402


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".txt"}


def raster_paths(region_root: Path) -> list[tuple[Path, Path]]:
    source_root = region_root / "raw" / "local_imagery"
    return [
        (source, source_root / source.relative_to(source_root).with_suffix(".tif"))
        for source in sorted(
            path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() == ".img"
        )
    ]


def verify_pair(pair: tuple[Path, Path]) -> tuple[Path, int, int]:
    source, destination = pair
    if not destination.is_file():
        raise RuntimeError(f"missing GeoTIFF for {source}: {destination}")
    with rasterio.open(source) as original, rasterio.open(destination) as converted:
        properties = ("width", "height", "count", "dtypes", "crs", "transform", "nodata")
        for property_name in properties:
            if getattr(original, property_name) != getattr(converted, property_name):
                raise RuntimeError(f"raster metadata mismatch for {source}: {property_name}")
        for band in range(1, original.count + 1):
            if original.checksum(band) != converted.checksum(band):
                raise RuntimeError(f"raster checksum mismatch for {source}, band {band}")
    return source, source.stat().st_size, destination.stat().st_size


def verify_rasters(region_root: Path, *, workers: int = 4) -> tuple[int, int, int]:
    pairs = raster_paths(region_root)
    if not pairs:
        raise RuntimeError(f"no IMG rasters under {region_root / 'raw' / 'local_imagery'}")

    verified = []
    with ProcessPoolExecutor(max_workers=max(1, min(16, workers))) as executor:
        for index, result in enumerate(executor.map(verify_pair, pairs), start=1):
            verified.append(result)
            print(f"[{region_root.name}] verified {index}/{len(pairs)} {result[0].name}", flush=True)
    source_bytes = sum(source_bytes for _source, source_bytes, _destination_bytes in verified)
    destination_bytes = sum(destination_bytes for _source, _source_bytes, destination_bytes in verified)
    return len(pairs), source_bytes, destination_bytes


def rewrite_paths(value: str, region: str) -> str:
    return value.replace("raw/local_imagery_geotiff/", "raw/local_imagery/")


def update_text_references(region: str, region_root: Path, *, include_project_data: bool = False) -> int:
    roots = [region_root]
    if include_project_data:
        roots.insert(0, PROJECT_ROOT / "data")
    changed = 0
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            updated = rewrite_paths(original, region)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                changed += 1
    return changed


def update_gpkg(region: str, region_root: Path) -> None:
    path = region_root / "processed" / "site_metadata.gpkg"
    if not path.is_file():
        return
    path = path.resolve()
    temporary = path.with_name(f".{path.stem}.migrate-part.gpkg")
    temporary.unlink(missing_ok=True)
    path_columns = {
        "sites": {"source_path"},
        "imagery_assets": {"filename", "path", "label_path"},
        "local_label_features": {"source_path"},
    }
    layers = [row[0] for row in pyogrio.list_layers(path)]
    try:
        for index, layer in enumerate(layers):
            dataframe = pyogrio.read_dataframe(path, layer=layer)
            for column in path_columns.get(layer, set()).intersection(dataframe.columns):
                dataframe[column] = dataframe[column].map(
                    lambda value: rewrite_paths(str(value), region)
                    if value is not None
                    else value
                )
            pyogrio.write_dataframe(
                dataframe,
                temporary,
                layer=layer,
                driver="GPKG",
                append=index > 0,
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def update_products_csv(region_root: Path) -> None:
    path = region_root / "processed" / "sentinel_products.csv"
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    region = region_root.name
    for row in rows:
        for key, value in list(row.items()):
            row[key] = rewrite_paths(value or "", region)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def migrate_region(region_root: Path, *, workers: int) -> dict:
    source_root = region_root / "raw" / "local_imagery"
    if not source_root.is_dir():
        return {"region": region_root.name, "status": "skipped", "reason": "source_missing"}
    summary = convert_region(region_root, workers=workers)
    count, source_bytes, destination_bytes = verify_rasters(region_root, workers=workers)
    if count != summary["source_count"]:
        raise RuntimeError(f"conversion count changed during verification for {region_root.name}")
    update_gpkg(region_root.name, region_root)
    update_products_csv(region_root)
    changed_files = update_text_references(region_root.name, region_root)
    return {
        "region": region_root.name,
        "status": "migrated",
        "image_count": count,
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "reference_files": changed_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--regions", required=True, help="comma-separated region keys")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    summaries = []
    for region in (item.strip() for item in args.regions.split(",")):
        if not region:
            continue
        summary = migrate_region(root / region, workers=args.workers)
        summaries.append(summary)
        print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
