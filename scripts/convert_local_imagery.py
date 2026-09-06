#!/usr/bin/env python3
"""Convert local ENVI IMG rasters to lossless tiled BigTIFF files.

The source files are never modified or removed. Existing GeoTIFF output is
skipped unless ``--overwrite`` is supplied. Conversion preserves raster
dimensions, CRS, transform, dtype, bands, nodata, descriptions, and tags.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def convert_image(source: Path, destination: Path, *, overwrite: bool = False) -> dict:
    if source.stat().st_size == 0:
        return {
            "source": str(source),
            "destination": str(destination),
            "source_bytes": 0,
            "destination_bytes": 0,
            "status": "skipped_empty",
        }
    if destination.exists() and not overwrite:
        return {"source": str(source), "destination": str(destination), "status": "skipped"}

    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(source) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="YES",
                compress="ZSTD",
                zstd_level=9,
                predictor=2,
            )
            with rasterio.open(temporary, "w", **profile) as dst:
                for row in range(0, src.height, 256):
                    for col in range(0, src.width, 256):
                        height = min(256, src.height - row)
                        width = min(256, src.width - col)
                        window = Window(col, row, width, height)
                        dst.write(src.read(window=window), window=window)
                for index, description in enumerate(src.descriptions, start=1):
                    if description:
                        dst.set_band_description(index, description)
                dst.update_tags(**src.tags())
                for index in range(1, src.count + 1):
                    tags = src.tags(index)
                    if tags:
                        dst.update_tags(index, **tags)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source": str(source),
        "destination": str(destination),
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
        "status": "converted",
    }


def convert_region(
    region_root: Path,
    output_root: Path | None = None,
    *,
    overwrite: bool = False,
    workers: int = 4,
) -> dict:
    source_root = region_root / "raw" / "local_imagery"
    output_root = output_root or source_root
    sources = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".img"
    )
    destinations = [output_root / source.relative_to(source_root).with_suffix(".tif") for source in sources]

    def convert(item: tuple[Path, Path]) -> dict:
        source, destination = item
        return convert_image(source, destination, overwrite=overwrite)

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(16, workers))) as executor:
        for index, (source, result) in enumerate(zip(sources, executor.map(convert, zip(sources, destinations))), start=1):
            results.append(result)
            print(f"[{region_root.name}] {index}/{len(sources)} {source.name} {result['status']}", flush=True)
    source_bytes = sum(
        result["source_bytes"] if "source_bytes" in result else source.stat().st_size
        for result, source in zip(results, sources)
    )
    destination_bytes = sum(
        result["destination_bytes"]
        if "destination_bytes" in result
        else destination.stat().st_size
        for result, destination in zip(results, destinations)
    )
    summary = {
        "region": region_root.name,
        "source_count": len(sources),
        "converted_count": sum(result["status"] == "converted" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
        "empty_count": sum(result["status"] == "skipped_empty" for result in results),
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = convert_region(
        args.region_root.expanduser().resolve(),
        args.output_root.expanduser().resolve() if args.output_root else None,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
