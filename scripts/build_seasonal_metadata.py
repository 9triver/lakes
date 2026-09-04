#!/usr/bin/env python3
"""Build metadata for the transferred seasonal imagery dataset.

The seasonal dataset lives outside the repository's normal ``data/`` tree,
so this wrapper reuses the metadata builder with region paths redirected to
the transferred dataset. The original regional metadata is untouched.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_site_metadata import build_site_metadata, write_result  # noqa: E402
from lake_workbench.regions.config import load_region_configs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--regions", default="", help="comma-separated region keys; default: all transferred regions")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jrc-threshold", type=int, default=75)
    args = parser.parse_args()

    configured, _default = load_region_configs()
    dataset_root = args.dataset_root.expanduser().resolve()
    requested = {item.strip() for item in args.regions.split(",") if item.strip()}
    available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and path.name != "plans")
    region_keys = [key for key in available if not requested or key in requested]
    unknown = set(region_keys) - set(configured)
    if unknown:
        raise SystemExit(f"regions missing from config/regions.toml: {', '.join(sorted(unknown))}")

    for key in region_keys:
        base = configured[key]
        region_root = dataset_root / key
        region = replace(
            base,
            data_dir=region_root,
            processed_dir=region_root / "processed",
            local_imagery_root=region_root / "raw" / "local_imagery",
        )
        output_dir = region.processed_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        required_outputs = (
            output_dir / "site_metadata.gpkg",
            output_dir / "site_metadata.csv",
            output_dir / "sentinel_products.csv",
            output_dir / "active_imagery.json",
        )
        if all(path.exists() for path in required_outputs):
            print(f"SKIP {key}: metadata already exists", flush=True)
            continue
        print(f"BEGIN {key} source={region.local_imagery_root}", flush=True)
        result = build_site_metadata(
            region,
            jrc_threshold=max(1, min(100, args.jrc_threshold)),
            workers=max(1, min(16, args.workers)),
            label_output=output_dir / ".local_labels.building.gpkg",
        )
        write_result(region, output_dir, result)
        print(f"DONE {key} sites={len(result.sites)} imagery={len(result.imagery)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
