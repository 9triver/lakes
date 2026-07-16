#!/usr/bin/env python3
"""Materialize a named training dataset from included logical patches."""

from __future__ import annotations

import argparse

from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.logical_patches import build_training_dataset, dataset_configs


def main() -> None:
    regions, default = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=[*regions, "all"], default=default)
    parser.add_argument("--config", choices=sorted(dataset_configs()), default="resize256_v1")
    args = parser.parse_args()
    selected = regions.values() if args.region == "all" else [regions[args.region]]
    for region in selected:
        if not region.logical_patch_manifest.exists():
            continue
        result = build_training_dataset(region, args.config)
        print(f"{region.key}: config={args.config} patches={result['patches']} manifest={result['manifest']}")


if __name__ == "__main__":
    main()
