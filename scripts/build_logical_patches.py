#!/usr/bin/env python3
"""Build canonical 512 x 512 logical patches from recorded training samples."""

from __future__ import annotations

import argparse

from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.logical_patches import build_logical_patches


def main() -> None:
    regions, default = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=[*regions, "all"], default=default)
    args = parser.parse_args()
    selected = regions.values() if args.region == "all" else [regions[args.region]]
    for region in selected:
        result = build_logical_patches(region)
        print(f"{region.key}: samples={result['samples']} logical_patches={result['patches']} manifest={result['manifest']}")


if __name__ == "__main__":
    main()
