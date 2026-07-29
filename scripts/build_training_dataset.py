#!/usr/bin/env python3
"""Materialize a named training dataset from included logical patches."""

from __future__ import annotations

import argparse

from lake_workbench.regions.config import load_region_configs
from lake_workbench.profiles import ProfileStore
from lake_workbench.training.logical_patches import build_profile_training_dataset, dataset_configs


def main() -> None:
    regions, default = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=[*regions, "all"], default=default)
    parser.add_argument("--config", choices=sorted(dataset_configs()), default="resize256_v1")
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()
    profiles = ProfileStore(regions)
    profiles.ensure_default_profile()
    profiles.get(args.profile, allow_archived=False)
    selected = regions.values() if args.region == "all" else [regions[args.region]]
    for region in selected:
        if not profiles.ensure_profile_logical_patch_manifest(args.profile, region.key).exists():
            continue
        if not profiles.members(args.profile, region.key):
            continue
        result = build_profile_training_dataset(region, args.config, profiles, args.profile)
        print(f"{region.key}: profile={args.profile} config={args.config} patches={result['patches']} manifest={result['manifest']}")


if __name__ == "__main__":
    main()
