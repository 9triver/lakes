#!/usr/bin/env python3
"""Build canonical 512 x 512 logical patches from recorded training samples."""

from __future__ import annotations

import argparse

from lake_workbench.regions.config import load_region_configs
from lake_workbench.profiles import ProfileStore
from lake_workbench.training.logical_patches import build_logical_patches
from lake_workbench.utils import read_csv_records


def main() -> None:
    regions, default = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=[*regions, "all"], default=default)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--sample", default="", help="Append one shared Training Sample to this user")
    args = parser.parse_args()
    profiles = ProfileStore(regions)
    profiles.ensure_default_profile()
    profiles.get(args.profile, allow_archived=False)
    selected = regions.values() if args.region == "all" else [regions[args.region]]
    for region in selected:
        manifest = profiles.ensure_profile_logical_patch_manifest(args.profile, region.key)
        existing = read_csv_records(manifest)
        before = {row.get("logical_patch_id", "") for row in existing}
        sample_ids = {row.get("sample_id", "") for row in existing if row.get("sample_id")}
        if args.sample:
            sample_ids.add(args.sample)
        result = build_logical_patches(region, output_dir=manifest.parent, sample_ids=sample_ids)
        after = {row.get("logical_patch_id", "") for row in read_csv_records(manifest)}
        created = sorted(value for value in after - before if value)
        if created:
            profiles.update_members(args.profile, region.key, created, "include")
        print(f"{region.key}: profile={args.profile} samples={result['samples']} logical_patches={result['patches']} added={len(created)} manifest={result['manifest']}")


if __name__ == "__main__":
    main()
