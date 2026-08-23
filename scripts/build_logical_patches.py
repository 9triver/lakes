#!/usr/bin/env python3
"""Build workspace-owned logical patches from recorded training samples."""

from __future__ import annotations

import argparse

from lake_workbench.regions.config import load_region_configs
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.training.logical_patches import build_logical_patches
from lake_workbench.utils import read_csv_records


def main() -> None:
    regions, default = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=[*regions, "all"], default=default)
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--sample", default="", help="Append one shared Training Sample to this workspace")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=0)
    args = parser.parse_args()
    workspaces = WorkspaceStore(regions)
    workspaces.ensure_default_workspace()
    workspaces.get(args.workspace, allow_archived=False)
    selected = regions.values() if args.region == "all" else [regions[args.region]]
    for region in selected:
        manifest = workspaces.ensure_workspace_logical_patch_manifest(args.workspace, region.key)
        existing = read_csv_records(manifest)
        before = {row.get("logical_patch_id", "") for row in existing}
        sample_path = workspaces.ensure_workspace_training_samples(args.workspace, region.key)
        samples = read_csv_records(sample_path)
        sample_ids = {row.get("sample_id", "") for row in samples if row.get("sample_id")}
        if args.sample:
            sample_ids.add(args.sample)
        result = build_logical_patches(
            region,
            output_dir=manifest.parent,
            sample_ids=sample_ids,
            samples=samples,
            patch_size=args.patch_size,
            stride=args.stride or args.patch_size,
        )
        after = {row.get("logical_patch_id", "") for row in read_csv_records(manifest)}
        created = sorted(value for value in after - before if value)
        if created:
            workspaces.update_members(args.workspace, region.key, created, "include")
        print(f"{region.key}: workspace={args.workspace} samples={result['samples']} logical_patches={result['patches']} added={len(created)} manifest={result['manifest']}")


if __name__ == "__main__":
    main()
