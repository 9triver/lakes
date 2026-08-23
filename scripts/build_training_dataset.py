#!/usr/bin/env python3
"""Materialize a named training dataset from included logical patches."""

from __future__ import annotations

import argparse

from lake_workbench.regions.config import load_region_configs
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.training.logical_patches import build_workspace_training_dataset, dataset_configs


def main() -> None:
    regions, default = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=[*regions, "all"], default=default)
    parser.add_argument("--config", choices=sorted(dataset_configs()), default="resize256_v1")
    parser.add_argument("--workspace", default="default")
    args = parser.parse_args()
    workspaces = WorkspaceStore(regions)
    workspaces.ensure_default_workspace()
    workspaces.get(args.workspace, allow_archived=False)
    selected = regions.values() if args.region == "all" else [regions[args.region]]
    for region in selected:
        if not workspaces.ensure_workspace_logical_patch_manifest(args.workspace, region.key).exists():
            continue
        if not workspaces.members(args.workspace, region.key):
            continue
        result = build_workspace_training_dataset(region, args.config, workspaces, args.workspace)
        print(f"{region.key}: workspace={args.workspace} config={args.config} patches={result['patches']} manifest={result['manifest']}")


if __name__ == "__main__":
    main()
