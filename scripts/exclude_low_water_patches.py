#!/usr/bin/env python3
"""Exclude generated Workspace patches whose water fraction is below a limit."""

from __future__ import annotations

import argparse
import math

from lake_workbench.regions.config import load_region_configs
from lake_workbench.utils import read_csv_records
from lake_workbench.workspaces import WorkspaceStore


def water_ratio(row: dict) -> float:
    """Use integer counts to avoid rounding near the threshold."""
    valid = int(row["valid_pixels"])
    water = int(row["water_pixels"])
    if valid < 0 or water < 0 or water > valid:
        raise ValueError(f"Invalid pixel counts: {row.get('logical_patch_id')}")
    return water / valid if valid else 0.0


def main() -> None:
    regions, _ = load_region_configs()
    parser = argparse.ArgumentParser(description="将水体占有效像元比例低于阈值的 Patch 标记为已排除")
    parser.add_argument("--region", required=True, choices=list(regions))
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--sites", nargs="+", help="仅处理指定 Site；省略则处理该省所有 Patch")
    parser.add_argument("--min-water-ratio", type=float, default=0.01, help="比例阈值，0.01 表示 1%%")
    parser.add_argument("--dry-run", action="store_true", help="只显示数量，不保存排除状态")
    args = parser.parse_args()
    if not math.isfinite(args.min_water_ratio) or not 0 <= args.min_water_ratio <= 1:
        parser.error("--min-water-ratio 必须在 0 到 1 之间")

    store = WorkspaceStore(regions)
    store.get(args.workspace, allow_archived=False)
    manifest = store.workspace_logical_patch_manifest(args.workspace, args.region)
    if not manifest.is_file():
        parser.error(f"Patch 清单不存在: {manifest}")
    sites = set(args.sites or [])
    rows = [row for row in read_csv_records(manifest) if not sites or row.get("site_id") in sites]
    # Validate every row before changing membership.
    wanted = [row["logical_patch_id"] for row in rows if water_ratio(row) < args.min_water_ratio]
    print(f"检查 {len(rows)} 个 Patch，水体占比低于 {args.min_water_ratio:.2%}：{len(wanted)} 个")
    if args.dry_run:
        print("预览完成，未修改数据")
    elif wanted:
        result = store.update_members(args.workspace, args.region, wanted, "exclude")
        print(f"新增排除 {result['updated']} 个 Patch；刷新网页可查看已排除状态")
    else:
        print("无需排除")


if __name__ == "__main__":
    main()
