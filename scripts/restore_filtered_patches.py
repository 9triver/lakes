#!/usr/bin/env python3
"""Restore excluded Workspace patches that satisfy water and black-border limits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import math

import rasterio

from exclude_black_border_patches import black_border_counts
from lake_workbench.regions.config import load_region_configs
from lake_workbench.utils import read_csv_records, resolve_data_path
from lake_workbench.workspaces import WorkspaceStore, WorkspacePatchConflict
from lake_workbench.workspaces.membership import patch_is_included


def water_fraction(row: dict) -> Fraction:
    valid, water = int(row["valid_pixels"]), int(row["water_pixels"])
    if valid < 0 or not 0 <= water <= valid:
        raise ValueError(f"无效像元统计: {row.get('logical_patch_id')}")
    return Fraction(water, valid) if valid else Fraction(0)


def restore_without_conflicts(store, workspace: str, region: str, patch_ids: list[str]) -> tuple[int, int]:
    """Retry the atomic restore after removing incoming conflicting patches."""
    pending = list(dict.fromkeys(patch_ids))
    skipped: set[str] = set()
    while pending:
        try:
            result = store.update_members(workspace, region, pending, "restore")
            return result["updated"], len(skipped)
        except WorkspacePatchConflict as exc:
            conflicting = {item.get("patch_id") for item in exc.conflicts} & set(pending)
            if not conflicting:
                raise
            for patch_id in sorted(conflicting):
                print(f"跳过冲突 Patch：{patch_id}", flush=True)
            skipped.update(conflicting)
            pending = [patch_id for patch_id in pending if patch_id not in conflicting]
    return 0, len(skipped)


def main(argv: list[str] | None = None) -> int:
    regions, _ = load_region_configs()
    parser = argparse.ArgumentParser(description="按水体和黑边占比，将符合条件的已排除 Patch 重新纳入工作区")
    parser.add_argument("--region", required=True, choices=list(regions))
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--sites", nargs="+", help="只处理指定 Site，省略则处理全省")
    parser.add_argument("--min-water-ratio", type=float, default=0.01, help="水体占有效像元比例下限，默认 0.01（1%%）")
    parser.add_argument("--max-black-ratio", type=float, default=0.2, help="黑边占 Patch 总像元比例上限，默认 0.2（20%%）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不修改数据")
    args = parser.parse_args(argv)
    for name in ("min_water_ratio", "max_black_ratio"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} 必须在 0 到 1 之间")

    store = WorkspaceStore(regions)
    store.get(args.workspace, allow_archived=False)
    region = regions[args.region]
    manifest = store.workspace_logical_patch_manifest(args.workspace, args.region)
    if not manifest.is_file():
        parser.error(f"Patch 清单不存在: {manifest}")
    sites = set(args.sites or [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    water_limit = Fraction(str(args.min_water_ratio))
    black_limit = Fraction(str(args.max_black_ratio))
    checked = 0
    for row in read_csv_records(manifest):
        if patch_is_included(row) or (sites and row.get("site_id") not in sites):
            continue
        checked += 1
        if water_fraction(row) >= water_limit:
            if not row.get("image_path"):
                parser.error(f"Patch 缺少影像路径: {row.get('logical_patch_id')}")
            grouped[row["image_path"]].append(row)

    wanted = []
    # Finish scanning before writing any membership changes.
    for image_path, rows in grouped.items():
        with rasterio.open(resolve_data_path(image_path, region)) as source:
            for row in rows:
                black, total = black_border_counts(source, row)
                if Fraction(black, total) <= black_limit:
                    wanted.append(row["logical_patch_id"])
                    print(f"可恢复 {row['logical_patch_id']}  水体 {float(water_fraction(row)):.2%}"
                          f"  黑边 {black / total:.2%}", flush=True)
    print(f"检查 {checked} 个已排除 Patch；符合水体 ≥ {args.min_water_ratio:.2%}"
          f" 且黑边 ≤ {args.max_black_ratio:.2%} 的共 {len(wanted)} 个")
    if args.dry_run:
        print("预览完成，未修改数据；实际恢复时将自动跳过与已纳入 Patch 重复/空间重叠的项")
    elif wanted:
        restored, skipped = restore_without_conflicts(store, args.workspace, args.region, wanted)
        print(f"已恢复 {restored} 个 Patch，跳过冲突 {skipped} 个；刷新网页查看。‘我的选择’列表不变。")
    else:
        print("没有需要恢复的 Patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
