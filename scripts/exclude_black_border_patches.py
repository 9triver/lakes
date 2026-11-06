#!/usr/bin/env python3
"""Inspect invalid imagery fractions and exclude Workspace patches above a limit.

Example: python scripts/exclude_black_border_patches.py --region shaanxi \
    --max-black-ratio 0.2 --dry-run
Black border means invalid source pixels, not visually dark water or shadows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import math

import rasterio
from rasterio.windows import Window

from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.regions.config import load_region_configs
from lake_workbench.utils import read_csv_records, resolve_data_path
from lake_workbench.workspaces import WorkspaceStore


def black_border_counts(source, row: dict) -> tuple[int, int]:
    """Count invalid pixels across the full original logical-patch window."""
    size = int(row["logical_size"])
    row_off, col_off = int(row["row_off"]), int(row["col_off"])
    if size <= 0 or row_off < 0 or col_off < 0:
        raise ValueError(f"无效 Patch 窗口: {row.get('logical_patch_id')}")
    window = Window(col_off, row_off, size, size)
    image = source.read(window=window, boundless=True, fill_value=0)
    masks = source.read_masks(window=window, boundless=True)
    valid = valid_pixel_mask(image, source.nodatavals, masks)
    total = size * size
    return total - int(valid.sum()), total


def main(argv: list[str] | None = None) -> int:
    regions, _ = load_region_configs()
    parser = argparse.ArgumentParser(description="检测 Patch 黑边/无效影像占比，超阈值则标记为已排除")
    parser.add_argument("--region", required=True, choices=list(regions))
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--sites", nargs="+", help="仅检查指定 Site，省略则检查全省")
    parser.add_argument("--max-black-ratio", type=float, default=0.2, help="允许的最大占比，0.2 表示 20%%")
    parser.add_argument("--dry-run", action="store_true", help="只检查和显示结果，不修改状态")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_black_ratio) or not 0 <= args.max_black_ratio <= 1:
        parser.error("--max-black-ratio 必须在 0 到 1 之间")
    store = WorkspaceStore(regions)
    store.get(args.workspace, allow_archived=False)
    region = regions[args.region]
    manifest = store.workspace_logical_patch_manifest(args.workspace, args.region)
    if not manifest.is_file():
        parser.error(f"Patch 清单不存在: {manifest}")
    sites = set(args.sites or [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv_records(manifest):
        if not sites or row.get("site_id") in sites:
            if not row.get("image_path"):
                parser.error(f"Patch 缺少影像路径: {row.get('logical_patch_id')}")
            grouped[row["image_path"]].append(row)
    wanted = []
    checked = 0
    threshold = Fraction(str(args.max_black_ratio))
    # Complete all reads before making any membership changes.
    for image_path, rows in grouped.items():
        with rasterio.open(resolve_data_path(image_path, region)) as source:
            for row in rows:
                black, total = black_border_counts(source, row)
                exceeds = Fraction(black, total) > threshold
                checked += 1
                print(f"{row['logical_patch_id']}  黑边 {black / total:.2%} "
                      f"({black}/{total})  {'超阈值' if exceeds else '通过'}", flush=True)
                if exceeds:
                    wanted.append(row["logical_patch_id"])
    print(f"检查 {checked} 个 Patch，黑边超过 {args.max_black_ratio:.2%} 的共 {len(wanted)} 个")
    if args.dry_run:
        print("预览完成，未修改数据")
    elif wanted:
        result = store.update_members(args.workspace, args.region, wanted, "exclude")
        print(f"新增排除 {result['updated']} 个 Patch，刷新网页可查看；原始文件保留")
        print("若这些 Patch 已加入‘我的选择’，该列表仍会保留它们，请在网页单独移出。")
    else:
        print("无需排除")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
