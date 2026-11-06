#!/usr/bin/env python3
"""Rank generated training sites by mean per-patch water fraction.

Each patch has equal weight. Fractions use water_pixels / valid_pixels,
not water_pixels / full patch size. No training data is modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction

from lake_workbench.regions.config import load_region_configs
from lake_workbench.utils import read_csv_records
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.workspaces.membership import patch_is_included


def rank_sites(rows: list[dict], status: str = "included") -> tuple[list[dict], int]:
    groups: dict[str, list[Fraction]] = defaultdict(list)
    skipped = 0
    for row in rows:
        included = patch_is_included(row)
        if (status == "included" and not included) or (status == "excluded" and included):
            continue
        site_id = row.get("site_id", "")
        if not site_id:
            raise ValueError(f"Patch 缺少 site_id: {row.get('logical_patch_id')}")
        try:
            valid = int(row["valid_pixels"])
            water = int(row["water_pixels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Patch 像元统计无效: {row.get('logical_patch_id')}") from exc
        if valid < 0 or not 0 <= water <= valid:
            raise ValueError(f"Patch 像元统计无效: {row.get('logical_patch_id')}")
        if valid == 0:
            skipped += 1
            continue
        groups[site_id].append(Fraction(water, valid))
    result = [
        {"site_id": site_id, "patches": len(ratios),
         "mean_water_ratio": sum(ratios, Fraction(0)) / len(ratios)}
        for site_id, ratios in groups.items()
    ]
    result.sort(key=lambda item: (-item["mean_water_ratio"], item["site_id"]))
    return result, skipped


def main(argv: list[str] | None = None) -> int:
    regions, _ = load_region_configs()
    parser = argparse.ArgumentParser(description="按已筛选 Patch 的平均水体占比，对 Site 降序排序（只读）")
    parser.add_argument("--region", required=True, choices=[*regions, "all"], help="省份键，all 表示全部省份")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--sites", nargs="+", help="仅统计指定 Site")
    parser.add_argument("--status", choices=["included", "excluded", "all"], default="included",
                        help="默认 included，仅统计已纳入 Patch；excluded 为已排除；all 为全部")
    parser.add_argument("--top", type=int, help="仅显示排名前 N 个 Site")
    args = parser.parse_args(argv)
    if args.top is not None and args.top <= 0:
        parser.error("--top 必须为正整数")
    store = WorkspaceStore(regions)
    store.get(args.workspace)
    selected_regions = list(regions) if args.region == "all" else [args.region]
    rows = []
    for key in selected_regions:
        manifest = store.workspace_logical_patch_manifest(args.workspace, key)
        if not manifest.is_file():
            if args.region != "all":
                parser.error(f"该省尚无工作区 Patch 清单: {manifest}")
            continue
        rows.extend(read_csv_records(manifest))
    if args.sites:
        sites = set(args.sites)
        rows = [row for row in rows if row.get("site_id") in sites]
    try:
        ranked, skipped = rank_sites(rows, args.status)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"工作区 {args.workspace} / 区域 {args.region} / Patch 状态 {args.status}")
    print("平均水体占比 = 各 Patch 的（水体像元数 ÷ 有效像元数）的算术平均值")
    print(f"共 {len(ranked)} 个 Site，{sum(item['patches'] for item in ranked)} 个有效 Patch")
    if skipped:
        print(f"已跳过 {skipped} 个有效像元为 0 的 Patch")
    print(f"{'排名':<6} {'Site':<28} {'Patch数':>8} {'平均水体占比':>12}")
    for index, item in enumerate(ranked[:args.top], 1):
        print(f"{index:<8} {item['site_id']:<28} {item['patches']:>8} "
              f"{float(item['mean_water_ratio']):>14.4%}")
    if not ranked:
        print("没有符合条件的 Site")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
