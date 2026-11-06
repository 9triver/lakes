#!/usr/bin/env python3
"""List sites without Workspace patches and optionally generate the first N.

List: python scripts/generate_missing_site_training_data.py --region shaanxi
Preview: add --first 2 --dry-run
Generate: add --first 2
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pyogrio

from lake_workbench.imagery.inventory_sources import load_user_tci_rows
from lake_workbench.regions.config import load_region_configs
from lake_workbench.utils import read_csv_records
from lake_workbench.workspaces import WorkspaceStore


def missing_sites(region, store: WorkspaceStore, workspace: str) -> list[dict]:
    """Read metadata and manifests without initializing a mutable SiteCatalog."""
    if not region.site_metadata.is_file():
        raise FileNotFoundError(f"该省站点元数据不存在: {region.site_metadata}")
    sites = pyogrio.read_dataframe(
        region.site_metadata, layer="sites", columns=["site_id"], read_geometry=False
    )
    patch_rows = read_csv_records(store.workspace_logical_patch_manifest(workspace, region.key))
    generated = {row["site_id"] for row in patch_rows if row.get("site_id")}
    samples = read_csv_records(store.workspace_training_samples(workspace, region.key))
    recorded = {row["site_id"] for row in samples if row.get("site_id")}
    imagery = Counter(
        row["site_id"]
        for rows in load_user_tci_rows(region).values()
        for row in rows
        if row.get("source") in {"local_img", "local_imagery"}
        and row.get("tci_path") and row["tci_path"].is_file()
    )
    pending = [
        {"site_id": str(site_id), "images": imagery[str(site_id)],
         "sample_exists": str(site_id) in recorded}
        for site_id in set(sites["site_id"].astype(str))
        if str(site_id) not in generated
    ]
    def order(item):
        suffix = item["site_id"].rsplit("_", 1)[-1]
        return (0, int(suffix), item["site_id"]) if suffix.isdigit() else (1, 0, item["site_id"])
    return sorted(pending, key=order)


def main(argv: list[str] | None = None) -> int:
    regions, _ = load_region_configs()
    parser = argparse.ArgumentParser(description="列出尚无训练 Patch 的 Site，并调用现有脚本生成前 N 个")
    parser.add_argument("--region", required=True, choices=list(regions))
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--all-sites", action="store_true", help="交给分阶段脚本处理所有有影像Site，自动复用种子和完整Patch")
    parser.add_argument("--stage", choices=["all","seeds","patches"], default="all")
    parser.add_argument("--batch-sites", type=int, default=20)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--resume", type=Path, help="恢复任务原始选择，不能同时使用--first或--all-sites")
    parser.add_argument("--build-dataset", action="store_true", help="完成Patch后构建训练NPZ缓存")
    parser.add_argument("--first", type=int, help="生成前 N 个有可用本地影像的待生成 Site；省略时仅列出")
    parser.add_argument("--dry-run", action="store_true", help="仅预览选择的 Site 和生成命令")
    parser.add_argument("--max-dimension", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--continue-on-error", action="store_true", help="生成时单个 Site 失败后继续")
    args = parser.parse_args(argv)
    if args.resume and (args.first is not None or args.all_sites):
        parser.error("--resume不能同时使用--first/--all-sites")
    if args.all_sites and args.first is not None:
        parser.error("--all-sites不能同时使用--first")
    if args.batch_sites<=0:
        parser.error("--batch-sites必须为正整数")
    if args.first is not None and args.first <= 0:
        parser.error("--first 必须为正整数")
    if not 256 <= args.max_dimension <= 4096 or args.patch_size <= 0 or args.stride <= 0:
        parser.error("max-dimension 必须在 256–4096 之间，patch-size 和 stride 必须为正整数")
    if args.all_sites or args.resume:
        generator = Path(__file__).resolve().with_name("generate_spectral_osm_training_data.py")
        command = [sys.executable, str(generator), "--region", args.region, "--workspace", args.workspace,
                   "--stage", args.stage, "--max-dimension", str(args.max_dimension),
                   "--patch-size", str(args.patch_size), "--stride", str(args.stride),
                   "--batch-sites", str(args.batch_sites)]
        if args.all_sites: command.append("--all-sites")
        if args.resume: command.extend(["--resume", str(args.resume)])
        if args.state_file: command.extend(["--state-file", str(args.state_file)])
        for name in ["continue_on_error", "dry_run", "build_dataset"]:
            if getattr(args, name): command.append("--" + name.replace("_", "-"))
        print("生成命令：" + shlex.join(command), flush=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(generator.parent.parent / "src"), environment.get("PYTHONPATH", "")]))
        return subprocess.run(command, env=environment, check=False).returncode
    store = WorkspaceStore(regions)
    try:
        store.get(args.workspace, allow_archived=False)
        pending = missing_sites(regions[args.region], store, args.workspace)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    eligible = [item for item in pending if item["images"]]
    print(f"{regions[args.region].name} / 工作区 {args.workspace}：尚无 Patch 的 Site 共 {len(pending)} 个，"
          f"其中 {len(eligible)} 个有可用本地影像", flush=True)
    for index, item in enumerate(pending, 1):
        status = f"可生成（{item['images']} 个影像期次，使用当前活动期次）" if item["images"] else "无可用本地影像，跳过"
        if item["sample_exists"]:
            status += "；已存样本但尚无 Patch"
        print(f"{index:>4}. {item['site_id']}  {status}", flush=True)
    if args.first is None:
        print("仅查看；添加 --first N 可生成前 N 个有影像的 Site。", flush=True)
        return 0
    selected = eligible[:args.first]
    if not selected:
        print("没有可生成的 Site。", flush=True)
        return 0
    if len(selected) < args.first:
        print(f"可生成 Site 不足 {args.first} 个，将处理现有的 {len(selected)} 个。", flush=True)
    generator = Path(__file__).resolve().with_name("generate_spectral_osm_training_data.py")
    command = [sys.executable, str(generator), "--region", args.region,
               "--workspace", args.workspace, "--max-dimension", str(args.max_dimension),
               "--patch-size", str(args.patch_size), "--stride", str(args.stride)]
    command.extend(["--stage", args.stage, "--batch-sites", str(args.batch_sites)])
    if args.state_file: command.extend(["--state-file", str(args.state_file)])
    if args.build_dataset: command.append("--build-dataset")
    if args.continue_on_error:
        command.append("--continue-on-error")
    command.extend(["--sites", *(item["site_id"] for item in selected)])
    print("本次选择：" + ", ".join(item["site_id"] for item in selected), flush=True)
    print("生成命令：" + shlex.join(command), flush=True)
    if args.dry_run:
        print("预览完成，未执行生成。", flush=True)
        return 0
    environment = os.environ.copy()
    source_root = str(generator.parent.parent / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, environment.get("PYTHONPATH", "")]))
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
