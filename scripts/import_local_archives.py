#!/usr/bin/env python3
"""Import contributor archives into the regional local-imagery layout.

The importer is resumable: an interrupted archive keeps its staging directory
and is re-opened on the next run. Source archives are removed only after a
successful extraction and directory merge when ``--delete-source`` is used.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("/mnt/data")
STAGE_ROOT = ARCHIVE_ROOT / ".lakes_import_stage"

REGION_TOKENS = {
    "chongqing": "chongqing",
    "重庆": "chongqing",
    "guangdong": "guangdong",
    "guangxi": "guangxi",
    "广西": "guangxi",
    "guizhou": "guizhou",
    "贵州": "guizhou",
    "hunan": "hunan",
    "jiangsu": "jiangsu",
    "sichuan": "sichuan",
    "xizang": "xizang",
    "西藏": "xizang",
    "yunnan": "yunnan",
}
ARCHIVE_REGIONS = {
    "20260603_du.rar": "guizhou",
}
SKIP_DUPLICATES = {
    "sichuan20260607_duchenggong.rar",
    "xizang20260607_du.rar",
}


def archive_region(archive: Path) -> str | None:
    name = archive.name.lower()
    if region := ARCHIVE_REGIONS.get(name):
        return region
    if name.startswith("n6") or "n6_" in name:
        return "guangdong"
    for token, region in REGION_TOKENS.items():
        if token.lower() in name:
            return region
    return None


def is_site_directory(path: Path) -> bool:
    return path.is_dir() and path.name.removesuffix(".shp").isdigit()


def normalized_site_name(name: str) -> str:
    return name.removesuffix(".shp")


def merge_tree(source: Path, target: Path) -> int:
    """Merge a staged site without overwriting different existing files."""
    conflicts = 0
    target.mkdir(parents=True, exist_ok=True)
    for item in list(source.iterdir()):
        destination = target / item.name
        if item.is_dir():
            conflicts += merge_tree(item, destination)
            if item.exists() and not any(item.iterdir()):
                item.rmdir()
        elif destination.exists():
            if destination.is_file() and destination.stat().st_size == item.stat().st_size:
                item.unlink()
            else:
                print(
                    f"CONFLICT {destination} "
                    f"existing={destination.stat().st_size} "
                    f"incoming={item.stat().st_size}",
                    flush=True,
                )
                conflicts += 1
        else:
            shutil.move(str(item), str(destination))
    return conflicts


def extract_archive(archive: Path, stage: Path) -> bool:
    if archive.suffix.lower() == ".rar":
        command = ["unrar", "x", "-inul", "-p-", "-o+", str(archive), str(stage) + "/"]
    else:
        command = ["7z", "x", "-y", "-bd", "-bb0", "-aoa", str(archive), f"-o{stage}"]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode == 0


def import_archive(archive: Path, region: str, *, delete_source: bool) -> bool:
    target_root = PROJECT_ROOT / "data" / "regions" / region / "raw" / "local_imagery"
    stage = STAGE_ROOT / archive.name
    stage.mkdir(parents=True, exist_ok=True)
    print(f"EXTRACT {archive.name} -> {region}", flush=True)
    started = time.monotonic()
    if not extract_archive(archive, stage):
        print(f"FAILED {archive.name}: extractor failed; stage kept at {stage}", flush=True)
        return False

    site_dirs = sorted(
        (path for path in stage.rglob("*") if is_site_directory(path)),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    if not site_dirs:
        print(f"FAILED {archive.name}: no numeric site directories; stage kept", flush=True)
        return False

    moved = 0
    merged = 0
    conflicts = 0
    target_root.mkdir(parents=True, exist_ok=True)
    for source_site in site_dirs:
        if any(source_site in other.parents for other in site_dirs if other != source_site):
            continue
        existing = next(
            (
                path
                for path in target_root.iterdir()
                if is_site_directory(path)
                and normalized_site_name(path.name) == normalized_site_name(source_site.name)
            ),
            None,
        )
        if existing is None:
            shutil.move(
                str(source_site),
                str(target_root / normalized_site_name(source_site.name)),
            )
            moved += 1
        else:
            conflicts += merge_tree(source_site, existing)
            if source_site.exists() and not any(source_site.iterdir()):
                source_site.rmdir()
            merged += 1

    if conflicts:
        print(f"FAILED {archive.name}: {conflicts} conflicts; stage kept", flush=True)
        return False

    for path in stage.rglob("*.mxd"):
        if path.is_file():
            path.unlink()
    shutil.rmtree(stage)
    if delete_source:
        archive.unlink()
    elapsed = time.monotonic() - started
    site_count = sum(1 for path in target_root.iterdir() if is_site_directory(path))
    nonzero_images = sum(
        1
        for path in target_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".img", ".tif", ".tiff"}
        and path.stat().st_size > 0
    )
    print(
        f"DONE {archive.name} sites_added={moved} sites_merged={merged} "
        f"total_sites={site_count} nonzero_rasters={nonzero_images} "
        f"elapsed={elapsed:.0f}s source_deleted={'yes' if delete_source else 'no'}",
        flush=True,
    )
    return True


def rebuild_metadata(region: str) -> bool:
    command = [
        sys.executable,
        "scripts/build_site_metadata.py",
        "--region",
        region,
        "--workers",
        "4",
    ]
    print(f"REBUILD_METADATA {region}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        print(f"METADATA_FAILED {region} exit={result.returncode}", flush=True)
        return False
    print(f"METADATA_DONE {region}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="delete each archive after extraction and merge succeed",
    )
    parser.add_argument(
        "--rebuild-metadata",
        action="store_true",
        help="rebuild site metadata once after each region finishes",
    )
    args = parser.parse_args()

    archives = sorted(
        path
        for parent in ARCHIVE_ROOT.iterdir()
        if parent.is_dir()
        for path in parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".rar", ".zip"}
        and not path.name.startswith("._")
    )
    selected: dict[str, list[Path]] = {}
    for archive in archives:
        if archive.name in SKIP_DUPLICATES:
            print(f"SKIP_DUPLICATE {archive.name}", flush=True)
            continue
        region = archive_region(archive)
        if region is None:
            print(f"HOLD_UNKNOWN {archive.name}", flush=True)
            continue
        selected.setdefault(region, []).append(archive)

    for region in sorted(selected):
        print(f"BEGIN_REGION {region}", flush=True)
        for archive in selected[region]:
            if not import_archive(archive, region, delete_source=args.delete_source):
                return 2
        if args.rebuild_metadata and not rebuild_metadata(region):
            return 3
    print("IMPORT_BATCH_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
