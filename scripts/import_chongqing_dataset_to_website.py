#!/usr/bin/env python3
"""Add the strict Chongqing candidate manifest to the existing Lakes workspace.

This importer is additive: it refuses to overwrite an existing Chongqing
workspace manifest and never removes source data. Preview files are symlinks
to the immutable research-dataset previews so the website can display the
strict labels without rebuilding them with its older patch generator.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path


SOURCE_ROOT = Path(
    "/home/lake/lakes/experiments/dataset_builds/"
    "chongqing_full_v1_20261013/dataset"
)
WORKSPACE_ROOT = Path(
    "/home/lake/lakes/data/workspaces/default/logical_patches/chongqing"
)
MISSING_SOURCE = SOURCE_ROOT / "strict_npz_only_source_missing"


def _bounds(value: str) -> tuple[str, str, str, str]:
    values = json.loads(value)
    return tuple(str(float(item)) for item in values)


def _link(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == source:
            return
        raise FileExistsError(target)
    target.symlink_to(source)


def main() -> None:
    source_manifest = SOURCE_ROOT / "manifest.csv"
    output_manifest = WORKSPACE_ROOT / "manifest.csv"
    if output_manifest.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {output_manifest}")
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    with source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("source manifest is empty")

    preview_dir = WORKSPACE_ROOT / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for source in source_rows:
        patch_id = source["patch_id"]
        preview = Path(source["preview_path"])
        _link(preview, preview_dir / f"{patch_id}.png")
        _link(preview, preview_dir / f"{patch_id}.base.png")
        left, bottom, right, top = _bounds(source["bounds"])
        valid_ratio = float(source.get("valid_ratio") or 0)
        valid_pixels = round(valid_ratio * int(source.get("patch_size") or 256) ** 2)
        rows.append(
            {
                "logical_patch_id": patch_id,
                "patch_id": patch_id,
                "source_patch_id": patch_id,
                "region": "chongqing",
                "site_id": source.get("site_id", ""),
                "site_name": source.get("site_id", ""),
                "sample_id": source.get("sample_id", ""),
                "image_index": "0",
                "image_path": str(MISSING_SOURCE),
                "label_path": str(MISSING_SOURCE),
                "npz_path": source.get("npz_path", ""),
                "preview_path": str(preview_dir / f"{patch_id}.png"),
                "preview_base_path": str(preview_dir / f"{patch_id}.base.png"),
                "row_off": source.get("row_off", "0"),
                "col_off": source.get("col_off", "0"),
                "logical_size": source.get("patch_size", "256"),
                "window_width": source.get("patch_size", "256"),
                "window_height": source.get("patch_size", "256"),
                "grid_stride": source.get("grid_stride", source.get("patch_size", "256")),
                "split": source.get("split", ""),
                "valid_ratio": source.get("valid_ratio", "0"),
                "valid_pixels": str(valid_pixels),
                "labeled_ratio": source.get("labeled_ratio", "0"),
                "water_ratio_valid": source.get("water_ratio_valid", "0"),
                "water_pixels": source.get("water_pixels", "0"),
                "ignore_pixels": source.get("ignore_pixels", "0"),
                "product_name": source.get("sample_id", ""),
                "product_date": source.get("date", ""),
                "date": source.get("date", ""),
                "period": source.get("period", ""),
                "label_method": source.get("label_method", ""),
                "label_source": source.get("label_method", ""),
                "label_sources": source.get("label_method", ""),
                "crs": source.get("crs", ""),
                "bounds_left": left,
                "bounds_bottom": bottom,
                "bounds_right": right,
                "bounds_top": top,
                "include": "false",
                "included": "false",
                "review_status": "unreviewed",
                "exclude_reason": "待人工审核",
            }
        )

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    fd, temporary = tempfile.mkstemp(
        prefix=".manifest.", dir=WORKSPACE_ROOT, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_manifest)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    print(json.dumps({"manifest": str(output_manifest), "patches": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
