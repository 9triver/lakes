#!/usr/bin/env python3
"""Export water segmentation training patches from recorded training samples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lakes_browser.region_config import DEFAULT_CONFIG_PATH, load_region_configs  # noqa: E402


REGIONS, DEFAULT_REGION_KEY = load_region_configs(DEFAULT_CONFIG_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--sample-id", action="append", default=None, help="Only export selected sample id(s).")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-valid-ratio", type=float, default=0.6)
    parser.add_argument("--min-water-pixels", type=int, default=1)
    parser.add_argument("--negative-ratio", type=float, default=0.25, help="Keep this fraction of valid patches with no water.")
    parser.add_argument("--all-touched", action="store_true", help="Rasterize polygons with all_touched=True.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=0, help="0 means write previews for every exported patch.")
    parser.add_argument("--preview-scale", type=int, default=2, help="Scale preview PNGs by this factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_training_patches(args)
    print(f"region={result['region']}")
    print(f"samples={result['samples']}")
    print(f"patches={result['patches']}")
    print(f"manifest={result['manifest']}")
    print(f"npz_dir={result['npz_dir']}")
    print(f"preview_dir={result['preview_dir']}")


def export_training_patches(args: argparse.Namespace) -> dict:
    region = REGIONS[args.region]
    output_dir = args.output_dir or region.processed_dir / "training_patches" / f"ps{args.patch_size}_st{args.stride}"
    previous_patch_state = read_patch_state(output_dir / "manifest.csv")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    (output_dir / "npz").mkdir(parents=True, exist_ok=True)
    (output_dir / "preview").mkdir(parents=True, exist_ok=True)

    samples = read_samples(region.training_samples)
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [row for row in samples if row.get("sample_id") in wanted]
    if not samples:
        raise SystemExit(f"no training samples found for region {region.key}")

    manifest_rows = []
    for row in samples:
        manifest_rows.extend(export_sample(row, output_dir, args))
    apply_patch_state(manifest_rows, previous_patch_state)

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    return {
        "region": region.key,
        "samples": len(samples),
        "patches": len(manifest_rows),
        "manifest": str(manifest_path),
        "npz_dir": str(output_dir / "npz"),
        "preview_dir": str(output_dir / "preview"),
    }


def read_samples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        table = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return []
    return table.to_dict("records")


def read_patch_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        table = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return {}
    state = {}
    for row in table.to_dict("records"):
        patch_id = row.get("patch_id")
        if not patch_id:
            continue
        state[patch_id] = {
            key: row.get(key, "")
            for key in ["include", "included", "patch_notes"]
            if row.get(key, "") != ""
        }
    return state


def apply_patch_state(rows: list[dict], previous: dict[str, dict]) -> None:
    for row in rows:
        state = previous.get(row.get("patch_id", ""))
        row["include"] = state.get("include") or state.get("included") or "true" if state else "true"
        row["patch_notes"] = state.get("patch_notes", "") if state else ""


def export_sample(row: dict, output_dir: Path, args: argparse.Namespace) -> list[dict]:
    sample_id = row.get("sample_id") or ""
    label_path = resolve_project_path(row.get("label_path") or "")
    image_paths = [resolve_project_path(item) for item in split_semicolon(row.get("tci_path") or "")]
    if not image_paths:
        print(f"skip {sample_id}: no image path")
        return []
    if not label_path.exists():
        print(f"skip {sample_id}: missing label {label_path}")
        return []
    rows = []
    for image_index, image_path in enumerate(image_paths):
        if not image_path.exists():
            print(f"skip {sample_id} image {image_index}: missing image {image_path}")
            continue
        rows.extend(export_sample_image(row, image_path, label_path, output_dir, args, image_index, len(image_paths)))
    print(f"{sample_id}: patches={len(rows)}")
    return rows


def export_sample_image(
    row: dict,
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    image_index: int,
    image_count: int,
) -> list[dict]:
    sample_id = row.get("sample_id") or ""
    with rasterio.open(image_path) as src:
        image = src.read()
        valid = valid_mask(image, src.nodata)
        label = rasterize_label(label_path, src, all_touched=args.all_touched)
        target = np.where(valid, label, 255).astype("uint8")
        windows = patch_windows(src.width, src.height, args.patch_size, args.stride)

        rows = []
        negative_index = 0
        for window in windows:
            y0 = int(window.row_off)
            x0 = int(window.col_off)
            y1 = y0 + args.patch_size
            x1 = x0 + args.patch_size
            valid_patch = valid[y0:y1, x0:x1]
            valid_pixels = int(valid_patch.sum())
            valid_ratio = valid_pixels / float(args.patch_size * args.patch_size)
            if valid_ratio < args.min_valid_ratio:
                continue
            mask_patch = target[y0:y1, x0:x1]
            water_pixels = int(np.count_nonzero(mask_patch == 1))
            if water_pixels < args.min_water_pixels:
                if args.negative_ratio <= 0:
                    continue
                keep_negative = (negative_index % max(1, round(1 / args.negative_ratio))) == 0
                negative_index += 1
                if not keep_negative:
                    continue

            image_patch = image[:, y0:y1, x0:x1]
            image_part = f"_i{image_index:02d}" if image_count > 1 else ""
            patch_id = f"{sample_id}{image_part}_r{y0:05d}_c{x0:05d}"
            npz_path = output_dir / "npz" / f"{patch_id}.npz"
            np.savez_compressed(
                npz_path,
                image=image_patch,
                mask=mask_patch,
                valid=valid_patch.astype("uint8"),
            )
            preview_path = output_dir / "preview" / f"{patch_id}.png"
            if args.preview_limit <= 0 or len(rows) < args.preview_limit:
                write_preview(preview_path, image_patch, mask_patch, valid_patch, scale=max(1, args.preview_scale))
            else:
                preview_path = Path("")

            bounds = rasterio.windows.bounds(window, src.transform)
            rows.append(
                {
                    "patch_id": patch_id,
                    "sample_id": sample_id,
                    "lake_id": row.get("lake_id", ""),
                    "lake_name": row.get("lake_name", ""),
                    "region": args.region,
                    "image_index": image_index,
                    "image_path": display_path(image_path),
                    "label_path": display_path(label_path),
                    "npz_path": display_path(npz_path),
                    "preview_path": display_path(preview_path) if str(preview_path) else "",
                    "row_off": y0,
                    "col_off": x0,
                    "patch_size": args.patch_size,
                    "valid_ratio": f"{valid_ratio:.6f}",
                    "valid_pixels": valid_pixels,
                    "water_pixels": water_pixels,
                    "water_ratio_valid": f"{water_pixels / valid_pixels:.6f}" if valid_pixels else "0",
                    "ignore_pixels": int(np.count_nonzero(mask_patch == 255)),
                    "bounds_left": bounds[0],
                    "bounds_bottom": bounds[1],
                    "bounds_right": bounds[2],
                    "bounds_top": bounds[3],
                    "crs": str(src.crs or ""),
                    "product_name": row.get("product_name", ""),
                    "imagery_asset_type": row.get("imagery_asset_type", ""),
                    "context_sources": row.get("context_sources", ""),
                }
            )
        if image_count > 1:
            print(f"{sample_id} image {image_index}: patches={len(rows)}")
        return rows


def valid_mask(image: np.ndarray, nodata) -> np.ndarray:
    valid = np.any(image != 0, axis=0)
    if nodata is not None:
        valid &= np.all(image != nodata, axis=0)
    return valid


def rasterize_label(label_path: Path, src, all_touched: bool = False) -> np.ndarray:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    features = payload.get("features") if payload.get("type") == "FeatureCollection" else [payload]
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) if src.crs else None
    shapes = []
    for feature in features or []:
        geom_json = feature.get("geometry")
        if not geom_json:
            continue
        geom = make_valid(shape(geom_json))
        if geom.is_empty:
            continue
        if transformer and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geom = shapely_transform(transformer.transform, geom)
        shapes.append((geom, 1))
    if not shapes:
        return np.zeros((src.height, src.width), dtype="uint8")
    return rasterize(
        shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )


def patch_windows(width: int, height: int, patch_size: int, stride: int) -> list[Window]:
    xs = offsets(width, patch_size, stride)
    ys = offsets(height, patch_size, stride)
    return [Window(x, y, patch_size, patch_size) for y in ys for x in xs]


def offsets(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    values = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not values or values[-1] != last:
        values.append(last)
    return values


def write_preview(path: Path, image: np.ndarray, mask: np.ndarray, valid: np.ndarray, scale: int = 1) -> None:
    rgb = stretch_rgb(image)
    rgb[~valid] = np.array([34, 34, 34], dtype="uint8")
    water = mask == 1
    rgb[water] = (rgb[water].astype("uint16") * 35 // 100 + np.array([0, 120, 255], dtype="uint16") * 65 // 100).astype("uint8")
    image_out = Image.fromarray(rgb)
    if scale > 1:
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        image_out = image_out.resize((image_out.width * scale, image_out.height * scale), resampling)
    image_out.save(path)


def stretch_rgb(image: np.ndarray) -> np.ndarray:
    if image.shape[0] >= 3:
        arr = np.stack([image[2], image[1], image[0]], axis=-1)
    else:
        arr = np.repeat(image[0][:, :, None], 3, axis=2)
    arr = arr.astype("float32")
    out = np.zeros_like(arr, dtype="uint8")
    for band in range(3):
        values = arr[:, :, band]
        nonzero = values[values != 0]
        if nonzero.size:
            lo, hi = np.percentile(nonzero, [2, 98])
        else:
            lo, hi = 0, 1
        if hi <= lo:
            hi = lo + 1
        out[:, :, band] = np.clip((values - lo) * 255 / (hi - lo), 0, 255).astype("uint8")
    return out


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


if __name__ == "__main__":
    main()
