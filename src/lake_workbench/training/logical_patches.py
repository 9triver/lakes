"""Canonical logical patches and reproducible derived training datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from io import BytesIO
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.regions.config import RegionConfig
from lake_workbench.utils import display_path, read_csv_records, resolve_data_path, truthy_flag

if TYPE_CHECKING:
    from lake_workbench.profiles import ProfileStore


LOGICAL_PATCH_SIZE = 512
DATASET_CONFIG_PATH = PROJECT_ROOT / "config" / "training_datasets.toml"


def dataset_configs(path: Path = DATASET_CONFIG_PATH) -> dict[str, dict]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return {key: {"id": key, **value} for key, value in payload.get("datasets", {}).items()}


def logical_patch_signature(region: RegionConfig, config: dict) -> str:
    digest = hashlib.sha256()
    digest.update(region.logical_patch_manifest.read_bytes())
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def logical_patch_id(sample_id: str, image_index: int, row_off: int, col_off: int) -> str:
    return f"{sample_id}_i{image_index:02d}_lr{row_off:05d}_lc{col_off:05d}"


def grid_offsets(length: int) -> list[int]:
    return list(range(0, max(1, length), LOGICAL_PATCH_SIZE))


def build_logical_patches(
    region: RegionConfig,
    output_dir: Path | None = None,
    sample_ids: set[str] | None = None,
) -> dict:
    logical_dir = output_dir or region.logical_patch_dir
    manifest_path = logical_dir / "manifest.csv"
    previous = {
        row.get("logical_patch_id", ""): row
        for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id")
    }
    samples = read_csv_records(region.training_samples)
    if sample_ids is not None:
        samples = [row for row in samples if row.get("sample_id") in sample_ids]
    if not samples:
        return {"region": region.key, "samples": 0, "patches": 0, "manifest": ""}
    parent = logical_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="logical_patches.", dir=parent))
    preview_dir = temp_dir / "preview"
    preview_dir.mkdir(parents=True)
    rows: list[dict] = []
    try:
        for sample in samples:
            rows.extend(_build_sample_logical_patches(region, sample, previous, preview_dir, logical_dir))
        rows.sort(key=lambda row: (row["sample_id"], int(row["image_index"]), int(row["row_off"]), int(row["col_off"])))
        _write_csv(temp_dir / "manifest.csv", rows)
        _replace_directory(temp_dir, logical_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"region": region.key, "samples": len(samples), "patches": len(rows), "manifest": display_path(manifest_path)}


def _build_sample_logical_patches(
    region: RegionConfig,
    sample: dict,
    previous: dict[str, dict],
    preview_dir: Path,
    logical_dir: Path,
) -> list[dict]:
    sample_id = sample.get("sample_id", "")
    label_path = resolve_data_path(sample.get("label_path", ""), region)
    image_paths = [resolve_data_path(value, region) for value in str(sample.get("tci_path", "")).split(";") if value.strip()]
    if not sample_id or not label_path.exists():
        return []
    products = _split_values(sample.get("products") or sample.get("product_name"))
    dates = _split_values(sample.get("product_dates") or sample.get("product_date"))
    result: list[dict] = []
    for image_index, image_path in enumerate(image_paths):
        if not image_path.exists():
            continue
        with rasterio.open(image_path) as src:
            image = src.read()
            valid = _valid_mask(image, src.nodata)
            label = _rasterize_label(label_path, src)
            target = np.where(valid, label, 255).astype("uint8")
            for row_off in grid_offsets(src.height):
                for col_off in grid_offsets(src.width):
                    image_patch = _padded_crop(image, row_off, col_off, fill=0)
                    valid_patch = _padded_crop(valid, row_off, col_off, fill=False)
                    if not np.any(valid_patch):
                        continue
                    mask_patch = _padded_crop(target, row_off, col_off, fill=255)
                    patch_id = logical_patch_id(sample_id, image_index, row_off, col_off)
                    preview_path = preview_dir / f"{patch_id}.png"
                    _write_preview(preview_path, image_patch, mask_patch, valid_patch)
                    bounds = _grid_cell_bounds(src.transform, row_off, col_off)
                    valid_pixels = int(valid_patch.sum())
                    water_pixels = int(np.count_nonzero(mask_patch == 1))
                    old = previous.get(patch_id, {})
                    result.append(
                        {
                            "logical_patch_id": patch_id,
                            "sample_id": sample_id,
                            "site_id": sample.get("site_id", ""),
                            "site_name": sample.get("site_name", ""),
                            "region": region.key,
                            "image_index": image_index,
                            "image_path": display_path(image_path),
                            "label_path": display_path(label_path),
                            "preview_path": display_path(logical_dir / "preview" / preview_path.name),
                            "row_off": row_off,
                            "col_off": col_off,
                            "logical_size": LOGICAL_PATCH_SIZE,
                            "valid_ratio": f"{valid_pixels / float(LOGICAL_PATCH_SIZE**2):.6f}",
                            "valid_pixels": valid_pixels,
                            "water_pixels": water_pixels,
                            "water_ratio_valid": f"{water_pixels / valid_pixels:.6f}",
                            "ignore_pixels": int(np.count_nonzero(mask_patch == 255)),
                            "bounds_left": bounds[0],
                            "bounds_bottom": bounds[1],
                            "bounds_right": bounds[2],
                            "bounds_top": bounds[3],
                            "crs": str(src.crs or ""),
                            "product_name": _value_at(products, image_index),
                            "product_date": _value_at(dates, image_index),
                            "label_source": sample.get("label_source", ""),
                            "label_sources": sample.get("context_sources") or sample.get("label_source", ""),
                            "label_scope": sample.get("label_scope", ""),
                            "mask_policy": sample.get("mask_policy", ""),
                            "include": "true" if truthy_flag(old.get("include"), default=True) else "false",
                            "patch_notes": old.get("patch_notes", ""),
                        }
                    )
    return result


def build_training_dataset(region: RegionConfig, config_id: str) -> dict:
    configs = dataset_configs()
    if config_id not in configs:
        raise KeyError(f"unknown training dataset config: {config_id}")
    config = configs[config_id]
    logical_rows = read_csv_records(region.logical_patch_manifest)
    if not logical_rows:
        raise FileNotFoundError(f"logical patch manifest is empty: {display_path(region.logical_patch_manifest)}")
    output_dir = region.training_dataset_dir / config_id
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{config_id}.", dir=output_dir.parent))
    npz_dir = temp_dir / "npz"
    npz_dir.mkdir(parents=True)
    output_rows: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in logical_rows:
        if truthy_flag(row.get("include"), default=True):
            grouped[(row.get("image_path", ""), row.get("label_path", ""))].append(row)
    try:
        for (image_text, label_text), rows in grouped.items():
            image_path = resolve_data_path(image_text, region)
            label_path = resolve_data_path(label_text, region)
            if not image_path.exists() or not label_path.exists():
                continue
            with rasterio.open(image_path) as src:
                image = src.read()
                valid = _valid_mask(image, src.nodata)
                label = _rasterize_label(label_path, src)
                target = np.where(valid, label, 255).astype("uint8")
                for logical in rows:
                    actual = _derive_patch(image, target, valid, logical, int(config["output_size"]))
                    if not _eligible_actual_patch(actual, logical["logical_patch_id"], config):
                        continue
                    npz_path = npz_dir / f"{logical['logical_patch_id']}.npz"
                    np.savez_compressed(npz_path, image=actual["image"], mask=actual["mask"], valid=actual["valid"].astype("uint8"))
                    output_rows.append(
                        {
                            **logical,
                            "patch_id": logical["logical_patch_id"],
                            "dataset_config_id": config_id,
                            "npz_path": display_path(output_dir / "npz" / npz_path.name),
                            "patch_size": config["output_size"],
                            "valid_ratio": f"{actual['valid_ratio']:.6f}",
                            "valid_pixels": actual["valid_pixels"],
                            "water_pixels": actual["water_pixels"],
                            "water_ratio_valid": f"{actual['water_ratio_valid']:.6f}",
                            "ignore_pixels": actual["ignore_pixels"],
                            "include": "true",
                        }
                    )
        output_rows.sort(key=lambda row: row["logical_patch_id"])
        _write_csv(temp_dir / "manifest.csv", output_rows)
        signature = logical_patch_signature(region, config)
        (temp_dir / "build.json").write_text(
            json.dumps({"config": config, "signature": signature, "patches": len(output_rows)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _replace_directory(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"region": region.key, "config_id": config_id, "patches": len(output_rows), "manifest": display_path(output_dir / "manifest.csv")}


def training_dataset_status(region: RegionConfig, config_id: str) -> dict:
    config = dataset_configs()[config_id]
    output_dir = region.training_dataset_dir / config_id
    metadata_path = output_dir / "build.json"
    manifest_path = output_dir / "manifest.csv"
    if not region.logical_patch_manifest.exists():
        status = "missing_logical"
        metadata = {}
    elif not metadata_path.exists() or not manifest_path.exists():
        status = "missing"
        metadata = {}
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = "ready" if metadata.get("signature") == logical_patch_signature(region, config) else "stale"
        except (OSError, json.JSONDecodeError):
            status, metadata = "stale", {}
    return {
        "config": config,
        "config_id": config_id,
        "region": region.key,
        "status": status,
        "ready": status == "ready",
        "patches": int(metadata.get("patches") or 0),
        "manifest": display_path(manifest_path),
    }


def profile_dataset_signature(
    region: RegionConfig,
    config: dict,
    profile_store: "ProfileStore",
    profile_id: str,
) -> str:
    members = {patch_id for _region, patch_id in profile_store.members(profile_id, region.key)}
    manifest_path = profile_store.ensure_profile_logical_patch_manifest(profile_id, region.key)
    logical_rows = [
        row for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id") in members
    ]
    source_payload = {}
    for site_id in sorted({row.get("site_id", "") for row in logical_rows if row.get("site_id")}):
        source_payload[site_id] = [
            (item["id"], item["content_hash"])
            for item in profile_store.selected_variants(profile_id, site_id)
        ]
    payload = {
        "config": config,
        "logical_rows": logical_rows,
        "sources": source_payload,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_profile_training_dataset(
    region: RegionConfig,
    config_id: str,
    profile_store: "ProfileStore",
    profile_id: str,
) -> dict:
    profile_store.assert_trainable(profile_id)
    configs = dataset_configs()
    if config_id not in configs:
        raise KeyError(f"unknown training dataset config: {config_id}")
    config = configs[config_id]
    member_ids = {patch_id for _region, patch_id in profile_store.members(profile_id, region.key)}
    manifest_path = profile_store.ensure_profile_logical_patch_manifest(profile_id, region.key)
    logical_rows = [
        row for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id") in member_ids
    ]
    if not logical_rows:
        raise FileNotFoundError(f"Profile has no selected logical patches for {region.key}: {profile_id}")
    output_dir = profile_store.profile_dataset_dir(profile_id, region.key, config_id)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{config_id}.", dir=output_dir.parent))
    cache_dir = profile_store.profile_training_patch_cache_dir(profile_id, region.key, config_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in logical_rows:
        grouped[row.get("image_path", "")].append(row)
    try:
        for image_text, rows in grouped.items():
            image_path = resolve_data_path(image_text, region)
            if not image_path.exists():
                continue
            with rasterio.open(image_path) as src:
                image = src.read()
                valid = _valid_mask(image, src.nodata)
                targets: dict[str, np.ndarray] = {}
                for logical in rows:
                    site_id = logical.get("site_id", "")
                    variants = profile_store.selected_variants(profile_id, site_id)
                    if not variants:
                        raise ValueError(f"Profile has no resolved source variants for {site_id}")
                    source_signature = profile_store.source_signature(profile_id, site_id)
                    if source_signature not in targets:
                        targets[source_signature] = np.where(
                            valid,
                            _rasterize_source_variants(variants, src),
                            255,
                        ).astype("uint8")
                    actual = _derive_patch(
                        image,
                        targets[source_signature],
                        valid,
                        logical,
                        int(config["output_size"]),
                    )
                    patch_id = logical["logical_patch_id"]
                    if not _eligible_actual_patch(actual, patch_id, config):
                        continue
                    cache_name = f"{patch_id}-{source_signature[:16]}.npz"
                    cache_path = cache_dir / cache_name
                    if not cache_path.exists():
                        temporary = cache_path.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(
                                handle,
                                image=actual["image"],
                                mask=actual["mask"],
                                valid=actual["valid"].astype("uint8"),
                            )
                        os.replace(temporary, cache_path)
                    output_rows.append(
                        {
                            **logical,
                            "patch_id": patch_id,
                            "profile_id": profile_id,
                            "dataset_config_id": config_id,
                            "source_variant_ids": ",".join(item["id"] for item in variants),
                            "source_signature": source_signature,
                            "npz_path": display_path(cache_path),
                            "patch_size": config["output_size"],
                            "valid_ratio": f"{actual['valid_ratio']:.6f}",
                            "valid_pixels": actual["valid_pixels"],
                            "water_pixels": actual["water_pixels"],
                            "water_ratio_valid": f"{actual['water_ratio_valid']:.6f}",
                            "ignore_pixels": actual["ignore_pixels"],
                            "include": "true",
                        }
                    )
        if not output_rows:
            raise FileNotFoundError(f"Profile selection produced no eligible patches for {region.key}: {profile_id}")
        output_rows.sort(key=lambda row: row["logical_patch_id"])
        _write_csv(temp_dir / "manifest.csv", output_rows)
        signature = profile_dataset_signature(region, config, profile_store, profile_id)
        (temp_dir / "build.json").write_text(
            json.dumps(
                {"profile_id": profile_id, "config": config, "signature": signature, "patches": len(output_rows)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _replace_directory(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        "profile_id": profile_id,
        "region": region.key,
        "config_id": config_id,
        "patches": len(output_rows),
        "manifest": display_path(output_dir / "manifest.csv"),
    }


def profile_training_dataset_status(
    region: RegionConfig,
    config_id: str,
    profile_store: "ProfileStore",
    profile_id: str,
) -> dict:
    config = dataset_configs()[config_id]
    output_dir = profile_store.profile_dataset_dir(profile_id, region.key, config_id)
    metadata_path = output_dir / "build.json"
    manifest_path = output_dir / "manifest.csv"
    members = profile_store.members(profile_id, region.key)
    profile = profile_store.get(profile_id)
    if not members:
        status, metadata = "missing_selection", {}
    elif profile["status"] == "needs_resolution":
        status, metadata = "needs_resolution", {}
    elif not metadata_path.exists() or not manifest_path.exists():
        status, metadata = "missing", {}
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = "ready" if metadata.get("signature") == profile_dataset_signature(region, config, profile_store, profile_id) else "stale"
        except (OSError, json.JSONDecodeError):
            status, metadata = "stale", {}
    return {
        "profile_id": profile_id,
        "config": config,
        "config_id": config_id,
        "region": region.key,
        "status": status,
        "ready": status == "ready",
        "patches": int(metadata.get("patches") or 0),
        "manifest": display_path(manifest_path),
    }


def _rasterize_source_variants(variants: list[dict], src: Any) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) if src.crs else None
    geometries = []
    seen = set()
    for variant in variants:
        for feature in variant.get("snapshot", {}).get("features", []):
            geometry_payload = feature.get("geometry")
            if not geometry_payload:
                continue
            token = json.dumps(geometry_payload, sort_keys=True, separators=(",", ":"))
            if token in seen:
                continue
            seen.add(token)
            geometry = make_valid(shape(geometry_payload))
            if geometry.is_empty:
                continue
            if transformer and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                geometry = shapely_transform(transformer.transform, geometry)
            geometries.append((geometry, 1))
    return rasterize(
        geometries,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    ) if geometries else np.zeros((src.height, src.width), dtype="uint8")


def profile_logical_patch_preview(
    region: RegionConfig,
    row: dict,
    profile_store: "ProfileStore",
    profile_id: str,
) -> bytes:
    image_path = resolve_data_path(row.get("image_path", ""), region)
    if not image_path.exists():
        raise FileNotFoundError(f"logical patch source image not found: {row.get('logical_patch_id', '')}")
    variants = profile_store.selected_variants(profile_id, row.get("site_id", ""))
    with rasterio.open(image_path) as src:
        image = src.read()
        valid = _valid_mask(image, src.nodata)
        if variants:
            label = _rasterize_source_variants(variants, src)
        else:
            label_path = resolve_data_path(row.get("label_path", ""), region)
            if not label_path.exists():
                raise FileNotFoundError(f"logical patch label snapshot not found: {row.get('logical_patch_id', '')}")
            label = _rasterize_label(label_path, src)
        target = np.where(valid, label, 255).astype("uint8")
        actual = _derive_patch(image, target, valid, row, LOGICAL_PATCH_SIZE)
    preview = _preview_image(actual["image"], actual["mask"], actual["valid"])
    output = BytesIO()
    preview.save(output, format="PNG")
    return output.getvalue()


def _derive_patch(image: np.ndarray, target: np.ndarray, valid: np.ndarray, row: dict, output_size: int) -> dict:
    row_off, col_off = int(row["row_off"]), int(row["col_off"])
    image_patch = _padded_crop(image, row_off, col_off, fill=0)
    mask_patch = _padded_crop(target, row_off, col_off, fill=255)
    valid_patch = _padded_crop(valid, row_off, col_off, fill=False)
    if output_size != LOGICAL_PATCH_SIZE:
        bilinear = Image.Resampling.BILINEAR
        nearest = Image.Resampling.NEAREST
        dtype = image_patch.dtype
        image_patch = np.stack(
            [np.asarray(Image.fromarray(band.astype("float32"), mode="F").resize((output_size, output_size), bilinear)) for band in image_patch]
        ).round().astype(dtype)
        mask_patch = np.asarray(Image.fromarray(mask_patch).resize((output_size, output_size), nearest))
        valid_patch = np.asarray(Image.fromarray(valid_patch.astype("uint8")).resize((output_size, output_size), nearest)).astype(bool)
    mask_patch = np.where(valid_patch, mask_patch, 255).astype("uint8")
    valid_pixels = int(valid_patch.sum())
    water_pixels = int(np.count_nonzero(mask_patch == 1))
    return {
        "image": image_patch,
        "mask": mask_patch,
        "valid": valid_patch,
        "valid_pixels": valid_pixels,
        "valid_ratio": valid_pixels / float(output_size**2),
        "water_pixels": water_pixels,
        "water_ratio_valid": water_pixels / valid_pixels if valid_pixels else 0.0,
        "ignore_pixels": int(np.count_nonzero(mask_patch == 255)),
    }


def _eligible_actual_patch(actual: dict, patch_id: str, config: dict) -> bool:
    if actual["valid_ratio"] < float(config["min_valid_ratio"]):
        return False
    if actual["water_pixels"] >= int(config["min_water_pixels"]):
        return True
    ratio = float(config["negative_ratio"])
    token = f"{config['id']}:{int(config.get('negative_seed', 42))}:{patch_id}".encode("utf-8")
    sample = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)
    return sample < ratio


def _padded_crop(array: np.ndarray, row_off: int, col_off: int, fill: Any) -> np.ndarray:
    shape = (*array.shape[:-2], LOGICAL_PATCH_SIZE, LOGICAL_PATCH_SIZE)
    out = np.full(shape, fill, dtype=array.dtype)
    height = min(LOGICAL_PATCH_SIZE, max(0, array.shape[-2] - row_off))
    width = min(LOGICAL_PATCH_SIZE, max(0, array.shape[-1] - col_off))
    if height and width:
        out[..., :height, :width] = array[..., row_off : row_off + height, col_off : col_off + width]
    return out


def _grid_cell_bounds(transform: Any, row_off: int, col_off: int) -> tuple[float, float, float, float]:
    corners = [
        transform * (col_off, row_off),
        transform * (col_off + LOGICAL_PATCH_SIZE, row_off),
        transform * (col_off + LOGICAL_PATCH_SIZE, row_off + LOGICAL_PATCH_SIZE),
        transform * (col_off, row_off + LOGICAL_PATCH_SIZE),
    ]
    xs, ys = zip(*corners)
    return (min(xs), min(ys), max(xs), max(ys))


def _valid_mask(image: np.ndarray, nodata: Any) -> np.ndarray:
    valid = np.any(image != 0, axis=0)
    if nodata is not None:
        valid &= np.all(image != nodata, axis=0)
    return valid


def _rasterize_label(path: Path, src: Any) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features") if payload.get("type") == "FeatureCollection" else [payload]
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) if src.crs else None
    geometries = []
    for feature in features or []:
        if not feature.get("geometry"):
            continue
        geometry = make_valid(shape(feature["geometry"]))
        if geometry.is_empty:
            continue
        if transformer and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geometry = shapely_transform(transformer.transform, geometry)
        geometries.append((geometry, 1))
    return rasterize(geometries, out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8") if geometries else np.zeros((src.height, src.width), dtype="uint8")


def _write_preview(path: Path, image: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> None:
    _preview_image(image, mask, valid).save(path)


def _preview_image(image: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> Image.Image:
    indexes = [2, 1, 0] if image.shape[0] >= 3 else [0, 0, 0]
    rgb = np.stack([image[index] for index in indexes], axis=-1).astype("float32")
    output = np.zeros_like(rgb, dtype="uint8")
    for index in range(3):
        values = rgb[..., index]
        selected = values[valid & np.isfinite(values)]
        low, high = np.percentile(selected, [2, 98]) if selected.size else (0, 1)
        if high <= low:
            high = low + 1
        output[..., index] = np.clip((values - low) * 255 / (high - low), 0, 255).astype("uint8")
    output[~valid] = [34, 34, 34]
    water = mask == 1
    output[water] = (output[water].astype("uint16") * 25 // 100 + np.array([255, 32, 128], dtype="uint16") * 75 // 100).astype("uint8")
    return Image.fromarray(output)


def _split_values(value: Any) -> list[str]:
    text = str(value or "")
    separator = ";" if ";" in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]


def _value_at(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else (values[0] if values else "")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _replace_directory(source: Path, target: Path) -> None:
    backup = target.with_name(f"{target.name}.previous")
    shutil.rmtree(backup, ignore_errors=True)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(source, target)
    except Exception:
        if backup.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
