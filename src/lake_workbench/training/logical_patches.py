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
    from lake_workbench.workspaces import WorkspaceStore


LOGICAL_PATCH_SIZE = 512
DATASET_CONFIG_PATH = PROJECT_ROOT / "config" / "training_datasets.toml"


def dataset_configs(path: Path = DATASET_CONFIG_PATH) -> dict[str, dict]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return {key: {"id": key, **value} for key, value in payload.get("datasets", {}).items()}


def logical_patch_id(
    sample_id: str,
    image_index: int,
    row_off: int,
    col_off: int,
    patch_size: int = LOGICAL_PATCH_SIZE,
    image_fingerprint: str = "",
    label_fingerprint: str = "",
) -> str:
    identity = f"{sample_id}|{image_index}|{row_off}|{col_off}|{patch_size}|{image_fingerprint}|{label_fingerprint}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{sample_id}_i{image_index:02d}_r{row_off:05d}_c{col_off:05d}_s{patch_size}_{suffix}"


def grid_offsets(length: int, patch_size: int = LOGICAL_PATCH_SIZE, stride: int | None = None) -> list[int]:
    stride = stride or patch_size
    return list(range(0, max(1, length), stride))


def build_logical_patches(
    region: RegionConfig,
    output_dir: Path,
    sample_ids: set[str] | None = None,
    samples: list[dict] | None = None,
    patch_size: int = LOGICAL_PATCH_SIZE,
    stride: int | None = None,
) -> dict:
    logical_dir = output_dir
    manifest_path = logical_dir / "manifest.csv"
    previous_rows = read_csv_records(manifest_path)
    previous = {
        row.get("logical_patch_id", ""): row
        for row in previous_rows
        if row.get("logical_patch_id")
    }
    samples = list(samples) if samples is not None else read_csv_records(region.training_samples)
    if sample_ids is not None:
        samples = [row for row in samples if row.get("sample_id") in sample_ids]
    if not samples:
        return {"region": region.key, "samples": 0, "patches": 0, "manifest": ""}
    stride = stride or patch_size
    rebuilt_sample_ids = {row.get("sample_id", "") for row in samples if row.get("sample_id")}
    retained = [
        row
        for row in previous_rows
        if not (
            row.get("sample_id") in rebuilt_sample_ids
            and int(row.get("logical_size") or LOGICAL_PATCH_SIZE) == patch_size
            and int(row.get("grid_stride") or row.get("logical_size") or LOGICAL_PATCH_SIZE) == stride
        )
    ]
    parent = logical_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="logical_patches.", dir=parent))
    preview_dir = temp_dir / "preview"
    preview_dir.mkdir(parents=True)
    generated: list[dict] = []
    try:
        for sample in samples:
            generated.extend(_build_sample_logical_patches(region, sample, previous, preview_dir, logical_dir, patch_size, stride))
        retained_ids = {row.get("logical_patch_id", "") for row in retained}
        for row in retained:
            preview_path = resolve_data_path(row.get("preview_path", ""), region) if row.get("preview_path") else None
            if preview_path and preview_path.is_file():
                shutil.copy2(preview_path, preview_dir / preview_path.name)
        rows = retained + [row for row in generated if row.get("logical_patch_id", "") not in retained_ids]
        rows.sort(key=lambda row: (row["sample_id"], int(row["image_index"]), int(row["row_off"]), int(row["col_off"]), int(row["logical_size"])))
        _write_csv(temp_dir / "manifest.csv", rows)
        _replace_directory(temp_dir, logical_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        "region": region.key,
        "samples": len(samples),
        "generated_patches": len(generated),
        "patches": len(rows),
        "manifest": display_path(manifest_path),
    }


def _build_sample_logical_patches(
    region: RegionConfig,
    sample: dict,
    previous: dict[str, dict],
    preview_dir: Path,
    logical_dir: Path,
    patch_size: int,
    stride: int | None,
) -> list[dict]:
    sample_id = sample.get("sample_id", "")
    label_path = resolve_data_path(sample.get("label_path", ""), region)
    image_paths = [resolve_data_path(value, region) for value in str(sample.get("tci_path", "")).split(";") if value.strip()]
    if not sample_id or not label_path.exists():
        return []
    label_fingerprint = hashlib.sha256(label_path.read_bytes()).hexdigest()
    products = _split_values(sample.get("products") or sample.get("product_name"))
    dates = _split_values(sample.get("product_dates") or sample.get("product_date"))
    result: list[dict] = []
    for image_index, image_path in enumerate(image_paths):
        if not image_path.exists():
            continue
        with rasterio.open(image_path) as src:
            image = src.read()
            image_fingerprint = _image_fingerprint(image_path, src)
            valid = _valid_mask(image, src.nodata)
            label = _rasterize_label(label_path, src)
            target = np.where(valid, label, 255).astype("uint8")
            for row_off in grid_offsets(src.height, patch_size, stride):
                for col_off in grid_offsets(src.width, patch_size, stride):
                    image_patch = _padded_crop(image, row_off, col_off, fill=0, patch_size=patch_size)
                    valid_patch = _padded_crop(valid, row_off, col_off, fill=False, patch_size=patch_size)
                    if not np.any(valid_patch):
                        continue
                    mask_patch = _padded_crop(target, row_off, col_off, fill=255, patch_size=patch_size)
                    patch_id = logical_patch_id(sample_id, image_index, row_off, col_off, patch_size, image_fingerprint, label_fingerprint)
                    preview_path = preview_dir / f"{patch_id}.png"
                    _write_preview(preview_path, image_patch, mask_patch, valid_patch)
                    bounds = _grid_cell_bounds(src.transform, row_off, col_off, patch_size)
                    valid_pixels = int(valid_patch.sum())
                    water_pixels = int(np.count_nonzero(mask_patch == 1))
                    old = previous.get(patch_id, {})
                    old_status = str(old.get("review_status") or "")
                    old_included = truthy_flag(old.get("include"), default=True) if "include" in old else old_status == "included"
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
                            "logical_size": patch_size,
                            "grid_stride": stride,
                            "window_width": patch_size,
                            "window_height": patch_size,
                            "image_fingerprint": image_fingerprint,
                            "label_fingerprint": label_fingerprint,
                            "valid_ratio": f"{valid_pixels / float(patch_size**2):.6f}",
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
                            "include": "true" if old_included else "false",
                            "review_status": "included" if old_included else "excluded",
                            "patch_notes": old.get("patch_notes", ""),
                        }
                    )
    return result


def workspace_dataset_signature(
    region: RegionConfig,
    config: dict,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
) -> str:
    members = {patch_id for _region, patch_id in workspace_store.members(workspace_id, region.key)}
    manifest_path = workspace_store.ensure_workspace_logical_patch_manifest(workspace_id, region.key)
    logical_rows = [
        row for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id") in members
    ]
    payload = {
        "config": config,
        "logical_rows": logical_rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_workspace_training_dataset(
    region: RegionConfig,
    config_id: str,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
) -> dict:
    workspace_store.assert_trainable(workspace_id)
    configs = dataset_configs()
    if config_id not in configs:
        raise KeyError(f"unknown training dataset config: {config_id}")
    config = configs[config_id]
    member_ids = {patch_id for _region, patch_id in workspace_store.members(workspace_id, region.key)}
    manifest_path = workspace_store.ensure_workspace_logical_patch_manifest(workspace_id, region.key)
    logical_rows = [
        row for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id") in member_ids
    ]
    if not logical_rows:
        raise FileNotFoundError(f"Workspace has no selected logical patches for {region.key}: {workspace_id}")
    output_dir = workspace_store.workspace_dataset_dir(workspace_id, region.key, config_id)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{config_id}.", dir=output_dir.parent))
    cache_dir = workspace_store.workspace_training_patch_cache_dir(workspace_id, region.key, config_id)
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
                for logical in rows:
                    label_path = resolve_data_path(logical.get("label_path", ""), region)
                    if label_path.is_file():
                        label_fingerprint = logical.get("label_fingerprint") or hashlib.sha256(label_path.read_bytes()).hexdigest()
                        source_signature = str(label_fingerprint)
                        target_mask = _rasterize_label(label_path, src)
                    else:
                        variants = workspace_store.selected_variants(workspace_id, logical.get("site_id", ""))
                        if not variants:
                            continue
                        source_signature = workspace_store.source_signature(workspace_id, logical.get("site_id", ""))
                        target_mask = _rasterize_source_variants(variants, src)
                    target = np.where(valid, target_mask, 255).astype("uint8")
                    actual = _derive_patch(
                        image,
                        target,
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
                            "workspace_id": workspace_id,
                            "dataset_config_id": config_id,
                            "source_variant_ids": "",
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
            raise FileNotFoundError(f"Workspace selection produced no eligible patches for {region.key}: {workspace_id}")
        output_rows.sort(key=lambda row: row["logical_patch_id"])
        _write_csv(temp_dir / "manifest.csv", output_rows)
        signature = workspace_dataset_signature(region, config, workspace_store, workspace_id)
        (temp_dir / "build.json").write_text(
            json.dumps(
                {"workspace_id": workspace_id, "config": config, "signature": signature, "patches": len(output_rows)},
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
        "workspace_id": workspace_id,
        "region": region.key,
        "config_id": config_id,
        "patches": len(output_rows),
        "manifest": display_path(output_dir / "manifest.csv"),
    }


def build_global_training_dataset(
    scope: str,
    config_id: str,
    regions: dict[str, RegionConfig],
    workspace_store: "WorkspaceStore",
    dataset_registry: Any,
) -> dict:
    """Materialize a global Dataset from immutable contributed Patch rows."""
    configs = dataset_configs()
    if config_id not in configs:
        raise KeyError(f"unknown training dataset config: {config_id}")
    config = configs[config_id]
    rows = dataset_registry.list(scope)["items"]
    if not rows:
        raise FileNotFoundError(f"Global Dataset has no contributed patches: {scope}")
    output_dir = dataset_registry.dataset_dir(scope, config_id)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{config_id}.", dir=output_dir.parent))
    cache_dir = temp_dir / "patch_cache"
    cache_dir.mkdir(parents=True)
    output_rows: list[dict] = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        region_key = row.get("region", "")
        if region_key not in regions:
            continue
        variants = _global_row_variants(row, workspace_store)
        signature = hashlib.sha256(json.dumps(variants, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        grouped[(region_key, row.get("image_path", ""), signature)].append({**row, "_variants": variants})
    try:
        for (region_key, image_text, source_signature), group in grouped.items():
            region = regions[region_key]
            image_path = resolve_data_path(image_text, region)
            if not image_path.exists():
                continue
            with rasterio.open(image_path) as src:
                image = src.read()
                valid = _valid_mask(image, src.nodata)
                variants = group[0]["_variants"]
                label_snapshot = group[0].get("label_snapshot_json") or ""
                if label_snapshot:
                    target_mask = _rasterize_label_payload(json.loads(label_snapshot), src)
                elif variants:
                    target_mask = _rasterize_source_variants(variants, src)
                else:
                    target_mask = _rasterize_label(resolve_data_path(group[0].get("label_path", ""), region), src)
                target = np.where(valid, target_mask, 255).astype("uint8")
                for logical in group:
                    actual = _derive_patch(image, target, valid, logical, int(config["output_size"]))
                    patch_id = logical.get("source_patch_id") or logical.get("logical_patch_id", "")
                    if not _eligible_actual_patch(actual, patch_id, config):
                        continue
                    cache_path = cache_dir / f"{logical.get('source_workspace_id', 'workspace')}-{patch_id}-{source_signature[:16]}.npz"
                    with cache_path.open("wb") as handle:
                        np.savez_compressed(handle, image=actual["image"], mask=actual["mask"], valid=actual["valid"].astype("uint8"))
                    output_rows.append({
                        **{key: value for key, value in logical.items() if not key.startswith("_")},
                        "patch_id": patch_id,
                        "dataset_id": scope,
                        "dataset_config_id": config_id,
                        "source_signature": source_signature,
                        "npz_path": display_path(cache_path),
                        "patch_size": config["output_size"],
                        "valid_ratio": f"{actual['valid_ratio']:.6f}",
                        "valid_pixels": actual["valid_pixels"],
                        "water_pixels": actual["water_pixels"],
                        "water_ratio_valid": f"{actual['water_ratio_valid']:.6f}",
                        "ignore_pixels": actual["ignore_pixels"],
                        "include": "true",
                    })
        if not output_rows:
            raise FileNotFoundError(f"Global Dataset produced no eligible patches: {scope}")
        output_rows.sort(key=lambda row: (row.get("region", ""), row.get("source_patch_id", "")))
        _write_csv(temp_dir / "manifest.csv", output_rows)
        (temp_dir / "build.json").write_text(json.dumps({"dataset_id": scope, "config": config, "patches": len(output_rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_directory(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"dataset_id": scope, "config_id": config_id, "patches": len(output_rows), "manifest": display_path(output_dir / "manifest.csv")}


def _global_row_variants(row: dict, workspace_store: "WorkspaceStore") -> list[dict]:
    text = row.get("source_variants_json") or ""
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    workspace_id = row.get("source_workspace_id", "")
    return workspace_store.selected_variants(workspace_id, row.get("site_id", "")) if workspace_id else []


def workspace_training_dataset_status(
    region: RegionConfig,
    config_id: str,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
) -> dict:
    config = dataset_configs()[config_id]
    output_dir = workspace_store.workspace_dataset_dir(workspace_id, region.key, config_id)
    metadata_path = output_dir / "build.json"
    manifest_path = output_dir / "manifest.csv"
    members = workspace_store.members(workspace_id, region.key)
    workspace = workspace_store.get(workspace_id)
    if not members:
        status, metadata = "missing_selection", {}
    elif workspace["status"] == "needs_resolution":
        status, metadata = "needs_resolution", {}
    elif not metadata_path.exists() or not manifest_path.exists():
        status, metadata = "missing", {}
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = "ready" if metadata.get("signature") == workspace_dataset_signature(region, config, workspace_store, workspace_id) else "stale"
        except (OSError, json.JSONDecodeError):
            status, metadata = "stale", {}
    return {
        "workspace_id": workspace_id,
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


def workspace_logical_patch_preview(
    region: RegionConfig,
    row: dict,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
) -> bytes:
    image_path = resolve_data_path(row.get("image_path", ""), region)
    if not image_path.exists():
        raise FileNotFoundError(f"logical patch source image not found: {row.get('logical_patch_id', '')}")
    with rasterio.open(image_path) as src:
        image = src.read()
        valid = _valid_mask(image, src.nodata)
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
    logical_size = int(row.get("logical_size") or row.get("window_width") or LOGICAL_PATCH_SIZE)
    image_patch = _padded_crop(image, row_off, col_off, fill=0, patch_size=logical_size)
    mask_patch = _padded_crop(target, row_off, col_off, fill=255, patch_size=logical_size)
    valid_patch = _padded_crop(valid, row_off, col_off, fill=False, patch_size=logical_size)
    if output_size != logical_size:
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


def _padded_crop(array: np.ndarray, row_off: int, col_off: int, fill: Any, patch_size: int = LOGICAL_PATCH_SIZE) -> np.ndarray:
    shape = (*array.shape[:-2], patch_size, patch_size)
    out = np.full(shape, fill, dtype=array.dtype)
    height = min(patch_size, max(0, array.shape[-2] - row_off))
    width = min(patch_size, max(0, array.shape[-1] - col_off))
    if height and width:
        out[..., :height, :width] = array[..., row_off : row_off + height, col_off : col_off + width]
    return out


def _grid_cell_bounds(transform: Any, row_off: int, col_off: int, patch_size: int = LOGICAL_PATCH_SIZE) -> tuple[float, float, float, float]:
    corners = [
        transform * (col_off, row_off),
        transform * (col_off + patch_size, row_off),
        transform * (col_off + patch_size, row_off + patch_size),
        transform * (col_off, row_off + patch_size),
    ]
    xs, ys = zip(*corners)
    return (min(xs), min(ys), max(xs), max(ys))


def _image_fingerprint(path: Path, src: Any) -> str:
    payload = {
        "path": display_path(path),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "width": src.width,
        "height": src.height,
        "count": src.count,
        "dtype": list(src.dtypes),
        "crs": str(src.crs or ""),
        "transform": tuple(src.transform),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _valid_mask(image: np.ndarray, nodata: Any) -> np.ndarray:
    valid = np.any(image != 0, axis=0)
    if nodata is not None:
        valid &= np.all(image != nodata, axis=0)
    return valid


def _rasterize_label(path: Path, src: Any) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _rasterize_label_payload(payload, src)


def _rasterize_label_payload(payload: dict, src: Any) -> np.ndarray:
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
