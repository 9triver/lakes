"""Canonical logical patches and reproducible derived training datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from io import BytesIO
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from lake_workbench.regions.config import RegionConfig
from lake_workbench.training.patch_processing import (
    eligible_actual_patch,
    grid_cell_bounds,
    image_fingerprint,
    label_geometries,
    preview_image,
    read_patch_window,
    resize_patch,
    write_preview,
)
from lake_workbench.utils import (
    display_path,
    read_csv_records,
    resolve_data_path,
    truthy_flag,
)
from lake_workbench.training.dataset_status import (
    dataset_configs,
    workspace_dataset_signature,
    workspace_training_dataset_status as _workspace_training_dataset_status,
)

workspace_training_dataset_status = _workspace_training_dataset_status

if TYPE_CHECKING:
    from lake_workbench.workspaces import WorkspaceStore


LOGICAL_PATCH_SIZE = 512


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


def grid_offsets(
    length: int, patch_size: int = LOGICAL_PATCH_SIZE, stride: int | None = None
) -> list[int]:
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
    samples = (
        list(samples)
        if samples is not None
        else read_csv_records(region.training_samples)
    )
    if sample_ids is not None:
        samples = [row for row in samples if row.get("sample_id") in sample_ids]
    if not samples:
        return {"region": region.key, "samples": 0, "patches": 0, "manifest": ""}
    stride = stride or patch_size
    rebuilt_sample_ids = {
        row.get("sample_id", "") for row in samples if row.get("sample_id")
    }
    retained = [
        row
        for row in previous_rows
        if not (
            row.get("sample_id") in rebuilt_sample_ids
            and int(row.get("logical_size") or LOGICAL_PATCH_SIZE) == patch_size
            and int(
                row.get("grid_stride") or row.get("logical_size") or LOGICAL_PATCH_SIZE
            )
            == stride
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
            generated.extend(
                _build_sample_logical_patches(
                    region,
                    sample,
                    previous,
                    preview_dir,
                    logical_dir,
                    patch_size,
                    stride,
                )
            )
        retained_ids = {row.get("logical_patch_id", "") for row in retained}
        for row in retained:
            for field in ("preview_path", "preview_base_path"):
                preview_path = (
                    resolve_data_path(row.get(field, ""), region)
                    if row.get(field)
                    else None
                )
                if preview_path and preview_path.is_file():
                    shutil.copy2(preview_path, preview_dir / preview_path.name)
        rows = retained + [
            row
            for row in generated
            if row.get("logical_patch_id", "") not in retained_ids
        ]
        rows.sort(
            key=lambda row: (
                row["sample_id"],
                int(row["image_index"]),
                int(row["row_off"]),
                int(row["col_off"]),
                int(row["logical_size"]),
            )
        )
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
    stride = stride or patch_size
    sample_id = sample.get("sample_id", "")
    label_path = resolve_data_path(sample.get("label_path", ""), region)
    image_paths = [
        resolve_data_path(value, region)
        for value in str(sample.get("tci_path", "")).split(";")
        if value.strip()
    ]
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
            image_signature = image_fingerprint(image_path, src)
            label_payload = json.loads(label_path.read_text(encoding="utf-8"))
            geometries = label_geometries(label_payload, src.crs)
            scope_geometry = _sample_scope_geometry(sample, src.crs)
            row_offsets = _scope_grid_offsets(
                src.height, patch_size, stride, scope_geometry, src.transform, axis="row"
            )
            col_offsets = _scope_grid_offsets(
                src.width, patch_size, stride, scope_geometry, src.transform, axis="col"
            )
            for row_off in row_offsets:
                for col_off in col_offsets:
                    image_patch, mask_patch, valid_patch = read_patch_window(
                        src,
                        row_off,
                        col_off,
                        patch_size,
                        geometries,
                        scope_geometry,
                    )
                    if not np.any(valid_patch):
                        continue
                    patch_id = logical_patch_id(
                        sample_id,
                        image_index,
                        row_off,
                        col_off,
                        patch_size,
                        image_signature,
                        label_fingerprint,
                    )
                    preview_path = preview_dir / f"{patch_id}.png"
                    preview_base_path = preview_dir / f"{patch_id}.base.png"
                    write_preview(
                        preview_path, image_patch, mask_patch, valid_patch, overlay=True
                    )
                    write_preview(
                        preview_base_path,
                        image_patch,
                        mask_patch,
                        valid_patch,
                        overlay=False,
                    )
                    bounds = grid_cell_bounds(
                        src.transform, row_off, col_off, patch_size
                    )
                    valid_pixels = int(valid_patch.sum())
                    water_pixels = int(np.count_nonzero(mask_patch == 1))
                    old = previous.get(patch_id, {})
                    old_status = str(old.get("review_status") or "")
                    old_included = (
                        truthy_flag(old.get("include"), default=True)
                        if "include" in old
                        else old_status == "included"
                    )
                    # Negative patches stay available for review but do not enter the workspace by default.
                    is_new = not old
                    has_review_metadata = "exclude_reason" in old
                    auto_excluded = water_pixels == 0 and (
                        is_new or not has_review_metadata
                    )
                    included = (
                        False
                        if auto_excluded
                        else (old_included if not is_new else True)
                    )
                    exclude_reason = old.get("exclude_reason", "")
                    if auto_excluded:
                        exclude_reason = "no_water"
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
                            "preview_path": display_path(
                                logical_dir / "preview" / preview_path.name
                            ),
                            "preview_base_path": display_path(
                                logical_dir / "preview" / preview_base_path.name
                            ),
                            "row_off": row_off,
                            "col_off": col_off,
                            "logical_size": patch_size,
                            "grid_stride": stride,
                            "window_width": patch_size,
                            "window_height": patch_size,
                            "image_fingerprint": image_signature,
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
                            "label_sources": sample.get("context_sources")
                            or sample.get("label_source", ""),
                            "view_west": sample.get("view_west", ""),
                            "view_south": sample.get("view_south", ""),
                            "view_east": sample.get("view_east", ""),
                            "view_north": sample.get("view_north", ""),
                            "include": "true" if included else "false",
                            "review_status": "included" if included else "excluded",
                            "exclude_reason": exclude_reason,
                            "patch_notes": old.get("patch_notes", ""),
                        }
                    )
    return result


def _sample_scope_geometry(sample: dict, crs: Any) -> Any | None:
    try:
        values = [
            float(sample[key])
            for key in ("view_west", "view_south", "view_east", "view_north")
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    geometry = box(values[0], values[1], values[2], values[3])
    if crs and str(crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        geometry = shapely_transform(transformer.transform, geometry)
    return geometry


def _scope_grid_offsets(
    length: int,
    patch_size: int,
    stride: int,
    scope_geometry: Any | None,
    transform: Any,
    *,
    axis: str,
) -> list[int]:
    offsets = grid_offsets(length, patch_size, stride)
    if scope_geometry is None:
        return offsets
    window = from_bounds(*scope_geometry.bounds, transform=transform)
    start = float(window.row_off if axis == "row" else window.col_off)
    stop = start + float(window.height if axis == "row" else window.width)
    return [offset for offset in offsets if offset < stop and offset + patch_size > start]


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
    member_ids = {
        patch_id
        for _region, patch_id in workspace_store.members(workspace_id, region.key)
    }
    manifest_path = workspace_store.ensure_workspace_logical_patch_manifest(
        workspace_id, region.key
    )
    logical_rows = [
        row
        for row in read_csv_records(manifest_path)
        if row.get("logical_patch_id") in member_ids
    ]
    if not logical_rows:
        raise FileNotFoundError(
            f"Workspace has no selected logical patches for {region.key}: {workspace_id}"
        )
    output_dir = workspace_store.workspace_dataset_dir(
        workspace_id, region.key, config_id
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{config_id}.", dir=output_dir.parent))
    cache_dir = workspace_store.workspace_training_patch_cache_dir(
        workspace_id, region.key, config_id
    )
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
                geometry_cache: dict[Path, list[tuple[Any, int]]] = {}
                for logical in rows:
                    label_path = resolve_data_path(
                        logical.get("label_path", ""), region
                    )
                    if not label_path.is_file():
                        continue
                    label_fingerprint = (
                        logical.get("label_fingerprint")
                        or hashlib.sha256(label_path.read_bytes()).hexdigest()
                    )
                    source_signature = str(label_fingerprint)
                    if label_path not in geometry_cache:
                        payload = json.loads(label_path.read_text(encoding="utf-8"))
                        geometry_cache[label_path] = label_geometries(payload, src.crs)
                    logical_size = int(
                        logical.get("logical_size")
                        or logical.get("window_width")
                        or LOGICAL_PATCH_SIZE
                    )
                    image, target, valid = read_patch_window(
                        src,
                        int(logical["row_off"]),
                        int(logical["col_off"]),
                        logical_size,
                        geometry_cache[label_path],
                        _sample_scope_geometry(logical, src.crs),
                    )
                    actual = resize_patch(image, target, valid, int(config["output_size"]))
                    patch_id = logical["logical_patch_id"]
                    if not eligible_actual_patch(actual, patch_id, config):
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
            raise FileNotFoundError(
                f"Workspace selection produced no eligible patches for {region.key}: {workspace_id}"
            )
        output_rows.sort(key=lambda row: row["logical_patch_id"])
        _write_csv(temp_dir / "manifest.csv", output_rows)
        signature = workspace_dataset_signature(
            region, config, workspace_store, workspace_id
        )
        (temp_dir / "build.json").write_text(
            json.dumps(
                {
                    "workspace_id": workspace_id,
                    "config": config,
                    "signature": signature,
                    "patches": len(output_rows),
                },
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
    with dataset_registry.transaction():
        return _build_global_training_dataset_locked(
            scope, config_id, regions, workspace_store, dataset_registry
        )


def _build_global_training_dataset_locked(
    scope: str,
    config_id: str,
    regions: dict[str, RegionConfig],
    workspace_store: "WorkspaceStore",
    dataset_registry: Any,
) -> dict:
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
        label_snapshot = row.get("label_snapshot_json") or ""
        signature_source = label_snapshot or row.get("label_fingerprint") or row.get("label_path", "")
        signature = hashlib.sha256(str(signature_source).encode("utf-8")).hexdigest()
        grouped[(region_key, row.get("image_path", ""), signature)].append(
            row
        )
    try:
        for (region_key, image_text, source_signature), group in grouped.items():
            region = regions[region_key]
            image_path = resolve_data_path(image_text, region)
            if not image_path.exists():
                continue
            with rasterio.open(image_path) as src:
                label_snapshot = group[0].get("label_snapshot_json") or ""
                if label_snapshot:
                    payload = json.loads(label_snapshot)
                else:
                    label_path = resolve_data_path(group[0].get("label_path", ""), region)
                    if not label_path.is_file():
                        continue
                    payload = json.loads(label_path.read_text(encoding="utf-8"))
                geometries = label_geometries(payload, src.crs)
                for logical in group:
                    logical_size = int(
                        logical.get("logical_size")
                        or logical.get("window_width")
                        or LOGICAL_PATCH_SIZE
                    )
                    image, target, valid = read_patch_window(
                        src,
                        int(logical["row_off"]),
                        int(logical["col_off"]),
                        logical_size,
                        geometries,
                        _sample_scope_geometry(logical, src.crs),
                    )
                    actual = resize_patch(image, target, valid, int(config["output_size"]))
                    patch_id = logical.get("source_patch_id") or logical.get(
                        "logical_patch_id", ""
                    )
                    if not eligible_actual_patch(actual, patch_id, config):
                        continue
                    cache_name = f"{logical.get('source_workspace_id', 'workspace')}-{patch_id}-{source_signature[:16]}.npz"
                    cache_path = cache_dir / cache_name
                    final_cache_path = output_dir / "patch_cache" / cache_name
                    with cache_path.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            image=actual["image"],
                            mask=actual["mask"],
                            valid=actual["valid"].astype("uint8"),
                        )
                    output_rows.append(
                        {
                            **logical,
                            "patch_id": patch_id,
                            "dataset_id": scope,
                            "dataset_config_id": config_id,
                            "source_signature": source_signature,
                            "npz_path": display_path(final_cache_path),
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
            raise FileNotFoundError(
                f"Global Dataset produced no eligible patches: {scope}"
            )
        output_rows.sort(
            key=lambda row: (row.get("region", ""), row.get("source_patch_id", ""))
        )
        _write_csv(temp_dir / "manifest.csv", output_rows)
        (temp_dir / "build.json").write_text(
            json.dumps(
                {"dataset_id": scope, "config": config, "patches": len(output_rows)},
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
        "dataset_id": scope,
        "config_id": config_id,
        "patches": len(output_rows),
        "manifest": display_path(output_dir / "manifest.csv"),
    }
def workspace_logical_patch_preview(
    region: RegionConfig,
    row: dict,
    workspace_store: "WorkspaceStore",
    workspace_id: str,
    overlay: bool = True,
) -> bytes:
    preview_path = (
        resolve_data_path(row.get("preview_path", ""), region)
        if row.get("preview_path")
        else None
    )
    preview_base_path = (
        resolve_data_path(row.get("preview_base_path", ""), region)
        if row.get("preview_base_path")
        else None
    )
    image_path = resolve_data_path(row.get("image_path", ""), region)
    if not image_path.exists():
        fallback = preview_path if overlay else preview_base_path
        if fallback and fallback.is_file():
            return fallback.read_bytes()
        if preview_path and preview_path.is_file():
            return preview_path.read_bytes()
        raise FileNotFoundError(
            f"logical patch source image not found: {row.get('logical_patch_id', '')}"
        )
    with rasterio.open(image_path) as src:
        label_path = resolve_data_path(row.get("label_path", ""), region)
        if not label_path.exists():
            if not overlay:
                geometries = []
            elif preview_path and preview_path.is_file():
                return preview_path.read_bytes()
            else:
                raise FileNotFoundError(
                    f"logical patch label snapshot not found: {row.get('logical_patch_id', '')}"
                )
        else:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            geometries = label_geometries(payload, src.crs)
        logical_size = int(
            row.get("logical_size") or row.get("window_width") or LOGICAL_PATCH_SIZE
        )
        image, target, valid = read_patch_window(
            src,
            int(row["row_off"]),
            int(row["col_off"]),
            logical_size,
            geometries,
            _sample_scope_geometry(row, src.crs),
        )
        actual = resize_patch(image, target, valid, LOGICAL_PATCH_SIZE)
    preview = preview_image(
        actual["image"], actual["mask"], actual["valid"], overlay=overlay
    )
    output = BytesIO()
    preview.save(output, format="PNG")
    return output.getvalue()


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
