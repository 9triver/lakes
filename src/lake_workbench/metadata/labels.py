"""Import local vector labels into normalized site metadata rows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyogrio

from lake_workbench.metadata.geometry import geo_frame, geometry_areas_km2, json_value, polygonal_geometry
from lake_workbench.metadata.sources import display_path
from lake_workbench.regions.config import RegionConfig
from lake_workbench.utils import resolve_data_path


DATE_RE = re.compile(r"(19\d{2}|20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")


def load_local_label_features(
    region: RegionConfig,
    sites: list[dict],
    output_path: Path | None = None,
    *,
    reuse_existing: bool = False,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    rows: list[dict] = []
    asset_counts: dict[str, int] = {}
    feature_counts: dict[str, int] = {site["site_id"]: 0 for site in sites}

    def label_directories(site: dict) -> list[Path]:
        directories = []
        seen = set()
        directory = resolve_data_path(site["source_path"], region)
        key = str(directory.resolve())
        if directory.is_dir() and key not in seen:
            seen.add(key)
            directories.append(directory)
        return directories

    def label_paths(site: dict) -> list[Path]:
        paths = []
        seen = set()
        for directory in label_directories(site):
            for path in sorted(directory.glob("*.shp")):
                key = str(path.resolve())
                if path.stat().st_size > 0 and key not in seen:
                    seen.add(key)
                    paths.append(path)
        return paths

    if output_path is not None and output_path.exists() and reuse_existing:
        try:
            info = pyogrio.read_info(output_path, layer="local_label_features")
        except Exception as exc:  # noqa: BLE001 - fall back to rebuilding the cache.
            print(f"warning: cannot reuse local-label cache {output_path}: {exc}")
        else:
            label_ids = pyogrio.read_dataframe(
                output_path,
                layer="local_label_features",
                columns=["site_id"],
                read_geometry=False,
            )["site_id"].value_counts()
            for site in sites:
                site_id = site["site_id"]
                asset_counts[site_id] = len(label_paths(site))
                feature_counts[site_id] = int(label_ids.get(site_id, 0))
            print(f"reusing local-label cache {output_path}: {info['features']} features")
            return rows, asset_counts, feature_counts
    if output_path is not None and output_path.exists():
        output_path.unlink()
    batch: list[dict] = []
    append = False

    def flush_batch() -> None:
        nonlocal append
        if not batch or output_path is None:
            return
        pyogrio.write_dataframe(
            geo_frame(batch),
            output_path,
            layer="local_label_features",
            driver="GPKG",
            promote_to_multi=True,
            append=append,
        )
        append = True
        batch.clear()

    for site in sites:
        site_id = site["site_id"]
        paths = label_paths(site)
        asset_counts[site_id] = len(paths)
        for path in paths:
            pending_rows, error = read_local_label_asset((site_id, path))
            if error is not None:
                print(f"warning: failed local label {path}: {error}")
                continue
            feature_counts[site_id] += len(pending_rows)
            if output_path is None:
                rows.extend(pending_rows)
            else:
                batch.extend(pending_rows)
                if len(batch) >= 5000:
                    flush_batch()
    flush_batch()
    return rows, asset_counts, feature_counts


def read_local_label_asset(task: tuple[str, Path]) -> tuple[list[dict], str | None]:
    site_id, path = task
    label_path = display_path(path)
    label_asset_id = hashlib.sha1(label_path.encode("utf-8")).hexdigest()[:16]
    date = date_from_text(path.stem)
    try:
        frame = pyogrio.read_dataframe(path)
    except Exception as exc:  # noqa: BLE001 - preserve the rest of a batch.
        return [], str(exc)
    if frame.crs is not None:
        frame = frame.to_crs("EPSG:4326")
    pending_rows = []
    for feature_index, row in frame.iterrows():
        geometry = polygonal_geometry(row.geometry)
        if geometry.is_empty:
            continue
        properties = {
            key: json_value(row.get(key))
            for key in frame.columns
            if key != "geometry"
        }
        pending_rows.append(
            {
                "label_feature_id": f"{label_asset_id}_{feature_index}",
                "label_asset_id": label_asset_id,
                "site_id": site_id,
                "source": "local_shapefile",
                "source_path": label_path,
                "source_filename": path.name,
                "acquisition_date": date or None,
                "source_feature_id": str(feature_index),
                "area_km2": 0.0,
                "properties_json": json.dumps(properties, ensure_ascii=False, sort_keys=True),
                "geometry": geometry,
            }
        )
    areas = geometry_areas_km2([row["geometry"] for row in pending_rows])
    for pending, area in zip(pending_rows, areas, strict=True):
        pending["area_km2"] = float(area)
    return pending_rows, None


def date_from_text(value: str) -> str:
    match = DATE_RE.search(value)
    return "-".join(match.groups()) if match else ""
