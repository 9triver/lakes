"""Persist a completed site metadata build atomically."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

from lake_workbench.regions.config import RegionConfig


@dataclass
class BuildResult:
    sites: gpd.GeoDataFrame
    cores: gpd.GeoDataFrame
    imagery: gpd.GeoDataFrame
    labels: gpd.GeoDataFrame | None
    external: gpd.GeoDataFrame
    product_rows: list[dict]
    default_imagery: dict[str, str]
    label_cache: Path | None = None


def write_result(region: RegionConfig, output_dir: Path, result: BuildResult) -> None:
    gpkg_path = output_dir / "site_metadata.gpkg"
    csv_path = output_dir / "site_metadata.csv"
    temporary_path = output_dir / ".site_metadata.building.gpkg"
    temporary_path.unlink(missing_ok=True)
    layers = [
        ("sites", result.sites),
        ("site_coverage_core", result.cores),
        ("imagery_assets", result.imagery),
        ("external_water_features", result.external),
    ]
    for layer_name, frame in layers:
        pyogrio.write_dataframe(
            frame,
            temporary_path,
            layer=layer_name,
            driver="GPKG",
            promote_to_multi=True,
        )
    expected_counts = [(layer_name, len(frame)) for layer_name, frame in layers]
    if result.label_cache is not None:
        copy_gpkg_layer(result.label_cache, temporary_path, "local_label_features")
        label_count = pyogrio.read_info(result.label_cache, layer="local_label_features")["features"]
        expected_counts.append(("local_label_features", label_count))
    else:
        if result.labels is None:
            raise RuntimeError("local label rows are unavailable")
        pyogrio.write_dataframe(
            result.labels,
            temporary_path,
            layer="local_label_features",
            driver="GPKG",
            promote_to_multi=True,
        )
        expected_counts.append(("local_label_features", len(result.labels)))
    validate_written_layers(temporary_path, expected_counts)
    temporary_path.replace(gpkg_path)
    if result.label_cache is not None and result.label_cache.exists():
        result.label_cache.unlink()
    result.sites.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    products_path = output_dir / "sentinel_products.csv"
    generated = pd.DataFrame(result.product_rows)
    if products_path.exists():
        existing = pd.read_csv(products_path)
        preserved = existing[~existing.get("source", "").fillna("").isin(["local_img", "local_imagery"])].copy()
        if not preserved.empty:
            generated = pd.concat([generated, preserved], ignore_index=True, sort=False)
    generated.to_csv(products_path, index=False)

    active_path = output_dir / "active_imagery.json"
    existing_active = {}
    if active_path.exists():
        try:
            payload = json.loads(active_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                valid_products = {
                    str(row.get("product_name") or "")
                    for row in generated.to_dict("records")
                    if row.get("product_name")
                }
                existing_active = {
                    str(key): str(value)
                    for key, value in payload.items()
                    if str(value) in valid_products
                }
        except json.JSONDecodeError:
            pass
    active_path.write_text(
        json.dumps({**result.default_imagery, **existing_active}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    normalize_site_identity_files(output_dir, result.sites)

    print(f"wrote {gpkg_path}")
    for layer_name, count in expected_counts:
        print(f"  {layer_name}: {count}")
    print(f"wrote {csv_path}")
    print(f"wrote {products_path}")
    print(f"wrote {active_path}")


def normalize_site_identity_files(output_dir: Path, sites: gpd.GeoDataFrame) -> None:
    names = dict(zip(sites["site_id"], sites["display_name"]))
    paths = [output_dir / "training_samples.csv"]
    paths.extend((output_dir / "training_patches").glob("*/manifest.csv"))
    for path in paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        if "site_id" not in frame:
            raise ValueError(f"missing site_id in {path}")
        if "site_name" not in frame:
            frame["site_name"] = frame["site_id"].map(names).fillna("")
        else:
            generated_names = frame["site_id"].map(names).fillna("")
            frame["site_name"] = frame["site_name"].where(frame["site_name"] != "", generated_names)
        frame.to_csv(path, index=False)


def copy_gpkg_layer(source: Path, destination: Path, layer_name: str, batch_size: int = 5000) -> None:
    """Copy a temporary layer without materializing all features in memory."""
    info = pyogrio.read_info(source, layer=layer_name)
    total = int(info["features"])
    for offset in range(0, total, batch_size):
        frame = pyogrio.read_dataframe(
            source,
            layer=layer_name,
            skip_features=offset,
            max_features=batch_size,
        )
        pyogrio.write_dataframe(
            frame,
            destination,
            layer=layer_name,
            driver="GPKG",
            promote_to_multi=True,
            append=offset > 0,
        )


def validate_written_layers(path: Path, layers: list[tuple[str, int]]) -> None:
    actual = {name for name, _geometry_type in pyogrio.list_layers(path)}
    expected = {name for name, _count in layers}
    if actual != expected:
        raise RuntimeError(f"site metadata layers differ: expected={expected}, actual={actual}")
    for layer_name, count in layers:
        info = pyogrio.read_info(path, layer=layer_name)
        if info["features"] != count:
            raise RuntimeError(f"wrong feature count for {layer_name}: {info['features']} != {count}")
