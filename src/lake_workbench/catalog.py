#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regional lake metadata, imagery inventory, and Sentinel product catalog."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from shapely.validation import make_valid

from lake_workbench.sentinel_download import (
    product_date,
    product_tile_name,
    product_type_from_name,
    upsert_csv_row,
    valid_ratio_for_tci as calculate_valid_ratio_for_tci,
)
from lake_workbench.model_validation import ModelValidationMixin
from lake_workbench.imagery import (
    blank_png,
    image_cache_key,
    mosaic_source_meta,
    render_tci_mosaic_png,
    render_tci_xyz_tile,
    rows_bounds,
    tile_cache_key,
    xyz_tile_bounds,
)
from lake_workbench.geo import (
    geometry_coverage_ratio,
    lake_aoi_geometry,
    padded_bounds,
    product_geometry,
)
from lake_workbench.region_config import RegionConfig
from lake_workbench.training_catalog import TrainingCatalogMixin
from lake_workbench.water_catalog import WaterCatalogMixin
from lake_workbench.utils import (
    area_in_bucket,
    clean_optional,
    count_values,
    display_path,
    first_present,
    jsonable as _jsonable,
    legacy_lake_keys,
    metadata_tiles,
    parse_float,
    parse_float_or_default,
    resolve_data_path,
)


@dataclass
class LakeRecord:
    lake_id: str
    object_id: str
    name: str | None
    source: str
    area_km2: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    properties: dict
    geometry: object


class LakeCatalog(ModelValidationMixin, TrainingCatalogMixin, WaterCatalogMixin):
    def __init__(self, region: RegionConfig, default_region_key: str) -> None:
        self.region = region
        self.default_region_key = default_region_key
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self.base_tci_by_tile = self._load_tci_index()
        self.tci_by_tile = dict(self.base_tci_by_tile)
        self.user_tci_rows = self._load_user_tci_rows()
        self.active_imagery = self._load_active_imagery()
        self._rebuild_effective_tci()
        self.tci_footprints = self._load_tci_footprints()
        self.sentinel_tile_index = self._load_sentinel_tile_index()
        self._valid_ratio_cache: dict[str, float] = {}
        self.lakes = self._load_lakes()
        self._lake_lookup = self._build_lake_lookup()
        self._summary_cache = {lake.object_id: self._build_summary(lake) for lake in self.lakes}
        self._detail_cache: dict[str, dict] = {}

    def _load_tci_index(self) -> dict[str, dict]:
        if not self.region.tci_index.exists():
            return {}
        tci = pd.read_csv(self.region.tci_index)
        rows = {}
        for row in tci.to_dict("records"):
            path = resolve_data_path(row["tci_path"], self.region)
            if not path.exists():
                continue
            rows[str(row["tile"]).upper()] = {
                "tile": str(row["tile"]).upper(),
                "date": str(row["date"]),
                "source": row.get("source", ""),
                "valid_ratio": float(row.get("valid_ratio", 0) or 0),
                "product": row.get("product", ""),
                "cloud_cover": row.get("cloud_cover", ""),
                "tci_path": path,
            }
        return rows

    def _load_user_tci_rows(self) -> dict[str, list[dict]]:
        rows: dict[str, list[dict]] = {}
        if not self.region.user_sentinel_index.exists():
            return rows
        table = pd.read_csv(self.region.user_sentinel_index)
        for row in table.to_dict("records"):
            path = resolve_data_path(row.get("tci_path", ""), self.region)
            if not path.exists():
                continue
            safe_path_text = clean_optional(row.get("safe_path")) or ""
            tile = str(row.get("tile", "")).upper().removeprefix("T")
            if not tile:
                continue
            item = {
                "tile": tile,
                "lake_id": clean_optional(row.get("lake_id")) or "",
                "date": clean_optional(row.get("date")) or product_date(row.get("product_name", "")),
                "source": clean_optional(row.get("source")) or "user_download",
                "valid_ratio": parse_float_or_default(row.get("valid_ratio"), 1.0),
                "product": clean_optional(row.get("product_name")) or "",
                "product_id": clean_optional(row.get("product_id")) or "",
                "cloud_cover": clean_optional(row.get("cloud_cover")) or "",
                "downloaded_at": clean_optional(row.get("downloaded_at")) or "",
                "safe_path": resolve_data_path(safe_path_text, self.region) if safe_path_text else None,
                "tci_path": path,
            }
            rows.setdefault(tile, []).append(item)
        for tile in rows:
            rows[tile].sort(key=lambda item: (str(item.get("date", "")), str(item.get("product", ""))), reverse=True)
        return rows

    def _load_active_imagery(self) -> dict[str, str]:
        if not self.region.active_imagery.exists():
            return {}
        try:
            payload = json.loads(self.region.active_imagery.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        active = {}
        for key, value in payload.items():
            text = str(key)
            if ":" in text:
                lake_id, tile = text.split(":", 1)
                active[f"{lake_id}:{tile.upper().removeprefix('T')}"] = str(value)
            else:
                active[text.upper().removeprefix("T")] = str(value)
        return active

    def _save_active_imagery(self) -> None:
        self.region.active_imagery.parent.mkdir(parents=True, exist_ok=True)
        self.region.active_imagery.write_text(json.dumps(self.active_imagery, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rebuild_effective_tci(self) -> None:
        effective = {}
        tiles = set(self.base_tci_by_tile) | set(self.user_tci_rows) | set(self.active_imagery)
        for tile in tiles:
            active_product = self.active_imagery.get(tile)
            selected = None
            base = self.base_tci_by_tile.get(tile)
            if base and base.get("product") == active_product:
                selected = base
            if selected is None:
                selected = next((row for row in self.user_tci_rows.get(tile, []) if row.get("product") == active_product), None)
            if selected is not None:
                effective[tile] = selected
        self.tci_by_tile = effective

    def _active_imagery_key(self, tile: str, lake: LakeRecord | str | None = None) -> str:
        tile = str(tile).upper().removeprefix("T")
        lake_id = lake.object_id if isinstance(lake, LakeRecord) else clean_optional(lake)
        if lake_id and any(row.get("lake_id") == lake_id for row in self.user_tci_rows.get(tile, [])):
            return f"{lake_id}:{tile}"
        return tile

    def _active_imagery_row(self, tile: str, lake: LakeRecord | str | None = None) -> dict | None:
        tile = str(tile).upper().removeprefix("T")
        active_key = self._active_imagery_key(tile, lake)
        active_product = self.active_imagery.get(active_key)
        if not active_product:
            return None
        base = self.base_tci_by_tile.get(tile)
        if active_key == tile and base and base.get("product") == active_product:
            return base
        lake_id = active_key.split(":", 1)[0] if ":" in active_key else ""
        return next(
            (
                row
                for row in self.user_tci_rows.get(tile, [])
                if row.get("product") == active_product
                and (not lake_id or not row.get("lake_id") or row.get("lake_id") == lake_id)
            ),
            None,
        )

    def _imagery_asset_meta(self, row: dict, lake_id: str = "") -> dict:
        source = clean_optional(row.get("source")) or "preloaded"
        row_lake_id = clean_optional(row.get("lake_id")) or ""
        if source == "local_img":
            asset_type = "lake_native"
            asset_scope = "lake"
            asset_label = "本体影像"
        elif source == "preloaded":
            asset_type = "preloaded_tile"
            asset_scope = "tile"
            asset_label = "预置 Sentinel tile"
        else:
            asset_type = "sentinel_tile"
            asset_scope = "tile"
            asset_label = "已下载 Sentinel tile"
        return {
            "asset_type": asset_type,
            "asset_scope": asset_scope,
            "asset_label": asset_label,
            "is_lake_native": asset_type == "lake_native",
            "is_tile_product": asset_scope == "tile",
            "applies_to_lake": not row_lake_id or not lake_id or row_lake_id == lake_id,
        }

    def _imagery_product_payload(self, row: dict, tile: str, active_key: str, lake_id: str = "", preloaded: bool = False) -> dict:
        payload = {
            "tile": tile,
            "lake_id": row.get("lake_id", ""),
            "product": row.get("product", ""),
            "product_id": row.get("product_id", ""),
            "date": row.get("date", ""),
            "source": row.get("source", "preloaded" if preloaded else "user_download"),
            "cloud_cover": row.get("cloud_cover", ""),
            "downloaded_at": row.get("downloaded_at", ""),
            "valid_ratio": row.get("valid_ratio"),
            "safe_path": display_path(row["safe_path"]) if row.get("safe_path") else "",
            "tci_path": display_path(row["tci_path"]),
            "active": self.active_imagery.get(active_key) == row.get("product"),
            "downloaded": True,
            "preloaded": preloaded,
        }
        payload.update(self._imagery_asset_meta(row, lake_id=lake_id))
        return payload

    def _load_tci_footprints(self) -> list[dict]:
        footprints = []
        for tile, row in self.tci_by_tile.items():
            try:
                with rasterio.open(row["tci_path"]) as src:
                    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    xs = [src.bounds.left, src.bounds.right, src.bounds.right, src.bounds.left]
                    ys = [src.bounds.bottom, src.bounds.bottom, src.bounds.top, src.bounds.top]
                    lons, lats = transformer.transform(xs, ys)
                    footprint = box(min(lons), min(lats), max(lons), max(lats))
            except Exception:
                continue
            footprints.append({"tile": tile, "geometry": footprint})
        return footprints

    def _load_sentinel_tile_index(self):
        index_path = next((path for path in self.region.sentinel_tile_index_paths if path.exists()), None)
        if index_path is None:
            return None
        try:
            tiles = pyogrio.read_dataframe(index_path, columns=["Name"]).to_crs("EPSG:4326")
        except Exception:
            return None
        tiles["Name"] = tiles["Name"].astype(str).str.upper().str.removeprefix("T")
        return tiles

    def _load_lakes(self) -> list[LakeRecord]:
        if self.region.lake_metadata.exists():
            return self._load_lakes_from_metadata()
        if not self.region.osm_water.exists():
            self.load_error = (
                f"Region data is not prepared: missing {display_path(self.region.lake_metadata)} "
                f"or {display_path(self.region.osm_water)}"
            )
            return []
        return self._load_lakes_from_osm()

    def _load_lakes_from_metadata(self) -> list[LakeRecord]:
        data = pyogrio.read_dataframe(self.region.lake_metadata, layer="lake_metadata").to_crs("EPSG:4326")
        records: list[LakeRecord] = []
        for _, row in data.sort_values("area_km2", ascending=False).reset_index(drop=True).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            object_id = str(row.get("lake_uid"))
            lake_id = first_present(row.get("source_feature_id"), row.get("osm_id_text"), row.get("osm_way_id_text"), object_id)
            name = clean_optional(row.get("display_name")) or clean_optional(row.get("name"))
            bbox = (
                float(row.get("bbox_west")),
                float(row.get("bbox_south")),
                float(row.get("bbox_east")),
                float(row.get("bbox_north")),
            )
            attrs = {
                key: _jsonable(row.get(key))
                for key in data.columns
                if key != "geometry"
            }
            records.append(
                LakeRecord(
                    lake_id=str(lake_id),
                    object_id=object_id,
                    name=name,
                    source=clean_optional(row.get("source_primary")) or "metadata",
                    area_km2=float(row.get("area_km2")),
                    bbox=bbox,
                    center=(float(row.get("center_lon")), float(row.get("center_lat"))),
                    properties=attrs,
                    geometry=geom,
                )
            )
        return records

    def _load_lakes_from_osm(self) -> list[LakeRecord]:
        columns = [
            "osm_id",
            "osm_way_id",
            "name",
            "type",
            "natural",
            "landuse",
            "man_made",
            "other_tags",
        ]
        data = pyogrio.read_dataframe(self.region.osm_water, layer="osm_water_polygons", columns=columns)
        data = data.to_crs("EPSG:4326")
        metric = data.to_crs("EPSG:3857")
        data["area_km2"] = metric.geometry.area / 1_000_000
        records: list[LakeRecord] = []
        for index, row in data.sort_values("area_km2", ascending=False).reset_index(drop=True).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            other_tags = str(row.get("other_tags") or "")
            # Keep lakes/reservoir-like water objects as the primary workbench list.
            if '"water"=>"river"' in other_tags or '"water"=>"canal"' in other_tags:
                continue
            if float(row["area_km2"]) < 0.01:
                continue
            osm_id = first_present(row.get("osm_id"), row.get("osm_way_id"), f"feature_{index + 1}")
            object_id = f"osm_{osm_id}"
            name = row.get("name")
            if pd.isna(name):
                name = None
            xmin, ymin, xmax, ymax = geom.bounds
            attrs = {
                key: _jsonable(row.get(key))
                for key in ["osm_id", "osm_way_id", "name", "type", "natural", "landuse", "man_made", "other_tags"]
            }
            records.append(
                LakeRecord(
                    lake_id=str(osm_id),
                    object_id=object_id,
                    name=name,
                    source="osm",
                    area_km2=float(row["area_km2"]),
                    bbox=(xmin, ymin, xmax, ymax),
                    center=((xmin + xmax) / 2, (ymin + ymax) / 2),
                    properties=attrs,
                    geometry=geom,
                )
            )
        return records

    def _build_lake_lookup(self) -> dict[str, LakeRecord]:
        lookup = {}
        for lake in self.lakes:
            keys = [
                lake.object_id,
                lake.lake_id,
                lake.properties.get("lake_uid"),
                lake.properties.get("source_feature_id"),
                lake.properties.get("osm_id_text"),
                lake.properties.get("osm_way_id_text"),
            ]
            osm_id = clean_optional(lake.properties.get("osm_id_text"))
            osm_way_id = clean_optional(lake.properties.get("osm_way_id_text"))
            if osm_id:
                keys.append(f"osm_{osm_id}")
            if osm_way_id:
                keys.append(f"osm_{osm_way_id}")
            keys.extend(legacy_lake_keys(lake.object_id))
            for key in keys:
                text = clean_optional(key)
                if text:
                    lookup[text] = lake
        return lookup

    def list_lakes(
        self,
        query: str = "",
        limit: int = 200,
        offset: int = 0,
        filters: dict | None = None,
    ) -> dict:
        query = query.strip().lower()
        filtered = self.lakes
        if query:
            filtered = [
                lake
                for lake in filtered
                if query in lake.lake_id.lower()
                or query in lake.object_id.lower()
                or query in str(lake.name or "").lower()
                or query in str(lake.properties.get("display_name", "")).lower()
                or query in str(lake.properties.get("name_zh", "")).lower()
                or query in str(lake.properties.get("name_en", "")).lower()
                or query in str(lake.properties.get("hylak_id", "")).lower()
                or query in str(lake.properties.get("osm_way_id", "")).lower()
                or query in str(lake.properties.get("osm_way_id_text", "")).lower()
            ]
        filtered = self._apply_filters(filtered, filters or {})
        page = filtered[offset : offset + limit]
        return {
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "facets": self._facets(filtered),
            "items": [self._summary(lake) for lake in page],
        }

    def _apply_filters(self, lakes: list[LakeRecord], filters: dict) -> list[LakeRecord]:
        result = lakes
        for key in ["water_type", "province", "city", "county", "polygon_quality", "metadata_quality"]:
            value = clean_optional(filters.get(key))
            if value and value != "all":
                result = [lake for lake in result if clean_optional(lake.properties.get(key)) == value]

        has_tci = clean_optional(filters.get("has_tci"))
        if has_tci in {"true", "false"}:
            expected = has_tci == "true"
            result = [lake for lake in result if bool(lake.properties.get("has_tci", self._has_tci(lake))) is expected]

        has_name = clean_optional(filters.get("has_name"))
        if has_name in {"true", "false"}:
            expected = has_name == "true"
            result = [lake for lake in result if self._has_real_name(lake) is expected]

        area_bucket = clean_optional(filters.get("area_bucket"))
        if area_bucket and area_bucket != "all":
            result = [lake for lake in result if area_in_bucket(lake.area_km2, area_bucket)]

        min_area = parse_float(filters.get("min_area"))
        max_area = parse_float(filters.get("max_area"))
        if min_area is not None:
            result = [lake for lake in result if lake.area_km2 >= min_area]
        if max_area is not None:
            result = [lake for lake in result if lake.area_km2 <= max_area]
        return result

    def _facets(self, lakes: list[LakeRecord]) -> dict:
        return {
            "water_type": count_values(lake.properties.get("water_type") for lake in lakes),
            "province": count_values(lake.properties.get("province") for lake in lakes),
            "city": count_values(lake.properties.get("city") for lake in lakes),
            "county": count_values(lake.properties.get("county") for lake in lakes),
            "polygon_quality": count_values(lake.properties.get("polygon_quality") for lake in lakes),
            "metadata_quality": count_values(lake.properties.get("metadata_quality") for lake in lakes),
        }

    def get_lake(self, lake_key: str) -> LakeRecord | None:
        return self._lake_lookup.get(lake_key)

    def get_lake_detail(self, lake: LakeRecord) -> dict:
        if lake.object_id in self._detail_cache:
            return self._detail_cache[lake.object_id]
        attrs = dict(lake.properties)
        hydrolakes = self._match_hydrolakes(lake)
        osm_layer = self._osm_layer(lake)
        detail = {
            **self._summary(lake),
            "properties": attrs,
            "layers": {
                "osm": osm_layer,
                "hydrolakes": hydrolakes,
                "esa": None,
                "jrc": None,
            },
            "geometry": mapping(lake.geometry),
        }
        self._detail_cache[lake.object_id] = detail
        return detail


    def image_for_lake(self, lake: LakeRecord, size: int = 900, padding: float = 0.3) -> tuple[bytes, dict]:
        render_bounds = padded_bounds(lake.bbox, padding)
        tci_rows = self._tci_rows_for_lake(lake)
        cache_key = image_cache_key(lake, size, padding, tci_rows)
        cache_png = self.region.image_cache_dir / f"{cache_key}.png"
        cache_meta = self.region.image_cache_dir / f"{cache_key}.json"
        if cache_png.exists() and cache_meta.exists():
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            meta["cached"] = True
            return cache_png.read_bytes(), meta

        png, meta = render_tci_mosaic_png(tci_rows, render_bounds, size=size)
        self.region.image_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_png.write_bytes(png)
        cache_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return png, meta

    def tile_meta_for_lake(self, lake: LakeRecord, padding: float = 0.8) -> dict:
        lake_bounds = padded_bounds(lake.bbox, padding)
        rows = self._tci_rows_for_lake(lake)
        tile_url_prefix = "" if self.region.key == self.default_region_key else f"/regions/{self.region.key}"
        return {
            **mosaic_source_meta(rows),
            "bounds": list(lake_bounds),
            "lake_bounds": list(lake_bounds),
            "tile_bounds": list(rows_bounds(rows)),
            "center": list(lake.center),
            "padding": padding,
            "tile_url": f"/api{tile_url_prefix}/lakes/{lake.object_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        }

    def tile_png_for_lake(
        self,
        lake: LakeRecord,
        z: int,
        x: int,
        y: int,
        padding: float = 0.8,
        tile_size: int = 256,
    ) -> tuple[bytes, dict]:
        rows = self._tci_rows_for_lake(lake)
        bounds_3857 = xyz_tile_bounds(z, x, y)
        bounds_4326 = transform_bounds("EPSG:3857", "EPSG:4326", *bounds_3857, densify_pts=21)
        tile_bounds = rows_bounds(rows)
        if not box(*bounds_4326).intersects(box(*tile_bounds)):
            return blank_png(tile_size), {"empty": True, "bounds": list(bounds_4326)}

        cache_key = tile_cache_key(lake, z, x, y, padding, rows)
        cache_png = self.region.tile_cache_dir / f"{cache_key}.png"
        if cache_png.exists():
            return cache_png.read_bytes(), {"cached": True, "bounds": list(bounds_4326)}

        payload = render_tci_xyz_tile(rows, bounds_3857, tile_size=tile_size)
        self.region.tile_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_png.write_bytes(payload)
        return payload, {"cached": False, "bounds": list(bounds_4326)}

    def _tci_rows_for_lake(self, lake: LakeRecord) -> list[dict]:
        rows = []
        missing_tiles = []
        for item in self.sentinel_tiles_for_lake(lake)["tiles"]:
            row = self._active_imagery_row(item["tile"], lake)
            if row is None:
                missing_tiles.append(item["tile"])
                continue
            rows.append(row)
        if rows:
            return rows
        candidate_tiles = [item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*lake.bbox))]
        if not candidate_tiles:
            raise FileNotFoundError(f"No downloaded TCI for lake {lake.object_id}")
        return [self.tci_by_tile[tile] for tile in candidate_tiles]

    def imagery_for_lake(self, lake: LakeRecord) -> dict:
        tiles = self.sentinel_tiles_for_lake(lake)["tiles"]
        return {
            "lake_id": lake.object_id,
            "tiles": [
                {
                    **tile,
                    "products": self.imagery_products_for_tile(tile["tile"], lake),
                }
                for tile in tiles
            ],
        }

    def local_label_items(self, lake: LakeRecord) -> dict:
        labels = []
        seen = set()
        for directory in self._local_imagery_dirs_for_lake(lake):
            for path in sorted(directory.glob("*.shp")):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                labels.append(self._local_label_item(path))
        labels.sort(key=lambda item: (item.get("date") or "", item["name"]), reverse=True)
        return {"lake_id": lake.object_id, "items": labels}

    def local_label_geojson(self, lake: LakeRecord, label_id: str) -> dict:
        labels = {item["id"]: item for item in self.local_label_items(lake)["items"]}
        item = labels.get(label_id)
        if item is None:
            raise FileNotFoundError(f"Local label not found: {label_id}")
        path = resolve_data_path(item["path"], self.region)
        if not path.exists():
            raise FileNotFoundError(f"Local label file not found: {display_path(path)}")
        data = pyogrio.read_dataframe(path)
        if data.empty:
            return {
                "lake_id": lake.object_id,
                "label": item,
                "geojson": {"type": "FeatureCollection", "features": []},
            }
        if data.crs is not None:
            data = data.to_crs("EPSG:4326")
        features = []
        for _, row in data.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            props = {
                key: _jsonable(row.get(key))
                for key in data.columns
                if key != "geometry"
            }
            props.update({"label_id": item["id"], "label_name": item["name"], "source": "local_label"})
            features.append({"type": "Feature", "geometry": mapping(make_valid(geom)), "properties": props})
        return {
            "lake_id": lake.object_id,
            "label": {**item, "feature_count": len(features)},
            "geojson": {"type": "FeatureCollection", "features": features},
        }


    def _local_imagery_dirs_for_lake(self, lake: LakeRecord) -> list[Path]:
        directories = []
        seen = set()
        for rows in self.user_tci_rows.values():
            for row in rows:
                if row.get("source") != "local_img" or row.get("lake_id") != lake.object_id:
                    continue
                directory = row.get("safe_path")
                if not directory:
                    continue
                directory = Path(directory)
                if not directory.exists() or not directory.is_dir():
                    continue
                key = str(directory.resolve())
                if key in seen:
                    continue
                seen.add(key)
                directories.append(directory)
        return directories

    def _local_label_item(self, path: Path) -> dict:
        display = display_path(path)
        digest = hashlib.sha1(display.encode("utf-8")).hexdigest()[:12]
        date_match = re.search(r"(20\d{2}|19\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", path.stem)
        label_date = "-".join(date_match.groups()) if date_match else ""
        return {
            "id": digest,
            "name": path.name,
            "stem": path.stem,
            "date": label_date,
            "path": display,
        }

    def imagery_inventory_summary(self) -> dict:
        product_tiles = set(self.base_tci_by_tile) | set(self.user_tci_rows)
        return {
            "tci_tile_count": len(product_tiles),
            "active_imagery_count": len(self.active_imagery),
            "active_tile_count": len({key.rsplit(":", 1)[-1] for key in self.active_imagery}),
            "product_count": sum(len(rows) for rows in self.user_tci_rows.values()) + len(self.base_tci_by_tile),
        }

    def imagery_products_for_tile(self, tile: str, lake: LakeRecord | str | None = None) -> list[dict]:
        tile = str(tile).upper().removeprefix("T")
        active_key = self._active_imagery_key(tile, lake)
        lake_id = lake.object_id if isinstance(lake, LakeRecord) else clean_optional(lake)
        has_lake_products = bool(lake_id and any(row.get("lake_id") == lake_id for row in self.user_tci_rows.get(tile, [])))
        products = []
        base = self.base_tci_by_tile.get(tile)
        if base and not has_lake_products:
            products.append(self._imagery_product_payload(base, tile, active_key, lake_id=lake_id, preloaded=True))
        for row in self.user_tci_rows.get(tile, []):
            if lake_id and row.get("lake_id") and row.get("lake_id") != lake_id:
                continue
            products.append(self._imagery_product_payload(row, tile, active_key, lake_id=lake_id, preloaded=False))
        return products

    def local_product_status(self, product_id: str | None, product_name: str | None) -> dict:
        product_id = clean_optional(product_id)
        product_name = clean_optional(product_name)
        for rows in self.user_tci_rows.values():
            for row in rows:
                if (product_id and clean_optional(row.get("product_id")) == product_id) or (
                    product_name and clean_optional(row.get("product")) == product_name
                ):
                    return {
                        "downloaded": True,
                        "source": "user_download",
                        "tci_path": display_path(row["tci_path"]),
                        "coverage_ratio": self.valid_ratio_for_tci(row["tci_path"]),
                        "coverage_basis": "pixels",
                    }
        for row in self.base_tci_by_tile.values():
            if product_name and clean_optional(row.get("product")) == product_name:
                return {
                    "downloaded": True,
                    "source": "preloaded",
                    "tci_path": display_path(row["tci_path"]),
                    "coverage_ratio": self.valid_ratio_for_tci(row["tci_path"]),
                    "coverage_basis": "pixels",
                }
        return {"downloaded": False}

    def valid_ratio_for_tci(self, tci_path: Path) -> float:
        key = str(tci_path)
        if key in self._valid_ratio_cache:
            return self._valid_ratio_cache[key]
        ratio = calculate_valid_ratio_for_tci(tci_path)
        self._valid_ratio_cache[key] = ratio
        return ratio

    def set_active_imagery(self, tile: str, product_name: str, lake: LakeRecord | None = None) -> dict:
        tile = str(tile).upper().removeprefix("T")
        product_name = str(product_name)
        active_key = self._active_imagery_key(tile, lake)
        lake_id = lake.object_id if lake else ""
        with self._lock:
            candidates = []
            base = self.base_tci_by_tile.get(tile)
            if base and ":" not in active_key:
                candidates.append(base)
            candidates.extend(
                row
                for row in self.user_tci_rows.get(tile, [])
                if not lake_id or not row.get("lake_id") or row.get("lake_id") == lake_id
            )
            selected = next((row for row in candidates if row.get("product") == product_name), None)
            if selected is None:
                raise KeyError(f"Product not found for tile {tile}: {product_name}")
            self.active_imagery[active_key] = product_name
            self._save_active_imagery()
            self._rebuild_effective_tci()
            self.tci_footprints = self._load_tci_footprints()
        return {
            "tile": tile,
            "lake_id": lake_id,
            "product": product_name,
            "active": True,
            "imagery": self.imagery_products_for_tile(tile, lake),
        }

    def register_downloaded_product(self, product: dict, safe_dir: Path, tci_path: Path) -> dict:
        tile = product_tile_name(product.get("name", "")) or product_tile_name(str(safe_dir.name))
        date = product_date(product.get("name", "")) or product_date(str(safe_dir.name))
        row = {
            "product_id": clean_optional(product.get("product_id")) or "",
            "product_name": clean_optional(product.get("name")) or safe_dir.name,
            "tile": tile,
            "date": date,
            "cloud_cover": clean_optional(product.get("cloud_cover")) or "",
            "product_type": product_type_from_name(product.get("name", "")),
            "source": "user_download",
            "safe_path": display_path(safe_dir),
            "tci_path": display_path(tci_path),
            "download_status": "downloaded",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "valid_ratio": self.valid_ratio_for_tci(tci_path),
        }
        with self._lock:
            upsert_csv_row(self.region.user_sentinel_index, row, key="product_name")
            self.user_tci_rows = self._load_user_tci_rows()
            self._rebuild_effective_tci()
            self.tci_footprints = self._load_tci_footprints()
        return row



    def _summary(self, lake: LakeRecord) -> dict:
        return dict(self._summary_cache.get(lake.object_id) or self._build_summary(lake))

    def _build_summary(self, lake: LakeRecord) -> dict:
        tiles = metadata_tiles(lake.properties.get("sentinel_tiles"))
        if not tiles and lake.properties.get("best_tci_tile"):
            tiles = [str(lake.properties.get("best_tci_tile")).upper().removeprefix("T")]
        return {
            "lake_id": lake.lake_id,
            "object_id": lake.object_id,
            "shape_id": lake.object_id,
            "name": lake.name,
            "display_name": lake.properties.get("display_name") or lake.name,
            "source": lake.source,
            "water_type": lake.properties.get("water_type"),
            "area_km2": lake.area_km2,
            "center": lake.center,
            "bbox": lake.bbox,
            "province": lake.properties.get("province"),
            "city": lake.properties.get("city"),
            "county": lake.properties.get("county"),
            "hylak_id": lake.properties.get("hylak_id"),
            "best_tci_tile": lake.properties.get("best_tci_tile"),
            "best_tci_date": lake.properties.get("best_tci_date"),
            "best_tci_valid_ratio": lake.properties.get("best_tci_valid_ratio"),
            "polygon_quality": lake.properties.get("polygon_quality"),
            "metadata_quality": lake.properties.get("metadata_quality"),
            "tiles": tiles,
            "has_tci": bool(lake.properties.get("has_tci") or lake.properties.get("best_tci_tile")),
        }

    def _has_tci(self, lake: LakeRecord) -> bool:
        return any(item["geometry"].intersects(box(*lake.bbox)) for item in self.tci_footprints)

    def _has_real_name(self, lake: LakeRecord) -> bool:
        return any(
            clean_optional(lake.properties.get(key))
            for key in ["name", "name_zh", "name_en"]
        )

    def sentinel_tiles_for_lake(self, lake: LakeRecord) -> dict:
        target = lake.geometry
        tiles = self._required_sentinel_tiles_for_lake(lake)
        rows = []
        for tile in tiles:
            row = self._active_imagery_row(tile, lake)
            tile_geom = self._sentinel_tile_geometry(tile)
            coverage = geometry_coverage_ratio(target, tile_geom)
            rows.append(
                {
                    "tile": tile,
                    "downloaded": row is not None,
                    "lake_coverage_ratio": coverage,
                    "aoi_coverage_ratio": coverage,
                    "geometry": mapping(tile_geom) if tile_geom is not None else None,
                    "date": row.get("date") if row else None,
                    "product": row.get("product") if row else None,
                    "valid_ratio": row.get("valid_ratio") if row else None,
                    "tci_path": display_path(row["tci_path"]) if row else None,
                }
            )
        rows.sort(key=lambda item: (item.get("aoi_coverage_ratio") or 0, item["tile"]), reverse=True)
        return {
            "lake_id": lake.object_id,
            "aoi_bounds": list(lake.bbox),
            "tiles": rows,
        }


    def _required_sentinel_tiles_for_lake(self, lake: LakeRecord) -> list[str]:
        tiles = self._sentinel_tiles_for_geometry(lake.geometry)
        if tiles:
            return tiles
        fallback_tiles = metadata_tiles(lake.properties.get("sentinel_tiles"))
        if fallback_tiles:
            return fallback_tiles
        return [item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*lake.bbox))]

    def _sentinel_tiles_for_geometry(self, geom) -> list[str]:
        if self.sentinel_tile_index is None or self.sentinel_tile_index.empty:
            return []
        candidates = self.sentinel_tile_index[self.sentinel_tile_index.geometry.intersects(geom)].copy()
        if candidates.empty:
            return []
        candidates["coverage"] = [geometry_coverage_ratio(geom, tile_geom) for tile_geom in candidates.geometry]
        candidates = candidates[candidates["coverage"] > 0].sort_values(["coverage", "Name"], ascending=[False, True])
        return candidates["Name"].tolist()

    def _sentinel_tile_geometry(self, tile: str):
        if self.sentinel_tile_index is None or self.sentinel_tile_index.empty:
            return None
        tile = str(tile).upper().removeprefix("T")
        rows = self.sentinel_tile_index[self.sentinel_tile_index["Name"] == tile]
        if rows.empty:
            return None
        return rows.geometry.iloc[0]

    def enrich_products_for_lake(self, lake: LakeRecord | None, products: list[dict]) -> list[dict]:
        if lake is None:
            return products
        lake_geom = lake.geometry
        aoi_geom = lake_aoi_geometry(lake, padding=0.8)
        enriched = []
        for product in products:
            footprint = product_geometry(product)
            enriched.append(
                {
                    **product,
                    "lake_coverage_ratio": geometry_coverage_ratio(lake_geom, footprint),
                    "aoi_coverage_ratio": geometry_coverage_ratio(aoi_geom, footprint),
                }
            )
        enriched.sort(
            key=lambda item: (
                -(item.get("lake_coverage_ratio") or 0),
                -(item.get("aoi_coverage_ratio") or 0),
                parse_float_or_default(item.get("cloud_cover"), 101.0),
                str(item.get("date") or ""),
            ),
        )
        return enriched
