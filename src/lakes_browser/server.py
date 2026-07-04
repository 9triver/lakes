#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local web lake browser for regional lake sample datasets.

This is intentionally dependency-light on the web side: the HTTP server uses
Python's standard library, while GIS IO uses the project environment's
geopandas/rasterio stack.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from rasterio.warp import reproject, transform_bounds, transform_geom as rasterio_transform_geom
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from lakes_browser.sentinel_download import (
    disable_proxy_env,
    download_copernicus_product,
    product_date,
    product_tile_name,
    product_type_from_name,
    query_copernicus_tile_products,
    upsert_csv_row,
    valid_ratio_for_tci as calculate_valid_ratio_for_tci,
)
from lakes_browser.region_config import RegionConfig, load_region_configs
from lakes_browser.unet_inference import load_unet_checkpoint, predict_array


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
GLOBAL_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "all"
WEB_MERCATOR_LIMIT = 20037508.342789244


REGIONS, DEFAULT_REGION_KEY = load_region_configs()


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


class LakeCatalog:
    def __init__(self, region: RegionConfig | None = None) -> None:
        self.region = region or REGIONS[DEFAULT_REGION_KEY]
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
            # Keep lakes/reservoir-like water objects as the primary browser list.
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

    def _osm_layer(self, lake: LakeRecord) -> dict | None:
        if clean_optional(lake.properties.get("source_primary")) != "osm":
            return None
        if not truthy_flag(lake.properties.get("has_osm_polygon"), default=True):
            return None
        return {
            "source": "OSM",
            "geometry": mapping(lake.geometry),
            "properties": dict(lake.properties),
        }

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
        tile_url_prefix = "" if self.region.key == DEFAULT_REGION_KEY else f"/regions/{self.region.key}"
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

    def model_validation_models(self) -> dict:
        items = []
        default_key = ""
        default_path = self._default_model_path()
        for path, legacy in self._iter_model_paths():
            key = self._model_key(path)
            label = f"{path.parent.name}/{path.name}" if not legacy else f"旧目录 / {path.parent.name}/{path.name}"
            try:
                model = load_unet_checkpoint(path)
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "epoch": model.epoch,
                    "in_channels": model.in_channels,
                    "base_channels": model.base_channels,
                    "scope": self.region.key,
                    "legacy": legacy,
                    "default": path.resolve() == default_path.resolve(),
                }
            except Exception as exc:  # noqa: BLE001 - broken checkpoints should be visible, not fatal.
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "scope": self.region.key,
                    "legacy": legacy,
                    "error": f"{type(exc).__name__}: {exc}",
                    "default": path.resolve() == default_path.resolve(),
                }
            if item["default"]:
                default_key = item["key"]
            items.append(item)
        for path in iter_global_model_paths():
            key = global_model_key(path)
            label = f"全部区域 / {path.parent.name}/{path.name}"
            try:
                model = load_unet_checkpoint(path)
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "epoch": model.epoch,
                    "in_channels": model.in_channels,
                    "base_channels": model.base_channels,
                    "scope": "all",
                    "legacy": False,
                    "default": False,
                }
            except Exception as exc:  # noqa: BLE001 - broken checkpoints should be visible, not fatal.
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "scope": "all",
                    "legacy": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "default": False,
                }
            items.append(item)
        if not default_key and items:
            default_key = items[0]["key"]
        return {"region": self.region.key, "default": default_key, "items": items}

    def model_validation_random(self, threshold: float = 0.5, model_key: str = "") -> dict:
        model = self._load_validation_model(model_key)
        candidates = list(self.lakes)
        random.shuffle(candidates)
        skipped = []
        for lake in candidates:
            rows = self._model_validation_rows(lake, model.in_channels)
            if not rows:
                continue
            try:
                prediction = self.model_prediction_for_lake(lake, threshold=threshold, rows=rows, model=model)
            except Exception as exc:  # noqa: BLE001 - keep looking for a usable random validation target.
                skipped.append(f"{lake.object_id}: {type(exc).__name__}: {exc}")
                continue
            return {
                "region": self.region.key,
                "lake_id": lake.object_id,
                "lake": self._summary(lake),
                "model": prediction["model"],
                "prediction": prediction["prediction"],
                "stats": prediction["stats"],
                "imagery": prediction["imagery"],
                "skipped_count": len(skipped),
            }
        model_path = model.path
        raise FileNotFoundError(
            f"No lake with active imagery matching model bands ({model.in_channels}) for {self.region.key}: {display_path(model_path)}"
        )

    def model_prediction_for_lake(
        self,
        lake: LakeRecord,
        threshold: float = 0.5,
        rows: list[dict] | None = None,
        model=None,
        model_key: str = "",
    ) -> dict:
        model = model or self._load_validation_model(model_key)
        rows = rows or self._model_validation_rows(lake, model.in_channels)
        if not rows:
            raise FileNotFoundError(f"No active imagery matching model bands for lake {lake.object_id}")
        rows = sorted(rows, key=lambda row: float(row.get("valid_ratio", 0) or 0), reverse=True)
        cache_path = self._model_prediction_cache_path(lake, model.path, rows, threshold)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return payload

        row = rows[0]
        prediction, stats = predict_water_geojson(row["tci_path"], model, threshold=threshold)
        payload = {
            "region": self.region.key,
            "lake_id": lake.object_id,
            "cached": False,
            "model": {
                "key": self._model_key(model.path),
                "name": model.path.parent.name,
                "path": display_path(model.path),
                "device": str(model.device),
                "epoch": model.epoch,
                "in_channels": model.in_channels,
                "base_channels": model.base_channels,
                "threshold": threshold,
            },
            "imagery": {
                **mosaic_source_meta([row]),
                "source": row.get("source") or "",
                "asset": self._imagery_asset_meta(row, lake_id=lake.object_id),
            },
            "stats": stats,
            "prediction": prediction,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def _load_validation_model(self, model_key: str = ""):
        model_path = self._model_path_from_key(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {self.region.key}: {display_path(model_path)}")
        return load_unet_checkpoint(model_path)

    def _default_model_path(self) -> Path:
        preferred = self.region.model_dir / "unet_current_v1" / "best.pt"
        if preferred.exists():
            return preferred
        return self.region.legacy_model_dir / "unet_current_v1" / "best.pt"

    def _iter_model_paths(self) -> list[tuple[Path, bool]]:
        paths: list[tuple[Path, bool]] = []
        if self.region.model_dir.exists():
            paths.extend((path, False) for path in sorted(self.region.model_dir.glob("*/*.pt")))
        if self.region.legacy_model_dir.exists():
            paths.extend((path, True) for path in sorted(self.region.legacy_model_dir.glob("*/*.pt")))
        return paths

    def _model_path_from_key(self, model_key: str = "") -> Path:
        key = clean_optional(model_key) or ""
        if not key:
            default_path = self._default_model_path()
            if default_path.exists():
                return default_path
            candidates = [path for path, _legacy in self._iter_model_paths()]
            if candidates:
                return candidates[0]
            return default_path
        path = Path(key)
        if path.is_absolute():
            raise ValueError("absolute model paths are not allowed")
        parts = path.parts
        if len(parts) == 3 and parts[0] == "all":
            return global_model_path_from_key(key)
        if len(parts) == 3 and parts[0] == "legacy":
            if parts[1] in {"", ".", ".."} or parts[2] in {"", ".", ".."}:
                raise ValueError(f"invalid model key: {key}")
            return self.region.legacy_model_dir / parts[1] / parts[2]
        if len(parts) != 2 or parts[0] in {"", ".", ".."} or parts[1] in {"", ".", ".."}:
            raise ValueError(f"invalid model key: {key}")
        return self.region.model_dir / path

    def _model_key(self, path: Path) -> str:
        path = path.resolve()
        try:
            return f"all/{path.relative_to(GLOBAL_MODEL_DIR.resolve())}"
        except ValueError:
            pass
        try:
            return str(path.relative_to(self.region.model_dir.resolve()))
        except ValueError:
            pass
        try:
            return f"legacy/{path.relative_to(self.region.legacy_model_dir.resolve())}"
        except ValueError:
            return path.name

    def _model_validation_cache_dir(self, model_path: Path) -> Path:
        try:
            relative = model_path.resolve().relative_to((PROJECT_ROOT / "data" / "models").resolve())
            return PROJECT_ROOT / "data" / "model_predictions" / relative.parent
        except ValueError:
            return self.region.processed_dir / "model_predictions" / "legacy" / model_path.parent.name

    def _model_prediction_cache_path(self, lake: LakeRecord, model_path: Path, rows: list[dict], threshold: float) -> Path:
        payload = {
            "lake_id": lake.object_id,
            "threshold": round(float(threshold), 4),
            "model": display_path(model_path),
            "model_mtime": model_path.stat().st_mtime,
            "rows": [
                {
                    "tile": row.get("tile"),
                    "product": row.get("product"),
                    "path": display_path(row["tci_path"]),
                    "mtime": row["tci_path"].stat().st_mtime,
                }
                for row in rows
            ],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        return self._model_validation_cache_dir(model_path) / f"{safe_filename(lake.object_id)}_{digest}.geojson"

    def _model_validation_rows(self, lake: LakeRecord, in_channels: int) -> list[dict]:
        rows = []
        for tile in self._model_validation_tiles(lake):
            row = self._active_imagery_row(tile, lake)
            if row is None or row.get("tci_path") is None:
                continue
            path = row["tci_path"]
            if not path.exists():
                continue
            try:
                with rasterio.open(path) as src:
                    if src.count < in_channels:
                        continue
            except Exception:
                continue
            rows.append(row)
        return rows

    def _model_validation_tiles(self, lake: LakeRecord) -> list[str]:
        tiles = [item["tile"] for item in self.sentinel_tiles_for_lake(lake)["tiles"]]
        active_lake_tiles = [
            tile
            for tile, rows in self.user_tci_rows.items()
            if any(row.get("lake_id") == lake.object_id for row in rows)
        ]
        seen = set()
        result = []
        for tile in active_lake_tiles + tiles:
            tile = str(tile).upper().removeprefix("T")
            if tile and tile not in seen:
                seen.add(tile)
                result.append(tile)
        return result

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

    def create_training_sample(self, lake: LakeRecord, payload: dict) -> dict:
        readiness = self.training_sample_readiness(lake)
        label_source = clean_optional(payload.get("label_source")) or "osm"
        label_threshold = clean_optional(payload.get("label_threshold")) or ""
        view_state = payload.get("view_state") if isinstance(payload.get("view_state"), dict) else {}
        is_current_view = label_source == "current_view" or bool(view_state)
        label_scope = clean_optional(payload.get("label_scope")) or ("current_view" if is_current_view else "target_only")
        mask_policy = clean_optional(payload.get("mask_policy")) or ("current_view" if is_current_view else "other_water_ignore")
        context_sources = clean_optional(payload.get("context_sources")) or ("" if is_current_view else "osm,hydrolakes")
        ignore_sources = clean_optional(payload.get("ignore_sources")) or ("" if is_current_view else "osm,hydrolakes,esa,jrc")
        quality = clean_optional(payload.get("quality")) or ""
        notes = clean_optional(payload.get("notes")) or ""
        buffer_ratio = parse_float_or_default(payload.get("buffer_ratio"), 0.8)
        aoi = lake_aoi_geometry(lake, padding=buffer_ratio)
        if is_current_view:
            label_source = "current_view"
            label_layer = self.current_view_training_label_layer(lake, view_state, buffer_ratio=buffer_ratio)
            label_threshold = clean_optional(label_layer.get("properties", {}).get("jrc_threshold")) or label_threshold
            context_sources = clean_optional(label_layer.get("properties", {}).get("visible_sources")) or context_sources
        else:
            label_layer = self.training_label_layer(lake, label_source, label_threshold)
        has_label_geometry = bool(
            label_layer
            and (
                label_layer.get("geometry")
                or (label_layer.get("type") == "FeatureCollection" and label_layer.get("features"))
            )
        )
        if not has_label_geometry:
            raise ValueError(f"label source has no geometry: {label_source}")

        product_names = [item["product_name"] for item in readiness["products"]]
        tile_names = [item["tile"] for item in readiness["products"]]
        required_tile_names = readiness.get("required_tiles") or []
        missing_tile_names = readiness.get("missing_tiles") or []
        product_key = ",".join(product_names)
        asset_types = [item.get("asset_type", "") for item in readiness["products"]]
        asset_scopes = [item.get("asset_scope", "") for item in readiness["products"]]
        asset_labels = [item.get("asset_label", "") for item in readiness["products"]]
        view_state_json = json.dumps(view_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if view_state else ""
        sample_hash = hashlib.sha1(
            (product_key + label_source + label_threshold + label_scope + mask_policy + view_state_json).encode("utf-8")
        ).hexdigest()[:12]
        sample_id = f"{lake.object_id}_{sample_hash}"
        label_path = write_training_label(
            self.region,
            sample_id,
            label_layer,
            {
                "label_scope": label_scope,
                "mask_policy": mask_policy,
                "context_sources": context_sources,
                "ignore_sources": ignore_sources,
            },
        )
        row = {
            "sample_id": sample_id,
            "lake_id": lake.object_id,
            "lake_name": lake.name or "",
            "tile": ",".join(tile_names),
            "tiles": ",".join(tile_names),
            "required_tiles": ",".join(required_tile_names),
            "ready_tiles": ",".join(tile_names),
            "missing_tiles": ",".join(missing_tile_names),
            "imagery_ready": "true" if readiness.get("ready") else "false",
            "product_id": ",".join(item.get("product_id", "") for item in readiness["products"]),
            "product_name": product_key,
            "products": product_key,
            "product_source": ",".join(item.get("source", "") for item in readiness["products"]),
            "imagery_asset_type": ",".join(asset_types),
            "imagery_asset_types": ",".join(asset_types),
            "imagery_asset_scope": ",".join(asset_scopes),
            "imagery_asset_scopes": ",".join(asset_scopes),
            "imagery_asset_label": ",".join(asset_labels),
            "imagery_asset_labels": ",".join(asset_labels),
            "product_date": ",".join(item.get("date", "") for item in readiness["products"]),
            "safe_path": ";".join(item.get("safe_path", "") for item in readiness["products"]),
            "tci_path": ";".join(item.get("tci_path", "") for item in readiness["products"]),
            "label_source": label_source,
            "label_threshold": label_threshold,
            "label_scope": label_scope,
            "mask_policy": mask_policy,
            "context_sources": context_sources,
            "ignore_sources": ignore_sources,
            "label_path": display_path(label_path),
            "aoi_west": aoi.bounds[0],
            "aoi_south": aoi.bounds[1],
            "aoi_east": aoi.bounds[2],
            "aoi_north": aoi.bounds[3],
            "buffer_ratio": buffer_ratio,
            "quality": quality,
            "split": clean_optional(payload.get("split")) or "",
            "view_state_json": view_state_json,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "notes": notes,
        }
        upsert_csv_row(self.region.training_samples, row, key="sample_id")
        return row

    def current_view_training_label_layer(self, lake: LakeRecord, view_state: dict, buffer_ratio: float = 0.8) -> dict:
        visible = view_state.get("visible_layers") if isinstance(view_state.get("visible_layers"), dict) else {}
        features = []
        sources = []

        def add_layer(key: str, source_name: str, layer: dict | None) -> None:
            if not visible.get(key) or not layer or not layer.get("geometry"):
                return
            props = {**(layer.get("properties") or {})}
            props.update({"training_layer": key, "source": props.get("source") or layer.get("source") or source_name})
            features.append({"type": "Feature", "geometry": layer["geometry"], "properties": props})
            sources.append(key)

        def add_collection(key: str, collection: dict | None) -> None:
            if not visible.get(key) or not collection:
                return
            added = 0
            for feature in collection.get("features") or []:
                if not feature.get("geometry"):
                    continue
                props = {**(feature.get("properties") or {})}
                props["training_layer"] = key
                features.append({"type": "Feature", "geometry": feature["geometry"], "properties": props})
                added += 1
            if added:
                sources.append(key)

        add_layer("osm", "OSM", self._osm_layer(lake))
        add_layer("hydrolakes", "HydroLAKES", self._match_hydrolakes(lake))
        add_layer("esa", "ESA", self._esa_smoothed_layer(lake))

        if visible.get("jrc"):
            threshold = parse_int_or_default(view_state.get("jrc_threshold"), 75)
            add_layer("jrc", "JRC", self._jrc_occurrence_layer(lake, threshold=threshold))
        else:
            threshold = parse_int_or_default(view_state.get("jrc_threshold"), 75)

        if visible.get("context_osm") or visible.get("context_hydrolakes"):
            context = self.context_water_for_lake(lake, padding=buffer_ratio, min_area_km2=10, limit=500)
            add_collection("context_osm", context.get("sources", {}).get("osm"))
            add_collection("context_hydrolakes", context.get("sources", {}).get("hydrolakes"))

        local_label = view_state.get("selected_local_label") if isinstance(view_state.get("selected_local_label"), dict) else {}
        local_label_id = clean_optional(local_label.get("id"))
        if visible.get("local_label") and local_label_id:
            local_payload = self.local_label_geojson(lake, local_label_id)
            add_collection("local_label", local_payload.get("geojson"))

        if not features:
            raise ValueError("current view has no visible label geometry")

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "source": "current_view",
                "visible_sources": ",".join(sources),
                "jrc_threshold": threshold,
                "view_state": view_state,
            },
        }

    def list_training_samples(self) -> dict:
        rows = read_csv_records(self.region.training_samples)
        samples = [self._training_sample_summary(row) for row in rows]
        samples.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        quality_counts = count_values(item.get("quality") for item in samples)
        split_counts = count_values(item.get("split") for item in samples)
        return {
            "total": len(samples),
            "items": samples,
            "quality_counts": quality_counts,
            "split_counts": split_counts,
        }

    def update_training_sample(self, sample_id: str, payload: dict) -> dict:
        rows = read_csv_records(self.region.training_samples)
        sample_id = str(sample_id)
        updated = None
        allowed = {"quality", "split", "notes", "label_scope", "mask_policy", "context_sources", "ignore_sources"}
        for row in rows:
            if row.get("sample_id") != sample_id:
                continue
            for key in allowed:
                if key in payload:
                    row[key] = clean_optional(payload.get(key)) or ""
            updated = row
            break
        if updated is None:
            raise KeyError(f"training sample not found: {sample_id}")
        write_csv_records(self.region.training_samples, rows)
        return self._training_sample_summary(updated)

    def delete_training_sample(self, sample_id: str) -> dict:
        rows = read_csv_records(self.region.training_samples)
        sample_id = str(sample_id)
        kept = [row for row in rows if row.get("sample_id") != sample_id]
        if len(kept) == len(rows):
            raise KeyError(f"training sample not found: {sample_id}")
        write_csv_records(self.region.training_samples, kept)
        return {"sample_id": sample_id, "deleted": True}

    def list_training_patches(self, include: str = "") -> dict:
        rows = []
        for manifest_path in self.training_patch_manifest_paths():
            for row in read_csv_records(manifest_path):
                item = self._training_patch_summary(row, manifest_path)
                if include == "included" and not item["included"]:
                    continue
                if include == "excluded" and item["included"]:
                    continue
                rows.append(item)
        rows.sort(key=lambda item: (item.get("sample_id", ""), int(item.get("row_off") or 0), int(item.get("col_off") or 0)))
        included_count = sum(1 for item in rows if item["included"])
        excluded_count = len(rows) - included_count
        return {
            "total": len(rows),
            "included_count": included_count,
            "excluded_count": excluded_count,
            "items": rows,
        }

    def update_training_patch(self, patch_id: str, payload: dict) -> dict:
        patch_id = str(patch_id)
        for manifest_path in self.training_patch_manifest_paths():
            rows = read_csv_records(manifest_path)
            updated = None
            for row in rows:
                if row.get("patch_id") != patch_id:
                    continue
                if "include" in payload or "included" in payload:
                    value = payload.get("include") if "include" in payload else payload.get("included")
                    row["include"] = "true" if truthy_flag(value, default=False) else "false"
                if "patch_notes" in payload:
                    row["patch_notes"] = clean_optional(payload.get("patch_notes")) or ""
                updated = row
                break
            if updated is None:
                continue
            write_csv_records(manifest_path, rows)
            return self._training_patch_summary(updated, manifest_path)
        raise KeyError(f"training patch not found: {patch_id}")

    def training_patch_preview(self, patch_id: str) -> tuple[bytes, str]:
        patch = self.training_patch_by_id(patch_id)
        preview_path = resolve_data_path(patch.get("preview_path", ""), self.region)
        if not preview_path.exists():
            raise FileNotFoundError(f"patch preview not found: {patch_id}")
        return preview_path.read_bytes(), mimetypes.guess_type(preview_path.name)[0] or "image/png"

    def training_patch_by_id(self, patch_id: str) -> dict:
        patch_id = str(patch_id)
        for manifest_path in self.training_patch_manifest_paths():
            for row in read_csv_records(manifest_path):
                if row.get("patch_id") == patch_id:
                    return self._training_patch_summary(row, manifest_path)
        raise KeyError(f"training patch not found: {patch_id}")

    def training_patch_manifest_paths(self) -> list[Path]:
        root = self.region.processed_dir / "training_patches"
        if not root.exists():
            return []
        return sorted(root.glob("*/manifest.csv"), key=lambda path: path.stat().st_mtime, reverse=True)

    def _training_patch_summary(self, row: dict, manifest_path: Path) -> dict:
        patch_id = row.get("patch_id", "")
        preview_path = resolve_data_path(row.get("preview_path", ""), self.region) if row.get("preview_path") else None
        npz_path = resolve_data_path(row.get("npz_path", ""), self.region) if row.get("npz_path") else None
        include_value = clean_optional(row.get("include") or row.get("included"))
        included = True if include_value is None else truthy_flag(include_value, default=True)
        lake_id = row.get("lake_id", "")
        lake = self.get_lake(lake_id) if lake_id else None
        return {
            **row,
            "included": included,
            "include": "true" if included else "false",
            "lake_display_name": (lake.properties.get("display_name") if lake else None) or row.get("lake_name") or lake_id,
            "manifest_path": display_path(manifest_path),
            "preview_exists": bool(preview_path and preview_path.exists()),
            "npz_exists": bool(npz_path and npz_path.exists()),
            "preview_url": f"/api/regions/{self.region.key}/training-patches/{patch_id}/preview.png" if patch_id and preview_path and preview_path.exists() else "",
        }

    def _training_sample_summary(self, row: dict) -> dict:
        lake_id = row.get("lake_id", "")
        lake = self.get_lake(lake_id) if lake_id else None
        tci_paths = split_semicolon(row.get("tci_path"))
        safe_paths = split_semicolon(row.get("safe_path"))
        label_path = resolve_data_path(row.get("label_path", ""), self.region) if row.get("label_path") else None
        missing_tci = [path for path in tci_paths if not resolve_data_path(path, self.region).exists()]
        missing_safe = [path for path in safe_paths if path and not resolve_data_path(path, self.region).exists()]
        label_exists = bool(label_path and label_path.exists())
        status = "ok"
        if missing_tci or not label_exists:
            status = "missing_files"
        return {
            **row,
            "lake_display_name": (lake.properties.get("display_name") if lake else None) or row.get("lake_name") or lake_id,
            "tile_count": len(split_commas(row.get("tiles") or row.get("tile"))),
            "product_count": len(split_commas(row.get("products") or row.get("product_name"))),
            "imagery_asset_labels": split_commas(row.get("imagery_asset_labels") or row.get("imagery_asset_label")),
            "imagery_asset_types": split_commas(row.get("imagery_asset_types") or row.get("imagery_asset_type")),
            "label_exists": label_exists,
            "missing_tci": missing_tci,
            "missing_safe": missing_safe,
            "status": status,
        }

    def training_sample_readiness(self, lake: LakeRecord, buffer_ratio: float = 0.8) -> dict:
        aoi = lake_aoi_geometry(lake, padding=buffer_ratio)
        tiles = self._required_sentinel_tiles_for_lake(lake)
        products = []
        missing_tiles = []
        for tile in tiles:
            row = self._active_imagery_row(tile, lake)
            if row is None:
                missing_tiles.append(tile)
                continue
            products.append(
                {
                    "tile": tile,
                    "product_name": row.get("product", ""),
                    "product_id": row.get("product_id", ""),
                    "date": row.get("date", ""),
                    "source": row.get("source", ""),
                    **self._imagery_asset_meta(row, lake_id=lake.object_id),
                    "valid_ratio": row.get("valid_ratio"),
                    "safe_path": display_path(row["safe_path"]) if row.get("safe_path") else "",
                    "tci_path": display_path(row["tci_path"]) if row.get("tci_path") else "",
                }
            )
        return {
            "lake_id": lake.object_id,
            "ready": bool(tiles) and not missing_tiles,
            "required_tiles": tiles,
            "missing_tiles": missing_tiles,
            "ready_tiles": [item["tile"] for item in products],
            "required_count": len(tiles),
            "ready_count": len(products),
            "missing_count": len(missing_tiles),
            "products": products,
            "buffer_ratio": buffer_ratio,
            "aoi_bounds": list(aoi.bounds),
            "message": "ready" if tiles and not missing_tiles else "active imagery selection is incomplete",
        }

    def training_label_layer(self, lake: LakeRecord, label_source: str, label_threshold: str = "") -> dict | None:
        source = str(label_source).lower()
        if source == "osm":
            return self._osm_layer(lake)
        if source == "hydrolakes":
            return self._match_hydrolakes(lake)
        if source == "esa":
            return self._esa_smoothed_layer(lake)
        if source == "jrc":
            threshold = int(label_threshold or 75)
            return self._jrc_occurrence_layer(lake, threshold=threshold)
        raise ValueError(f"unknown label_source: {label_source}")

    def context_water_for_lake(self, lake: LakeRecord, padding: float = 0.8, min_area_km2: float = 1.0, limit: int = 500) -> dict:
        aoi = lake_aoi_geometry(lake, padding=padding)
        return {
            "lake_id": lake.object_id,
            "aoi_bounds": list(aoi.bounds),
            "min_area_km2": min_area_km2,
            "limit": limit,
            "sources": {
                "osm": self._context_osm_water(lake, aoi, min_area_km2=min_area_km2, limit=limit),
                "hydrolakes": self._context_hydrolakes_water(lake, aoi, min_area_km2=min_area_km2, limit=limit),
            },
        }

    def _context_osm_water(self, lake: LakeRecord, aoi, min_area_km2: float, limit: int) -> dict:
        if not self.region.osm_water.exists():
            return {"type": "FeatureCollection", "features": []}
        xmin, ymin, xmax, ymax = aoi.bounds
        items = []
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
        try:
            candidates = pyogrio.read_dataframe(
                self.region.osm_water,
                layer="osm_water_polygons",
                columns=columns,
                bbox=(xmin, ymin, xmax, ymax),
            ).to_crs("EPSG:4326")
        except Exception:
            return {"type": "FeatureCollection", "features": []}
        if candidates.empty:
            return {"type": "FeatureCollection", "features": []}
        metric = candidates.to_crs("EPSG:3857")
        candidates["area_km2"] = metric.geometry.area / 1_000_000
        target_ids = {
            clean_optional(lake.properties.get("osm_id_text")),
            clean_optional(lake.properties.get("osm_way_id_text")),
        }
        target_ids.discard(None)
        for _, row in candidates.sort_values("area_km2", ascending=False).iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.intersects(aoi):
                continue
            other_tags = str(row.get("other_tags") or "")
            if '"water"=>"river"' in other_tags or '"water"=>"canal"' in other_tags:
                continue
            area_km2 = parse_float(row.get("area_km2")) or 0.0
            if area_km2 < min_area_km2:
                continue
            osm_id = clean_optional(row.get("osm_id"))
            osm_way_id = clean_optional(row.get("osm_way_id"))
            if (osm_id and osm_id in target_ids) or (osm_way_id and osm_way_id in target_ids):
                continue
            if geometry_coverage_ratio(lake.geometry, geom) > 0.9 and geometry_coverage_ratio(geom, lake.geometry) > 0.9:
                continue
            clipped = make_valid(geom.intersection(aoi))
            if clipped.is_empty:
                continue
            items.append(
                {
                    "area_km2": area_km2,
                    "feature": {
                        "type": "Feature",
                        "geometry": mapping(clipped),
                        "properties": {
                            "source": "OSM",
                            "osm_id": osm_id,
                            "osm_way_id": osm_way_id,
                            "name": clean_optional(row.get("name")) or "",
                            "type": clean_optional(row.get("type")),
                            "natural": clean_optional(row.get("natural")),
                            "landuse": clean_optional(row.get("landuse")),
                            "man_made": clean_optional(row.get("man_made")),
                            "area_km2": area_km2,
                        },
                    },
                },
            )
        items.sort(key=lambda item: item["area_km2"], reverse=True)
        features = [item["feature"] for item in items[:limit]]
        return {"type": "FeatureCollection", "features": features}

    def _context_hydrolakes_water(self, lake: LakeRecord, aoi, min_area_km2: float, limit: int) -> dict:
        xmin, ymin, xmax, ymax = aoi.bounds
        items = []
        try:
            candidates = pyogrio.read_dataframe(
                self.region.hydrolakes,
                bbox=(xmin, ymin, xmax, ymax),
                columns=["Hylak_id", "Lake_name", "Country", "Lake_area"],
            ).to_crs("EPSG:4326")
        except Exception:
            candidates = pd.DataFrame()
        if candidates.empty:
            return {"type": "FeatureCollection", "features": []}
        target_hylak = clean_optional(lake.properties.get("hylak_id"))
        for _, row in candidates.iterrows():
            hylak_id = clean_optional(row.get("Hylak_id"))
            if target_hylak and hylak_id == target_hylak:
                continue
            area_km2 = parse_float(row.get("Lake_area")) or 0.0
            if area_km2 < min_area_km2:
                continue
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.intersects(aoi):
                continue
            clipped = make_valid(geom.intersection(aoi))
            if clipped.is_empty:
                continue
            items.append(
                {
                    "area_km2": area_km2,
                    "feature": {
                        "type": "Feature",
                        "geometry": mapping(clipped),
                        "properties": {
                            "source": "HydroLAKES",
                            "hylak_id": _jsonable(row.get("Hylak_id")),
                            "name": _jsonable(row.get("Lake_name")) or "",
                            "country": _jsonable(row.get("Country")),
                            "area_km2": _jsonable(row.get("Lake_area")),
                        },
                    },
                },
            )
        items.sort(key=lambda item: item["area_km2"], reverse=True)
        features = [item["feature"] for item in items[:limit]]
        return {"type": "FeatureCollection", "features": features}

    def local_product_for_tile(self, tile: str, product_name: str) -> dict:
        tile = str(tile).upper().removeprefix("T")
        product_name = str(product_name)
        for row in self.user_tci_rows.get(tile, []):
            if row.get("product") == product_name:
                return {
                    **row,
                    **self._imagery_asset_meta(row, lake_id=row.get("lake_id", "")),
                    "safe_path": display_path(row["safe_path"]) if row.get("safe_path") else "",
                    "tci_path": display_path(row["tci_path"]),
                }
        base = self.base_tci_by_tile.get(tile)
        if base and base.get("product") == product_name:
            return {
                **base,
                **self._imagery_asset_meta(base),
                "safe_path": "",
                "tci_path": display_path(base["tci_path"]),
            }
        return {}

    def _best_tile_for_lake(self, lake: LakeRecord, candidate_tiles: list[str], padding: float) -> str:
        xmin, ymin, xmax, ymax = lake.bbox
        width = xmax - xmin
        height = ymax - ymin
        pad_x = max(width * padding, 0.005)
        pad_y = max(height * padding, 0.005)
        bounds_wgs84 = (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)
        best_tile = candidate_tiles[0]
        best_area = -1.0
        for tile in candidate_tiles:
            tci = self.tci_by_tile[tile]
            try:
                with rasterio.open(tci["tci_path"]) as src:
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    xs, ys = transformer.transform(
                        [bounds_wgs84[0], bounds_wgs84[2]],
                        [bounds_wgs84[1], bounds_wgs84[3]],
                    )
                    request_geom = box(min(xs), min(ys), max(xs), max(ys))
                    raster_geom = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                    overlap = request_geom.intersection(raster_geom).area
            except Exception:
                overlap = 0.0
            if overlap > best_area:
                best_area = overlap
                best_tile = tile
        return best_tile

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

    def _candidate_sentinel_tiles(self, lake: LakeRecord, aoi=None) -> list[str]:
        aoi = aoi if aoi is not None else lake_aoi_geometry(lake, padding=0.8)
        tiles = self._sentinel_tiles_for_geometry(aoi)
        if tiles:
            return tiles
        fallback_tiles = metadata_tiles(lake.properties.get("sentinel_tiles"))
        if fallback_tiles:
            return fallback_tiles
        return [item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*lake.bbox))]

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

    def _match_hydrolakes(self, lake: LakeRecord) -> dict | None:
        xmin, ymin, xmax, ymax = lake.bbox
        pad_x = max((xmax - xmin) * 0.2, 0.01)
        pad_y = max((ymax - ymin) * 0.2, 0.01)
        try:
            candidates = pyogrio.read_dataframe(
                self.region.hydrolakes,
                bbox=(xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y),
                columns=["Hylak_id", "Lake_name", "Country", "Lake_area"],
            ).to_crs("EPSG:4326")
        except Exception:
            return None
        if candidates.empty:
            return None
        lake_m = transform_geom(lake.geometry, "EPSG:4326", "EPSG:3857")
        best = None
        best_area = 0.0
        for _, row in candidates.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.intersects(lake.geometry):
                continue
            inter = transform_geom(geom, "EPSG:4326", "EPSG:3857").intersection(lake_m).area
            if inter > best_area:
                best_area = inter
                best = row
        if best is None:
            return None
        return {
            "source": "HydroLAKES",
            "geometry": mapping(best.geometry),
            "properties": {
                "Hylak_id": _jsonable(best.get("Hylak_id")),
                "Lake_name": _jsonable(best.get("Lake_name")),
                "Country": _jsonable(best.get("Country")),
                "Lake_area": _jsonable(best.get("Lake_area")),
                "overlap_m2": best_area,
            },
        }

    def _esa_smoothed_layer(self, lake: LakeRecord) -> dict | None:
        cached = read_esa_polygon_cache(self.region, lake.object_id)
        if cached is not None:
            return cached
        if lake.area_km2 > 250:
            return {
                "source": "ESA WorldCover 2021 water mask, smoothed",
                "geometry": None,
                "properties": {
                    "water_id": f"ESA_{lake.object_id}",
                    "skipped": True,
                    "reason": "object too large for on-demand ESA polygonization; pre-generate this lake first",
                },
            }
        return build_esa_smoothed_layer(self.region, lake)

    def _jrc_occurrence_layer(self, lake: LakeRecord, threshold: int = 75) -> dict | None:
        threshold = max(1, min(100, int(threshold)))
        cached = read_jrc_polygon_cache(self.region, lake.object_id, threshold)
        if cached is not None:
            return cached
        if lake.area_km2 > 250:
            available = available_jrc_thresholds(self.region, lake.object_id)
            return {
                "source": "JRC GSW occurrence 2021",
                "geometry": None,
                "properties": {
                    "water_id": f"JRC_{lake.object_id}_{threshold}",
                    "threshold": threshold,
                    "skipped": True,
                    "reason": "object too large for on-demand JRC polygonization; pre-generate this threshold first",
                    "available_thresholds": available,
                },
            }
        return build_jrc_occurrence_layer(self.region, lake, threshold)


def _jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def esa_polygon_cache_path(region: RegionConfig, lake_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", lake_id)
    return region.esa_polygon_dir / safe_id / "esa_water.geojson"


def read_esa_polygon_cache(region: RegionConfig, lake_id: str) -> dict | None:
    path = esa_polygon_cache_path(region, lake_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload.setdefault("properties", {})
    payload["properties"]["cached"] = True
    payload["properties"]["cache_path"] = display_path(path)
    return payload


def write_esa_polygon_cache(region: RegionConfig, lake_id: str, layer: dict) -> Path:
    path = esa_polygon_cache_path(region, lake_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
    return path


def write_training_label(region: RegionConfig, sample_id: str, layer: dict, extra_properties: dict | None = None) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    path = region.training_label_dir / f"{safe_id}.geojson"
    properties = {**(layer.get("properties") or {}), **(extra_properties or {})}
    if layer.get("type") == "FeatureCollection":
        payload = {
            "type": "FeatureCollection",
            "properties": properties,
            "features": layer.get("features") or [],
        }
    else:
        payload = {
            "type": "Feature",
            "properties": properties,
            "geometry": layer.get("geometry"),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def build_esa_smoothed_layer(region: RegionConfig, lake: LakeRecord) -> dict | None:
    try:
        geom = lake.geometry.buffer(max(lake.bbox[2] - lake.bbox[0], lake.bbox[3] - lake.bbox[1]) * 0.15)
        paths = [region.esa_water_mask] if region.esa_water_mask.exists() else esa_worldcover_tile_paths(region, geom)
        geoms = []
        for path in paths:
            geoms.extend(water_polygons_from_raster(path, geom, lake.geometry, lambda arr: arr == 80 if path != region.esa_water_mask else arr == 1))
    except Exception:
        return None
    if not geoms:
        return {
            "source": "ESA WorldCover 2021 water mask, smoothed",
            "geometry": None,
            "properties": {
                "water_id": f"ESA_{lake.object_id}",
                "empty": True,
                "pre_generated": False,
            },
        }
    source = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
    smoothed = smooth_esa_geometry(source, lake.area_km2)
    return {
        "source": "ESA WorldCover 2021 water mask, smoothed",
        "geometry": mapping(smoothed),
        "properties": {
            "water_id": f"ESA_{lake.object_id}",
            "pre_generated": False,
        },
    }


def jrc_polygon_cache_path(region: RegionConfig, lake_id: str, threshold: int) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", lake_id)
    return region.jrc_polygon_dir / safe_id / f"jrc_occurrence_ge{threshold}.geojson"


def read_jrc_polygon_cache(region: RegionConfig, lake_id: str, threshold: int) -> dict | None:
    path = jrc_polygon_cache_path(region, lake_id, threshold)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload.setdefault("properties", {})
    payload["properties"]["cached"] = True
    payload["properties"]["cache_path"] = display_path(path)
    return payload


def write_jrc_polygon_cache(region: RegionConfig, lake_id: str, threshold: int, layer: dict) -> Path:
    path = jrc_polygon_cache_path(region, lake_id, threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
    return path


def available_jrc_thresholds(region: RegionConfig, lake_id: str) -> list[int]:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", lake_id)
    folder = region.jrc_polygon_dir / safe_id
    if not folder.exists():
        return []
    thresholds = []
    for path in folder.glob("jrc_occurrence_ge*.geojson"):
        match = re.search(r"ge(\d+)\.geojson$", path.name)
        if match:
            thresholds.append(int(match.group(1)))
    return sorted(thresholds)


def build_jrc_occurrence_layer(region: RegionConfig, lake: LakeRecord, threshold: int) -> dict | None:
    threshold = max(1, min(100, int(threshold)))
    try:
        geom = lake.geometry.buffer(max(lake.bbox[2] - lake.bbox[0], lake.bbox[3] - lake.bbox[1]) * 0.2)
        paths = [region.jrc_occurrence] if region.jrc_occurrence.exists() else jrc_occurrence_tile_paths(region, geom)
        geoms = []
        for path in paths:
            geoms.extend(water_polygons_from_raster(path, geom, lake.geometry, lambda arr: (arr >= threshold) & (arr <= 100)))
    except Exception:
        return None
    if not geoms:
        return {
            "source": "JRC GSW occurrence 2021",
            "geometry": None,
            "properties": {
                "water_id": f"JRC_{lake.object_id}_{threshold}",
                "threshold": threshold,
                "empty": True,
            },
        }
    source = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
    smoothed = smooth_jrc_geometry(source, lake.area_km2)
    return {
        "source": "JRC GSW occurrence 2021",
        "geometry": mapping(smoothed),
        "properties": {
            "water_id": f"JRC_{lake.object_id}_{threshold}",
            "threshold": threshold,
            "pre_generated": False,
        },
    }


def esa_worldcover_tile_paths(region: RegionConfig, geom) -> list[Path]:
    paths = sorted(region.esa_worldcover_dir.glob("ESA_WorldCover_10m_2021_v200_*_Map.tif"))
    return intersecting_raster_paths(paths, geom)


def jrc_occurrence_tile_paths(region: RegionConfig, geom) -> list[Path]:
    paths = sorted(region.jrc_gsw_dir.glob("occurrence_*v1_4_2021.tif"))
    return intersecting_raster_paths(paths, geom)


def intersecting_raster_paths(paths: list[Path], geom) -> list[Path]:
    selected = []
    for path in paths:
        try:
            with rasterio.open(path) as src:
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        except Exception:
            continue
        if box(*bounds).intersects(geom):
            selected.append(path)
    return selected


def water_polygons_from_raster(path: Path, clip_geom, target_geom, water_mask_fn) -> list:
    with rasterio.open(path) as src:
        clipped, transform = mask(src, [mapping(clip_geom)], crop=True, filled=True)
    arr = clipped[0]
    water = water_mask_fn(arr)
    geoms = []
    for geom_json, value in shapes(water.astype("uint8"), mask=water, transform=transform):
        if int(value) != 1:
            continue
        poly = shape(geom_json)
        if poly.is_empty:
            continue
        if not poly.intersects(target_geom):
            continue
        geoms.append(poly)
    return geoms


def predict_water_geojson(path: Path, model, threshold: float = 0.5) -> tuple[dict, dict]:
    threshold = float(threshold)
    with rasterio.open(path) as src:
        if src.count < model.in_channels:
            raise ValueError(f"Raster has {src.count} bands, model needs {model.in_channels}: {display_path(path)}")
        image = src.read(indexes=list(range(1, model.in_channels + 1))).astype(np.float32)
        finite = np.all(np.isfinite(image), axis=0)
        valid = finite & np.any(image != 0, axis=0)
        if src.nodata is not None:
            valid &= np.any(image != float(src.nodata), axis=0)
        probability = predict_array(model, image, valid=valid)
        predicted = (probability >= threshold) & valid
        transform = src.transform
        src_crs = src.crs

    valid_pixels = int(valid.sum())
    predicted_pixels = int(predicted.sum())
    raw_geoms = []
    for geom_json, value in shapes(predicted.astype("uint8"), mask=predicted, transform=transform):
        if int(value) != 1:
            continue
        geom = shape(geom_json)
        if geom.is_empty:
            continue
        if src_crs and str(src_crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
            geom = shape(rasterio_transform_geom(src_crs, "EPSG:4326", mapping(geom), precision=7))
        raw_geoms.append(make_valid(geom))

    metric_geoms = []
    for geom in raw_geoms:
        try:
            metric = make_valid(transform_geom(geom, "EPSG:4326", "EPSG:3857"))
        except Exception:
            continue
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        metric_geoms.extend(part for part in parts if not part.is_empty and part.area >= 1000)

    features = []
    area_m2 = 0.0
    if metric_geoms:
        merged = make_valid(unary_union(metric_geoms))
        parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
        parts = [make_valid(part.simplify(5, preserve_topology=True)) for part in parts if not part.is_empty]
        parts.sort(key=lambda part: part.area, reverse=True)
        for index, part in enumerate(parts[:500], start=1):
            if part.is_empty:
                continue
            area_m2 += float(part.area)
            geom_wgs84 = transform_geom(part, "EPSG:3857", "EPSG:4326")
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom_wgs84),
                    "properties": {
                        "source": "model_prediction",
                        "part": index,
                        "threshold": threshold,
                        "area_m2": round(float(part.area), 2),
                    },
                }
            )

    stats = {
        "threshold": threshold,
        "valid_pixels": valid_pixels,
        "predicted_pixels": predicted_pixels,
        "predicted_ratio": predicted_pixels / max(valid_pixels, 1),
        "polygon_count": len(features),
        "area_km2": area_m2 / 1_000_000,
        "mean_probability": float(probability[valid].mean()) if valid_pixels else 0.0,
        "max_probability": float(probability[valid].max()) if valid_pixels else 0.0,
    }
    return {"type": "FeatureCollection", "features": features}, stats


def smooth_jrc_geometry(geom, lake_area_km2: float):
    if lake_area_km2 > 250:
        metric = transform_geom(geom, "EPSG:4326", "EPSG:3857")
        metric = make_valid(metric)
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        min_area_m2 = 100_000
        kept = [part for part in parts if part.area >= min_area_m2]
        if not kept:
            kept = parts
        simplified = flatten_polygons(make_valid(part.simplify(30, preserve_topology=True)) for part in kept)
        result = simplified[0] if len(simplified) == 1 else MultiPolygon(simplified)
        return transform_geom(result, "EPSG:3857", "EPSG:4326")
    return smooth_water_geometry(geom)


def smooth_esa_geometry(geom, lake_area_km2: float):
    if lake_area_km2 > 250:
        metric = transform_geom(geom, "EPSG:4326", "EPSG:3857")
        metric = make_valid(metric)
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        min_area_m2 = 100_000
        kept = [part for part in parts if part.area >= min_area_m2]
        if not kept:
            kept = parts
        simplified = flatten_polygons(make_valid(part.simplify(30, preserve_topology=True)) for part in kept)
        result = simplified[0] if len(simplified) == 1 else MultiPolygon(simplified)
        return transform_geom(result, "EPSG:3857", "EPSG:4326")
    return smooth_water_geometry(geom)


def flatten_polygons(geoms) -> list[Polygon]:
    polygons: list[Polygon] = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(part for part in geom.geoms if not part.is_empty)
    return polygons


def clean_optional(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def first_present(*values):
    for value in values:
        text = clean_optional(value)
        if text:
            return text
    return ""


def legacy_lake_keys(value: str) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    aliases = []
    for current, legacy in [("hunan_", "hn_"), ("gansu_", "gs_")]:
        if text.startswith(current):
            aliases.append(legacy + text[len(current) :])
        elif text.startswith(legacy):
            aliases.append(current + text[len(legacy) :])
    for prefix in ("gansu", "shaanxi"):
        current = f"{prefix}_"
        legacy = f"{prefix}_mu_"
        if text.startswith(legacy):
            aliases.append(current + text[len(legacy) :])
        elif text.startswith(current) and not text.startswith(legacy):
            aliases.append(legacy + text[len(current) :])
    return aliases


def parse_float(value) -> float | None:
    text = clean_optional(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_float_or_default(value, default: float) -> float:
    parsed = parse_float(value)
    return default if parsed is None else parsed


def parse_int_or_default(value, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def truthy_flag(value, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none", "null"}:
            return default
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def split_commas(value) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def split_semicolon(value) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def run_patch_export(region_key: str, options: dict) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_training_patches import export_training_patches  # noqa: PLC0415

    patch_size = parse_int_or_default(options.get("patch_size"), 256)
    stride = parse_int_or_default(options.get("stride"), 128)
    min_valid_ratio = parse_float_or_default(options.get("min_valid_ratio"), 0.6)
    min_water_pixels = parse_int_or_default(options.get("min_water_pixels"), 1)
    negative_ratio = parse_float_or_default(options.get("negative_ratio"), 0.25)
    preview_scale = parse_int_or_default(options.get("preview_scale"), 2)
    preview_limit = parse_int_or_default(options.get("preview_limit"), 0)
    sample_ids = split_commas(options.get("sample_id") or options.get("sample_ids"))
    output_dir_text = clean_optional(options.get("output_dir")) or ""
    args = argparse.Namespace(
        region=region_key,
        sample_id=sample_ids or None,
        patch_size=patch_size,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
        min_water_pixels=min_water_pixels,
        negative_ratio=negative_ratio,
        all_touched=truthy_flag(options.get("all_touched"), default=False),
        output_dir=resolve_data_path(output_dir_text, REGIONS[region_key]) if output_dir_text else None,
        overwrite=truthy_flag(options.get("overwrite"), default=False),
        preview_limit=preview_limit,
        preview_scale=max(1, preview_scale),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = export_training_patches(args)
    return {
        **result,
        "manifest": display_path(Path(result["manifest"])),
        "npz_dir": display_path(Path(result["npz_dir"])),
        "preview_dir": display_path(Path(result["preview_dir"])),
        "options": {
            "patch_size": patch_size,
            "stride": stride,
            "min_valid_ratio": min_valid_ratio,
            "min_water_pixels": min_water_pixels,
            "negative_ratio": negative_ratio,
            "preview_scale": max(1, preview_scale),
            "preview_limit": preview_limit,
            "all_touched": truthy_flag(options.get("all_touched"), default=False),
            "overwrite": truthy_flag(options.get("overwrite"), default=False),
            "sample_ids": sample_ids,
        },
    }


def latest_patch_manifest_for_region(region: RegionConfig) -> Path:
    root = region.processed_dir / "training_patches"
    manifests = sorted(root.glob("*/manifest.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError(f"no patch manifest found under {display_path(root)}")
    return manifests[0]


def prepare_training_args(scope: str, options: dict) -> argparse.Namespace:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    scope = scope if scope == "all" else (scope if scope in REGIONS else DEFAULT_REGION_KEY)
    run_name = clean_optional(options.get("run_name")) or f"unet_{time.strftime('%Y%m%d_%H%M%S')}"
    run_name = safe_filename(run_name)
    output_dir = PROJECT_ROOT / "data" / "models" / scope / run_name
    manifest_text = clean_optional(options.get("manifest")) or ""
    patch_dir_text = clean_optional(options.get("patch_dir")) or ""
    manifest = resolve_data_path(manifest_text, REGIONS[DEFAULT_REGION_KEY]) if manifest_text else None
    patch_dir = resolve_data_path(patch_dir_text, REGIONS[DEFAULT_REGION_KEY]) if patch_dir_text else None
    if scope == "all" and manifest is None and patch_dir is None:
        manifest = build_combined_training_manifest(output_dir)
    return argparse.Namespace(
        region=scope,
        manifest=manifest,
        patch_dir=patch_dir,
        output_dir=output_dir,
        epochs=parse_int_or_default(options.get("epochs"), 30),
        batch_size=parse_int_or_default(options.get("batch_size"), 8),
        lr=parse_float_or_default(options.get("lr"), 1e-3),
        weight_decay=parse_float_or_default(options.get("weight_decay"), 1e-4),
        base_channels=parse_int_or_default(options.get("base_channels"), 32),
        val_ratio=parse_float_or_default(options.get("val_ratio"), 0.25),
        seed=parse_int_or_default(options.get("seed"), 42),
        num_workers=parse_int_or_default(options.get("num_workers"), 0),
        device=clean_optional(options.get("device")) or "auto",
        threshold=parse_float_or_default(options.get("threshold"), 0.5),
        pos_weight=clean_optional(options.get("pos_weight")) or "auto",
        max_norm_patches=parse_int_or_default(options.get("max_norm_patches"), 0),
        no_augment=truthy_flag(options.get("no_augment"), default=False),
        dry_run=truthy_flag(options.get("dry_run"), default=False),
    )


def build_combined_training_manifest(output_dir: Path) -> Path:
    rows = []
    for region in REGIONS.values():
        try:
            manifest = latest_patch_manifest_for_region(region)
        except FileNotFoundError:
            continue
        for row in read_csv_records(manifest):
            include = (row.get("include") or row.get("included") or "true").strip().lower()
            if include in {"0", "false", "no", "n"}:
                continue
            row = {**row, "source_region": region.key, "source_manifest": display_path(manifest)}
            rows.append(row)
    if not rows:
        raise FileNotFoundError("no included patch rows found for all-region training")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    write_csv_records(manifest, rows)
    return manifest


def run_training_job(scope: str, options: dict, progress_callback=None, cancel_event: threading.Event | None = None) -> dict:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_unet import train_unet  # noqa: PLC0415

    args = prepare_training_args(scope, options)
    return train_unet(args, progress_callback=progress_callback, cancel_event=cancel_event)


def iter_global_model_paths() -> list[Path]:
    if not GLOBAL_MODEL_DIR.exists():
        return []
    return sorted(GLOBAL_MODEL_DIR.glob("*/*.pt"))


def global_model_key(path: Path) -> str:
    try:
        return f"all/{path.resolve().relative_to(GLOBAL_MODEL_DIR.resolve())}"
    except ValueError:
        return f"all/{path.name}"


def global_model_path_from_key(model_key: str) -> Path:
    key = clean_optional(model_key) or ""
    path = Path(key)
    if path.is_absolute():
        raise ValueError("absolute model paths are not allowed")
    parts = path.parts
    if len(parts) == 3 and parts[0] == "all":
        run_name, weight = parts[1], parts[2]
    elif len(parts) == 2:
        run_name, weight = parts
    else:
        raise ValueError(f"invalid model key: {key}")
    if run_name in {"", ".", ".."} or weight in {"", ".", ".."}:
        raise ValueError(f"invalid model key: {key}")
    return GLOBAL_MODEL_DIR / run_name / weight


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        table = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return []
    return table.to_dict("records")


def write_csv_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def default_sentinel_date_range() -> tuple[str, str]:
    today = date.today()
    start = add_months(today, -2)
    return start.isoformat(), today.isoformat()


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def metadata_tiles(value) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    return [tile.strip() for tile in text.split(",") if tile.strip()]


def count_values(values) -> list[dict]:
    counts = {}
    for value in values:
        text = clean_optional(value)
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return [{"value": key, "count": counts[key]} for key in sorted(counts)]


def area_in_bucket(area_km2: float, bucket: str) -> bool:
    if bucket == "gte100":
        return area_km2 >= 100
    if bucket == "10_100":
        return 10 <= area_km2 < 100
    if bucket == "1_10":
        return 1 <= area_km2 < 10
    if bucket == "0_1_1":
        return 0.1 <= area_km2 < 1
    if bucket == "lt0_1":
        return area_km2 < 0.1
    return True


class DownloadManager:
    def __init__(self, catalog: LakeCatalog) -> None:
        self.catalog = catalog
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def create(self, product: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "排队中",
            "progress": 0,
            "downloaded_bytes": 0,
            "total_bytes": int(product.get("content_length") or 0),
            "product": product,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self._lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _run(self, job_id: str) -> None:
        product = self.get(job_id)["product"]
        try:
            self._update(job_id, status="authenticating", message="连接 Copernicus")

            def progress(done: int, total: int) -> None:
                pct = int(done * 100 / total) if total else 0
                self._update(
                    job_id,
                    status="downloading",
                    message=f"下载中 {pct}%",
                    progress=pct,
                    downloaded_bytes=done,
                    total_bytes=total,
                )

            safe_dir, tci_path = download_copernicus_product(
                product,
                self.catalog.region.sentinel_download_dir,
                progress=progress,
            )
            self._update(job_id, status="indexing", message="登记本地影像", progress=100)
            row = self.catalog.register_downloaded_product(product, safe_dir, tci_path)
            self._update(
                job_id,
                status="completed",
                message="下载完成",
                progress=100,
                result=row,
            )
        except Exception as exc:  # noqa: BLE001 - job error is surfaced to local UI.
            self._update(
                job_id,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )


class PatchExportManager:
    def __init__(self, catalog: LakeCatalog | None = None, catalogs: dict[str, LakeCatalog] | None = None) -> None:
        self.catalog = catalog
        self.catalogs = catalogs or {}
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def create(self, options: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "排队中",
            "progress": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "options": options,
        }
        with self._lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _run(self, job_id: str) -> None:
        options = self.get(job_id)["options"]
        try:
            self._update(job_id, status="running", message="生成 patch 中", progress=10)
            if self.catalogs:
                results = []
                for index, (region_key, _catalog) in enumerate(self.catalogs.items(), start=1):
                    self._update(job_id, message=f"生成 {region_key} patch 中", progress=max(10, int(index * 80 / max(1, len(self.catalogs)))))
                    results.append(run_patch_export(region_key, options))
                result = {
                    "region": "all",
                    "regions": results,
                    "samples": sum(item.get("samples", 0) for item in results),
                    "patches": sum(item.get("patches", 0) for item in results),
                }
            else:
                result = run_patch_export(self.catalog.region.key, options)
            self._update(
                job_id,
                status="completed",
                message=f"生成完成：{result.get('patches', 0)} 个 patch",
                progress=100,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to local UI.
            self._update(job_id, status="failed", message=f"{type(exc).__name__}: {exc}")


class TrainingManager:
    def __init__(self, scope: str) -> None:
        self.scope = scope
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.cancel_events: dict[str, threading.Event] = {}

    def create(self, options: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        cancel_event = threading.Event()
        job = {
            "job_id": job_id,
            "scope": self.scope,
            "status": "queued",
            "message": "排队中",
            "progress": 0,
            "epoch": 0,
            "epochs": parse_int_or_default(options.get("epochs"), 30),
            "history": [],
            "options": options,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self._lock:
            self.jobs[job_id] = job
            self.cancel_events[job_id] = cancel_event
        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> dict:
        with self._lock:
            jobs = [dict(job) for job in self.jobs.values()]
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return {"scope": self.scope, "items": jobs}

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            event = self.cancel_events.get(job_id)
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if event is not None:
                event.set()
            if job.get("status") in {"queued", "running", "configured"}:
                job["status"] = "cancel_requested"
                job["message"] = "正在取消"
                job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            return dict(job)

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _progress(self, job_id: str, payload: dict) -> None:
        updates = {
            "status": payload.get("status", "running"),
            "message": payload.get("message", ""),
            "progress": payload.get("progress", 0),
        }
        for key in ("epoch", "epochs", "record", "best_iou", "output_dir", "config", "result"):
            if key in payload:
                updates[key] = payload[key]
        if "record" in payload:
            current = self.get(job_id) or {}
            history = list(current.get("history") or [])
            history.append(payload["record"])
            updates["history"] = history
        self._update(job_id, **updates)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        options = job["options"]
        cancel_event = self.cancel_events[job_id]
        try:
            self._update(job_id, status="running", message="准备训练数据", progress=2)
            result = run_training_job(
                self.scope,
                options,
                progress_callback=lambda payload: self._progress(job_id, payload),
                cancel_event=cancel_event,
            )
            status = result.get("status") or "completed"
            if status == "cancelled":
                self._update(job_id, status="cancelled", message="训练已取消", progress=100, result=result)
            else:
                self._update(job_id, status="completed", message="训练完成", progress=100, result=result)
        except Exception as exc:  # noqa: BLE001 - surfaced to local UI.
            self._update(job_id, status="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self.cancel_events.pop(job_id, None)


def resolve_data_path(value, region: RegionConfig | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    region = region or REGIONS[DEFAULT_REGION_KEY]
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "data_download" and parts[1] == "downloads":
        return region.data_dir.joinpath(*parts[2:])
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def split_global_model_key(value: str) -> tuple[str, str]:
    text = clean_optional(value) or ""
    parts = text.split("/", 1)
    if len(parts) != 2:
        return "", text
    return parts[0], parts[1]


def is_frontend_route(path: str) -> bool:
    if path == "/":
        return True
    first = path.strip("/").split("/", 1)[0]
    return first in {"regions", "lakes", "training", "model"}


def transform_geom(geom, src_crs: str, dst_crs: str):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def _transform(x, y, z=None):
        return transformer.transform(x, y)

    from shapely.ops import transform as shapely_transform

    return shapely_transform(_transform, geom)


def lake_aoi_geometry(lake: LakeRecord, padding: float = 0.8):
    metric = transform_geom(lake.geometry, "EPSG:4326", "EPSG:3857")
    minx, miny, maxx, maxy = metric.bounds
    base = max(maxx - minx, maxy - miny)
    buffer_m = max(base * float(padding), 500)
    return transform_geom(metric.buffer(buffer_m), "EPSG:3857", "EPSG:4326")


def geometry_coverage_ratio(target_geom, cover_geom) -> float:
    if target_geom is None or cover_geom is None or target_geom.is_empty or cover_geom.is_empty:
        return 0.0
    try:
        target_m = transform_geom(make_valid(target_geom), "EPSG:4326", "EPSG:3857")
        cover_m = transform_geom(make_valid(cover_geom), "EPSG:4326", "EPSG:3857")
        area = target_m.area
        if area <= 0:
            return 0.0
        return float(target_m.intersection(cover_m).area / area)
    except Exception:
        return 0.0


def product_geometry(product: dict):
    footprint = product.get("footprint")
    if not footprint:
        return None
    try:
        geom = shape(footprint)
    except Exception:
        return None
    return None if geom.is_empty else geom


def smooth_water_geometry(geom):
    metric = transform_geom(geom, "EPSG:4326", "EPSG:3857")
    metric = make_valid(metric)
    opened = metric.buffer(-5, resolution=4, join_style=1).buffer(5, resolution=4, join_style=1)
    if opened.is_empty:
        opened = metric
    closed = opened.buffer(8, resolution=4, join_style=1).buffer(-8, resolution=4, join_style=1)
    if closed.is_empty:
        closed = opened
    simplified = make_valid(closed.simplify(2, preserve_topology=True))

    def _smooth_polygon(poly):
        exterior = chaikin_ring(poly.exterior.coords, iterations=2)
        holes = [chaikin_ring(ring.coords, iterations=2) for ring in poly.interiors]
        out = Polygon(exterior, holes)
        return make_valid(out)

    parts = []
    if isinstance(simplified, Polygon):
        parts = [simplified]
    elif isinstance(simplified, MultiPolygon):
        parts = list(simplified.geoms)
    else:
        return transform_geom(simplified, "EPSG:3857", "EPSG:4326")
    smoothed = []
    for part in parts:
        if part.area < 500:
            continue
        out = _smooth_polygon(part)
        if not out.is_empty:
            smoothed.append(out)
    if not smoothed:
        smoothed = parts
    result = smoothed[0] if len(smoothed) == 1 else MultiPolygon(smoothed)
    return transform_geom(result, "EPSG:3857", "EPSG:4326")


def chaikin_ring(coords, iterations=2):
    pts = list(coords)
    if len(pts) < 4:
        return pts
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    for _ in range(iterations):
        new = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            new.append((0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2))
            new.append((0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2))
        pts = new
    pts.append(pts[0])
    return pts


def padded_bounds(
    bbox_wgs84: tuple[float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox_wgs84
    width = xmax - xmin
    height = ymax - ymin
    pad_x = max(width * padding, 0.005)
    pad_y = max(height * padding, 0.005)
    return (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)


def image_cache_key(lake: LakeRecord, size: int, padding: float, tci_rows: list[dict]) -> str:
    payload = {
        "object_id": lake.object_id,
        "bbox": [round(value, 8) for value in lake.bbox],
        "size": size,
        "padding": round(padding, 4),
        "products": [
            {
                "tile": row["tile"],
                "date": row["date"],
                "product": row["product"],
                "path": display_path(row["tci_path"]),
                "mtime": row["tci_path"].stat().st_mtime,
            }
            for row in sorted(tci_rows, key=lambda item: item["tile"])
        ],
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:24]


def tile_cache_key(lake: LakeRecord, z: int, x: int, y: int, padding: float, tci_rows: list[dict]) -> str:
    payload = {
        "object_id": lake.object_id,
        "z": int(z),
        "x": int(x),
        "y": int(y),
        "padding": round(padding, 4),
        "products": [
            {
                "tile": row["tile"],
                "product": row["product"],
                "path": display_path(row["tci_path"]),
                "mtime": row["tci_path"].stat().st_mtime,
            }
            for row in sorted(tci_rows, key=lambda item: item["tile"])
        ],
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:28]


def mosaic_source_meta(tci_rows: list[dict]) -> dict:
    rows = sorted(tci_rows, key=lambda row: row["tile"])
    return {
        "tiles": [row["tile"] for row in rows],
        "dates": sorted({str(row["date"]) for row in rows}),
        "products": [str(row["product"]) for row in rows],
        "tci_path": [display_path(row["tci_path"]) for row in rows],
    }


def rows_bounds(tci_rows: list[dict]) -> tuple[float, float, float, float]:
    bounds = []
    for row in tci_rows:
        with rasterio.open(row["tci_path"]) as src:
            bounds.append(transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))
    west = min(item[0] for item in bounds)
    south = min(item[1] for item in bounds)
    east = max(item[2] for item in bounds)
    north = max(item[3] for item in bounds)
    return (west, south, east, north)


def xyz_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    tiles = 2 ** int(z)
    size = 2 * WEB_MERCATOR_LIMIT / tiles
    left = -WEB_MERCATOR_LIMIT + int(x) * size
    right = left + size
    top = WEB_MERCATOR_LIMIT - int(y) * size
    bottom = top - size
    return (left, bottom, right, top)


def render_tci_xyz_tile(
    tci_rows: list[dict],
    bounds_3857: tuple[float, float, float, float],
    tile_size: int = 256,
) -> bytes:
    output = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
    filled = np.zeros((tile_size, tile_size), dtype=bool)
    dst_transform = transform_from_bounds(*bounds_3857, tile_size, tile_size)
    ordered_rows = sorted(tci_rows, key=lambda row: float(row.get("valid_ratio", 0) or 0), reverse=True)
    for row in ordered_rows:
        with rasterio.open(row["tci_path"]) as src:
            raster_bounds_3857 = transform_bounds(src.crs, "EPSG:3857", *src.bounds, densify_pts=21)
            if not boxes_intersect(bounds_3857, raster_bounds_3857):
                continue
            raw = np.zeros((3, tile_size, tile_size), dtype=np.float32)
            reproject(
                source=rasterio.band(src, display_band_indexes(src, row)),
                destination=raw,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
            data = to_display_rgb(raw)
        valid = np.any(data != 0, axis=0) & ~filled
        if np.any(valid):
            output[:, valid] = data[:, valid]
            filled |= valid
        if np.all(filled):
            break
    rgb = np.moveaxis(output, 0, -1)
    image = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def blank_png(tile_size: int = 256) -> bytes:
    image = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def display_band_indexes(src, row: dict) -> list[int]:
    if row.get("source") == "local_img" and src.count >= 3:
        return [3, 2, 1]
    return [1, 2, 3]


def to_display_rgb(data) -> np.ndarray:
    array = np.asarray(data)
    if array.dtype == np.uint8:
        return array
    out = np.zeros(array.shape, dtype=np.uint8)
    for index in range(array.shape[0]):
        band = array[index].astype(np.float32, copy=False)
        valid = np.isfinite(band) & (band > 0)
        if not np.any(valid):
            continue
        low, high = np.percentile(band[valid], [2, 98])
        if high <= low:
            high = float(band[valid].max())
            low = float(band[valid].min())
        if high <= low:
            out[index, valid] = np.clip(band[valid], 0, 255).astype(np.uint8)
            continue
        scaled = (band - low) * 255.0 / (high - low)
        out[index] = np.clip(scaled, 0, 255).astype(np.uint8)
        out[index, ~valid] = 0
    return out


def render_tci_mosaic_png(
    tci_rows: list[dict],
    bounds_wgs84: tuple[float, float, float, float],
    size: int,
) -> tuple[bytes, dict]:
    ordered_rows = sorted(tci_rows, key=lambda row: float(row.get("valid_ratio", 0) or 0), reverse=True)
    srcs = []
    try:
        crs_values = set()
        for row in ordered_rows:
            src = rasterio.open(row["tci_path"])
            srcs.append(src)
            crs_values.add(str(src.crs))
        if len(crs_values) != 1:
            png, meta = render_tci_png(ordered_rows[0]["tci_path"], bounds_wgs84, size=size, padding=0)
            meta.update(mosaic_fallback_meta(ordered_rows))
            return png, meta

        crs = srcs[0].crs
        west, south, east, north = bounds_wgs84
        dst_bounds = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
        left, bottom, right, top = dst_bounds
        aspect = (right - left) / max(top - bottom, 1)
        out_width = size
        out_height = max(240, min(1400, round(size / max(aspect, 0.1))))
        if out_height > size:
            out_height = size
            out_width = max(240, min(1400, round(size * aspect)))
        xres = (right - left) / out_width
        yres = (top - bottom) / out_height

        mosaic, out_transform = merge(
            srcs,
            bounds=(left, bottom, right, top),
            res=(xres, yres),
            indexes=display_band_indexes(srcs[0], ordered_rows[0]),
            nodata=0,
            method="first",
            resampling=Resampling.bilinear,
        )
    finally:
        for src in srcs:
            src.close()

    display = to_display_rgb(mosaic)
    rgb = np.moveaxis(display, 0, -1)
    image = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    tiles = [row["tile"] for row in ordered_rows]
    dates = sorted({str(row["date"]) for row in ordered_rows})
    products = [str(row["product"]) for row in ordered_rows]
    height, width = mosaic.shape[1], mosaic.shape[2]
    left = out_transform.c
    top = out_transform.f
    right = left + out_transform.a * width
    bottom = top + out_transform.e * height
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = inv.transform(left, bottom)
    east, north = inv.transform(right, top)
    filled = np.any(display != 0, axis=0)
    return buf.getvalue(), {
        "bounds": [west, south, east, north],
        "width": width,
        "height": height,
        "crs": str(crs),
        "tile": ",".join(tiles),
        "tiles": tiles,
        "date": ",".join(dates),
        "dates": dates,
        "product": ",".join(products),
        "products": products,
        "valid_ratio": float(np.count_nonzero(filled) / filled.size) if filled.size else 0.0,
        "tci_path": [display_path(row["tci_path"]) for row in ordered_rows],
        "mosaic": True,
        "cached": False,
    }


def mosaic_fallback_meta(tci_rows: list[dict]) -> dict:
    tiles = [row["tile"] for row in tci_rows]
    dates = sorted({str(row["date"]) for row in tci_rows})
    return {
        "tile": tiles[0] if tiles else "",
        "tiles": tiles,
        "date": ",".join(dates),
        "dates": dates,
        "product": ",".join(str(row["product"]) for row in tci_rows),
        "products": [str(row["product"]) for row in tci_rows],
        "valid_ratio": float(tci_rows[0].get("valid_ratio", 0) or 0) if tci_rows else 0.0,
        "tci_path": [display_path(row["tci_path"]) for row in tci_rows],
        "mosaic": False,
        "mosaic_fallback": "mixed CRS",
    }


def render_tci_png(
    tci_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
    size: int,
    padding: float,
) -> tuple[bytes, dict]:
    with rasterio.open(tci_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xmin, ymin, xmax, ymax = bbox_wgs84
        width = xmax - xmin
        height = ymax - ymin
        pad_x = max(width * padding, 0.005)
        pad_y = max(height * padding, 0.005)
        bounds_wgs84 = (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)
        xs, ys = transformer.transform(
            [bounds_wgs84[0], bounds_wgs84[2]],
            [bounds_wgs84[1], bounds_wgs84[3]],
        )
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)
        src_bounds = src.bounds
        left = max(left, src_bounds.left)
        right = min(right, src_bounds.right)
        bottom = max(bottom, src_bounds.bottom)
        top = min(top, src_bounds.top)
        if right <= left or top <= bottom:
            raise ValueError(f"Lake bbox does not overlap raster {tci_path}")
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        aspect = (right - left) / max(top - bottom, 1)
        out_width = size
        out_height = max(240, min(1200, round(size / max(aspect, 0.1))))
        if out_height > size:
            out_height = size
            out_width = max(240, min(1200, round(size * aspect)))
        data = src.read(
            display_band_indexes(src, {"source": ""}),
            window=window,
            out_shape=(3, out_height, out_width),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        rgb = np.moveaxis(to_display_rgb(data), 0, -1)
        image = Image.fromarray(rgb, "RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        inv = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        west, south = inv.transform(left, bottom)
        east, north = inv.transform(right, top)
        return buf.getvalue(), {
            "bounds": [west, south, east, north],
            "width": out_width,
            "height": out_height,
            "crs": str(src.crs),
        }


class LakeHandler(BaseHTTPRequestHandler):
    catalogs: dict[str, LakeCatalog]
    downloads_by_region: dict[str, DownloadManager]
    patch_exports_by_region: dict[str, PatchExportManager]
    training_runs_by_scope: dict[str, TrainingManager]
    all_patch_exports: PatchExportManager
    catalog: LakeCatalog
    downloads: DownloadManager
    patch_exports: PatchExportManager
    training_runs: TrainingManager

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if is_frontend_route(path):
                self._serve_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                self._serve_file(STATIC_DIR / path.removeprefix("/static/"))
            elif path == "/api/regions":
                self._json(self._regions_payload())
            elif path == "/api/all/lakes":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["200"])[0])
                offset = int(params.get("offset", ["0"])[0])
                filters = {
                    key: params.get(key, [""])[0]
                    for key in [
                        "water_type",
                        "province",
                        "city",
                        "county",
                        "polygon_quality",
                        "metadata_quality",
                        "area_bucket",
                        "has_tci",
                        "has_name",
                        "min_area",
                        "max_area",
                    ]
                }
                self._json(self._all_lakes_payload(query=query, limit=limit, offset=offset, filters=filters))
            elif path == "/api/all/training-samples":
                self._json(self._all_training_samples_payload())
            elif path == "/api/all/training-patches":
                params = parse_qs(parsed.query)
                include = params.get("include", [""])[0]
                self._json(self._all_training_patches_payload(include=include))
            elif path == "/api/all/training-runs":
                self._json(self.training_runs.list())
            elif re.fullmatch(r"/api/all/training-runs/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.training_runs.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Training job not found")
                    return
                self._json(job)
            elif path == "/api/all/model-validation/models":
                self._json(self._all_model_validation_models_payload())
            elif path == "/api/all/model-validation/random":
                params = parse_qs(parsed.query)
                threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
                model_key = params.get("model", [""])[0]
                try:
                    self._json(self._all_model_validation_random(threshold=threshold, model_key=model_key))
                except (FileNotFoundError, ValueError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif path == "/api/lakes":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["200"])[0])
                offset = int(params.get("offset", ["0"])[0])
                filters = {
                    key: params.get(key, [""])[0]
                    for key in [
                        "water_type",
                        "province",
                        "city",
                        "county",
                        "polygon_quality",
                        "metadata_quality",
                        "area_bucket",
                        "has_tci",
                        "has_name",
                        "min_area",
                        "max_area",
                    ]
                }
                self._json(self.catalog.list_lakes(query=query, limit=limit, offset=offset, filters=filters))
            elif re.fullmatch(r"/api/lakes/[^/]+", path):
                lake_key = path.rsplit("/", 1)[-1]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.get_lake_detail(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/image.png", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                size = int(params.get("size", ["900"])[0])
                padding = float(params.get("padding", ["0.6"])[0])
                try:
                    payload, meta = self.catalog.image_for_lake(lake, size=size, padding=padding)
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("X-Image-Meta", json.dumps(meta, ensure_ascii=True))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif re.fullmatch(r"/api/lakes/[^/]+/tile-meta", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                try:
                    self._json(self.catalog.tile_meta_for_lake(lake, padding=padding))
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif re.fullmatch(r"/api/lakes/[^/]+/tiles/\d+/\d+/\d+\.png", path):
                match = re.fullmatch(r"/api/lakes/([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.png", path)
                lake = self.catalog.get_lake(match.group(1))
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                try:
                    payload, _meta = self.catalog.tile_png_for_lake(
                        lake,
                        z=int(match.group(2)),
                        x=int(match.group(3)),
                        y=int(match.group(4)),
                        padding=padding,
                    )
                except FileNotFoundError:
                    payload = blank_png(256)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif re.fullmatch(r"/api/lakes/[^/]+/esa", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json({"esa": self.catalog._esa_smoothed_layer(lake)})
            elif re.fullmatch(r"/api/lakes/[^/]+/jrc", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                threshold = int(params.get("threshold", ["75"])[0])
                self._json({"jrc": self.catalog._jrc_occurrence_layer(lake, threshold=threshold)})
            elif re.fullmatch(r"/api/lakes/[^/]+/sentinel/tiles", path):
                lake_key = path.split("/")[-3]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.sentinel_tiles_for_lake(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/context-water", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                min_area_km2 = float(params.get("min_area_km2", ["1.0"])[0])
                limit = int(params.get("limit", ["500"])[0])
                self._json(self.catalog.context_water_for_lake(lake, padding=padding, min_area_km2=min_area_km2, limit=limit))
            elif re.fullmatch(r"/api/lakes/[^/]+/local-labels", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.local_label_items(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/local-labels/[^/]+", path):
                parts = path.split("/")
                lake_key = parts[-3]
                label_id = parts[-1]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                try:
                    self._json(self.catalog.local_label_geojson(lake, label_id))
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif re.fullmatch(r"/api/lakes/[^/]+/imagery", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.imagery_for_lake(lake))
            elif re.fullmatch(r"/api/lakes/[^/]+/training-samples/readiness", path):
                lake_key = path.split("/")[-3]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                buffer_ratio = float(params.get("buffer_ratio", ["0.8"])[0])
                self._json(self.catalog.training_sample_readiness(lake, buffer_ratio=buffer_ratio))
            elif path == "/api/training-samples":
                self._json(self.catalog.list_training_samples())
            elif path == "/api/training-patches":
                params = parse_qs(parsed.query)
                include = params.get("include", [""])[0]
                self._json(self.catalog.list_training_patches(include=include))
            elif re.fullmatch(r"/api/(?:all/)?training-patches/export-jobs/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.patch_exports.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Patch export job not found")
                    return
                self._json(job)
            elif path == "/api/training-runs":
                self._json(self.training_runs.list())
            elif re.fullmatch(r"/api/training-runs/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.training_runs.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Training job not found")
                    return
                self._json(job)
            elif re.fullmatch(r"/api/training-patches/[^/]+/preview\.png", path):
                patch_id = path.split("/")[-2]
                try:
                    payload, content_type = self.catalog.training_patch_preview(patch_id)
                except (KeyError, FileNotFoundError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif path == "/api/model-validation/models":
                self._json(self.catalog.model_validation_models())
            elif path == "/api/model-validation/random":
                params = parse_qs(parsed.query)
                threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
                model_key = params.get("model", [""])[0]
                try:
                    self._json(self.catalog.model_validation_random(threshold=threshold, model_key=model_key))
                except (FileNotFoundError, ValueError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif re.fullmatch(r"/api/lakes/[^/]+/model-prediction", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                threshold = parse_float_or_default(params.get("threshold", ["0.5"])[0], 0.5)
                model_key = params.get("model", [""])[0]
                try:
                    self._json(self.catalog.model_prediction_for_lake(lake, threshold=threshold, model_key=model_key))
                except (FileNotFoundError, ValueError) as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif path == "/api/sentinel/products":
                params = parse_qs(parsed.query)
                tile = params.get("tile", [""])[0]
                if not tile:
                    self._error(HTTPStatus.BAD_REQUEST, "tile is required")
                    return
                lake = None
                lake_key = params.get("lake_id", [""])[0]
                if lake_key:
                    lake = self.catalog.get_lake(lake_key)
                default_start, default_end = default_sentinel_date_range()
                start = params.get("start", [default_start])[0]
                end = params.get("end", [default_end])[0]
                cloud = float(params.get("cloud", ["50"])[0])
                product_type = params.get("product_type", ["MSIL1C"])[0]
                limit = int(params.get("limit", ["50"])[0])
                products = query_copernicus_tile_products(tile, start, end, cloud, product_type, limit)
                products = self.catalog.enrich_products_for_lake(lake, products)
                enriched = []
                for product in products:
                    local = self.catalog.local_product_status(product.get("product_id"), product.get("name"))
                    enriched.append({**product, **local})
                products = enriched
                self._json({
                    "tile": str(tile).upper().removeprefix("T"),
                    "start": start,
                    "end": end,
                    "cloud": cloud,
                    "product_type": product_type,
                    "lake_id": lake.object_id if lake else None,
                    "products": products,
                })
            elif re.fullmatch(r"/api/sentinel/downloads/[^/]+", path):
                job_id = path.rsplit("/", 1)[-1]
                job = self.downloads.get(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Download job not found")
                    return
                self._json(job)
            elif re.fullmatch(r"/api/lakes/[^/]+/image-meta", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                try:
                    _, meta = self.catalog.image_for_lake(lake)
                    self._json(meta)
                except FileNotFoundError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
            elif path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "Not found")
            else:
                self._serve_file(STATIC_DIR / "index.html")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if path == "/api/sentinel/downloads":
                payload = self._read_json()
                product = payload.get("product") or payload
                if not product.get("product_id") or not product.get("name"):
                    self._error(HTTPStatus.BAD_REQUEST, "product_id and name are required")
                    return
                status = self.catalog.local_product_status(product.get("product_id"), product.get("name"))
                if status.get("downloaded"):
                    self._json({
                        "job_id": None,
                        "status": "completed",
                        "message": "产品已在本地",
                        "progress": 100,
                        "result": status,
                        "product": product,
                    })
                    return
                self._json(self.downloads.create(product))
            elif re.fullmatch(r"/api/lakes/[^/]+/imagery/active", path):
                lake_key = path.split("/")[-3]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                payload = self._read_json()
                tile = payload.get("tile")
                product = payload.get("product")
                if not tile or not product:
                    self._error(HTTPStatus.BAD_REQUEST, "tile and product are required")
                    return
                try:
                    result = self.catalog.set_active_imagery(tile, product, lake)
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json(result)
            elif re.fullmatch(r"/api/lakes/[^/]+/training-samples", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                payload = self._read_json()
                try:
                    result = self.catalog.create_training_sample(lake, payload)
                except ValueError as exc:
                    self._error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._json({"sample": result})
            elif path in {"/api/training-patches/export-jobs", "/api/all/training-patches/export-jobs"}:
                self._json(self.patch_exports.create(self._read_json()))
            elif path in {"/api/training-runs", "/api/all/training-runs"}:
                self._json(self.training_runs.create(self._read_json()))
            elif re.fullmatch(r"/api/(?:all/)?training-runs/[^/]+/cancel", path):
                job_id = path.split("/")[-2]
                job = self.training_runs.cancel(job_id)
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "Training job not found")
                    return
                self._json(job)
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if re.fullmatch(r"/api/training-samples/[^/]+", path):
                sample_id = path.rsplit("/", 1)[-1]
                try:
                    result = self.catalog.update_training_sample(sample_id, self._read_json())
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json({"sample": result})
            elif re.fullmatch(r"/api/training-patches/[^/]+", path):
                patch_id = path.rsplit("/", 1)[-1]
                try:
                    result = self.catalog.update_training_patch(patch_id, self._read_json())
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json({"patch": result})
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/regions/all/"):
                path = "/api/all" + path.removeprefix("/api/regions/all")
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.all_patch_exports
                self.training_runs = self.__class__.training_runs_by_scope["all"]
            elif path.startswith("/api/regions/"):
                context = self._route_region_path(path)
                if context is None:
                    return
                path, self.catalog, self.downloads = context
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]
            else:
                self.catalog = self.__class__.catalog
                self.downloads = self.__class__.downloads
                self.patch_exports = self.__class__.patch_exports_by_region[self.catalog.region.key]
                self.training_runs = self.__class__.training_runs_by_scope[self.catalog.region.key]

            if re.fullmatch(r"/api/training-samples/[^/]+", path):
                sample_id = path.rsplit("/", 1)[-1]
                try:
                    result = self.catalog.delete_training_sample(sample_id)
                except KeyError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json(result)
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _route_region_path(self, path: str) -> tuple[str, LakeCatalog, DownloadManager] | None:
        match = re.match(r"^/api/regions/([^/]+)(/.*)?$", path)
        if not match:
            return path, self.__class__.catalog, self.__class__.downloads
        region_key = match.group(1)
        if region_key not in self.__class__.catalogs:
            self._error(HTTPStatus.NOT_FOUND, f"Region not found: {region_key}")
            return None
        subpath = "/api" + (match.group(2) or "")
        return subpath, self.__class__.catalogs[region_key], self.__class__.downloads_by_region[region_key]

    def _regions_payload(self) -> dict:
        items = []
        for key, catalog in self.__class__.catalogs.items():
            region = catalog.region
            imagery_summary = catalog.imagery_inventory_summary()
            items.append(
                {
                    "key": key,
                    "name": region.name,
                    "default": key == DEFAULT_REGION_KEY,
                    "bounds": list(region.bounds) if region.bounds else None,
                    "ready": catalog.load_error is None,
                    "load_error": catalog.load_error,
                    "lake_count": len(catalog.lakes),
                    **imagery_summary,
                    "has_metadata": region.lake_metadata.exists(),
                    "has_osm_water": region.osm_water.exists(),
                    "metadata_path": display_path(region.lake_metadata),
                    "osm_water_path": display_path(region.osm_water),
                }
            )
        return {"default": DEFAULT_REGION_KEY, "items": items}

    def _all_lakes_payload(self, query: str, limit: int, offset: int, filters: dict) -> dict:
        merged = []
        total = 0
        for key, catalog in self.__class__.catalogs.items():
            payload = catalog.list_lakes(query=query, limit=10**9, offset=0, filters=filters)
            total += payload["total"]
            for item in payload["items"]:
                merged.append({
                    **item,
                    "region": key,
                    "region_name": catalog.region.name,
                })
        merged.sort(key=lambda item: (-(item.get("area_km2") or 0), item.get("region", ""), item.get("object_id", "")))
        page = merged[offset : offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": page,
            "all_regions": True,
        }

    def _all_training_samples_payload(self) -> dict:
        items = []
        total = 0
        for key, catalog in self.__class__.catalogs.items():
            payload = catalog.list_training_samples()
            total += payload.get("total", 0)
            for item in payload.get("items", []):
                items.append({**item, "region": key, "region_name": catalog.region.name})
        items.sort(key=lambda item: (item.get("region", ""), item.get("created_at", ""), item.get("sample_id", "")), reverse=True)
        return {"total": total, "items": items, "all_regions": True}

    def _all_training_patches_payload(self, include: str = "") -> dict:
        items = []
        included_count = 0
        excluded_count = 0
        for key, catalog in self.__class__.catalogs.items():
            payload = catalog.list_training_patches(include=include)
            included_count += payload.get("included_count", 0)
            excluded_count += payload.get("excluded_count", 0)
            for item in payload.get("items", []):
                items.append({**item, "region": key, "region_name": catalog.region.name})
        items.sort(key=lambda item: (item.get("region", ""), item.get("sample_id", ""), int(item.get("row_off") or 0), int(item.get("col_off") or 0)))
        return {
            "total": len(items),
            "included_count": included_count,
            "excluded_count": excluded_count,
            "items": items,
            "all_regions": True,
        }

    def _all_model_validation_models_payload(self) -> dict:
        items = []
        default_key = ""
        for path in iter_global_model_paths():
            key = global_model_key(path)
            label = f"全部区域 / {path.parent.name}/{path.name}"
            try:
                model = load_unet_checkpoint(path)
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "epoch": model.epoch,
                    "in_channels": model.in_channels,
                    "base_channels": model.base_channels,
                    "region": "all",
                    "region_name": "全部区域",
                    "scope": "all",
                    "default": path.name == "best.pt" and not default_key,
                }
            except Exception as exc:  # noqa: BLE001 - broken checkpoints should be visible, not fatal.
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "region": "all",
                    "region_name": "全部区域",
                    "scope": "all",
                    "error": f"{type(exc).__name__}: {exc}",
                    "default": False,
                }
            if item["default"] and not default_key:
                default_key = key
            items.append(item)
        for key, catalog in self.__class__.catalogs.items():
            payload = catalog.model_validation_models()
            for item in payload.get("items", []):
                if item.get("scope") == "all":
                    continue
                global_key = f"{key}/{item['key']}"
                out = {
                    **item,
                    "key": global_key,
                    "label": f"{catalog.region.name} / {item['label']}",
                    "region": key,
                    "region_name": catalog.region.name,
                    "scope": key,
                }
                if item.get("default") and not default_key:
                    default_key = global_key
                items.append(out)
        return {"region": "all", "default": default_key or (items[0]["key"] if items else ""), "items": items, "all_regions": True}

    def _all_model_validation_random(self, threshold: float = 0.5, model_key: str = "") -> dict:
        region_key, local_model_key = split_global_model_key(model_key)
        if not region_key:
            payload = self._all_model_validation_models_payload()
            if not payload["default"]:
                raise FileNotFoundError("No model weights found")
            region_key, local_model_key = split_global_model_key(payload["default"])
        if region_key == "all":
            return self._all_model_validation_random_global(threshold=threshold, model_key=local_model_key)
        if region_key not in self.__class__.catalogs:
            raise FileNotFoundError(f"Region not found for model: {region_key}")
        result = self.__class__.catalogs[region_key].model_validation_random(threshold=threshold, model_key=local_model_key)
        result["model"]["key"] = f"{region_key}/{result['model']['key']}"
        result["model"]["scope"] = region_key
        result["model"]["region_name"] = self.__class__.catalogs[region_key].region.name
        result["lake"]["region"] = region_key
        result["lake"]["region_name"] = self.__class__.catalogs[region_key].region.name
        return result

    def _all_model_validation_random_global(self, threshold: float = 0.5, model_key: str = "") -> dict:
        model_path = global_model_path_from_key(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Global model not found: {display_path(model_path)}")
        model = load_unet_checkpoint(model_path)
        catalogs = list(self.__class__.catalogs.items())
        random.shuffle(catalogs)
        skipped = []
        for region_key, catalog in catalogs:
            candidates = list(catalog.lakes)
            random.shuffle(candidates)
            for lake in candidates:
                rows = catalog._model_validation_rows(lake, model.in_channels)
                if not rows:
                    continue
                try:
                    prediction = catalog.model_prediction_for_lake(lake, threshold=threshold, rows=rows, model=model)
                except Exception as exc:  # noqa: BLE001 - keep looking for a usable validation target.
                    skipped.append(f"{region_key}/{lake.object_id}: {type(exc).__name__}: {exc}")
                    continue
                prediction["model"]["key"] = global_model_key(model_path)
                prediction["model"]["scope"] = "all"
                prediction["model"]["region_name"] = "全部区域"
                return {
                    "region": region_key,
                    "lake_id": lake.object_id,
                    "lake": {
                        **catalog._summary(lake),
                        "region": region_key,
                        "region_name": catalog.region.name,
                    },
                    "model": prediction["model"],
                    "prediction": prediction["prediction"],
                    "stats": prediction["stats"],
                    "imagery": prediction["imagery"],
                    "skipped_count": len(skipped),
                }
        raise FileNotFoundError(
            f"No lake with active imagery matching global model bands ({model.in_channels}): {display_path(model_path)}"
        )

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def _error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists():
            self._error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    disable_proxy_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    LakeHandler.catalogs = {key: LakeCatalog(region) for key, region in REGIONS.items()}
    LakeHandler.downloads_by_region = {
        key: DownloadManager(catalog) for key, catalog in LakeHandler.catalogs.items()
    }
    LakeHandler.patch_exports_by_region = {
        key: PatchExportManager(catalog) for key, catalog in LakeHandler.catalogs.items()
    }
    LakeHandler.all_patch_exports = PatchExportManager(catalogs=LakeHandler.catalogs)
    LakeHandler.training_runs_by_scope = {
        **{key: TrainingManager(key) for key in LakeHandler.catalogs},
        "all": TrainingManager("all"),
    }
    LakeHandler.catalog = LakeHandler.catalogs[DEFAULT_REGION_KEY]
    LakeHandler.downloads = LakeHandler.downloads_by_region[DEFAULT_REGION_KEY]
    LakeHandler.patch_exports = LakeHandler.patch_exports_by_region[DEFAULT_REGION_KEY]
    LakeHandler.training_runs = LakeHandler.training_runs_by_scope[DEFAULT_REGION_KEY]
    server = ThreadingHTTPServer((args.host, args.port), LakeHandler)
    print(f"Lake browser running: http://{args.host}:{args.port}")
    for key, catalog in LakeHandler.catalogs.items():
        status = "ready" if catalog.load_error is None else catalog.load_error
        imagery_summary = catalog.imagery_inventory_summary()
        print(
            f"Region {key}: {len(catalog.lakes)} lakes, "
            f"{imagery_summary['tci_tile_count']} imagery tiles, "
            f"{imagery_summary['active_imagery_count']} active imagery selections, {status}"
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
