#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local web lake browser for Hunan sample lakes.

This is intentionally dependency-light on the web side: the HTTP server uses
Python's standard library, while GIS IO uses the project environment's
geopandas/rasterio stack.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
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
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.validation import make_valid

from lakes_browser.sentinel_download import (
    download_copernicus_product,
    product_date,
    product_tile_name,
    product_type_from_name,
    query_copernicus_tile_products,
    upsert_csv_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LAKES_DATA_DIR", PROJECT_ROOT / "data" / "raw")).expanduser().resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"
TCI_INDEX = DATA_DIR / "hunan_single_tiles" / "hunan_selected_best_coverage_tci_valid.csv"
OSM_WATER = DATA_DIR / "hunan_osm_water" / "hunan_water_raw.gpkg"
LAKE_METADATA = PROJECT_ROOT / "data" / "processed" / "hunan_lake_metadata.gpkg"
HYDROLAKES = DATA_DIR / "hydrolakes" / "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp"
ESA_WATER_MASK = DATA_DIR / "hunan_external_water" / "esa_worldcover" / "hunan_esa_water_mask.tif"
JRC_OCCURRENCE = DATA_DIR / "hunan_external_water" / "jrc_gsw" / "hunan_jrc_occurrence_clip.tif"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "image_clips"
TILE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tiles"
JRC_POLYGON_DIR = PROJECT_ROOT / "data" / "processed" / "jrc_polygons"
USER_SENTINEL_INDEX = PROJECT_ROOT / "data" / "processed" / "sentinel_products.csv"
ACTIVE_IMAGERY = PROJECT_ROOT / "data" / "processed" / "active_imagery.json"
SENTINEL_DOWNLOAD_DIR = DATA_DIR / "hunan_single_tiles" / "products"
WEB_MERCATOR_LIMIT = 20037508.342789244


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
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.base_tci_by_tile = self._load_tci_index()
        self.tci_by_tile = dict(self.base_tci_by_tile)
        self.user_tci_rows = self._load_user_tci_rows()
        self.active_imagery = self._load_active_imagery()
        self._rebuild_effective_tci()
        self.tci_footprints = self._load_tci_footprints()
        self.lakes = self._load_lakes()
        self._lake_lookup = self._build_lake_lookup()
        self._detail_cache: dict[str, dict] = {}

    def _load_tci_index(self) -> dict[str, dict]:
        tci = pd.read_csv(TCI_INDEX)
        rows = {}
        for row in tci.to_dict("records"):
            path = resolve_data_path(row["tci_path"])
            if not path.exists():
                continue
            rows[str(row["tile"]).upper()] = {
                "tile": str(row["tile"]).upper(),
                "date": str(row["date"]),
                "source": row.get("source", ""),
                "valid_ratio": float(row.get("valid_ratio", 0) or 0),
                "product": row.get("product", ""),
                "tci_path": path,
            }
        return rows

    def _load_user_tci_rows(self) -> dict[str, list[dict]]:
        rows: dict[str, list[dict]] = {}
        if not USER_SENTINEL_INDEX.exists():
            return rows
        table = pd.read_csv(USER_SENTINEL_INDEX)
        for row in table.to_dict("records"):
            path = resolve_data_path(row.get("tci_path", ""))
            if not path.exists():
                continue
            tile = str(row.get("tile", "")).upper().removeprefix("T")
            if not tile:
                continue
            item = {
                "tile": tile,
                "date": str(row.get("date") or product_date(row.get("product_name", ""))),
                "source": row.get("source", "user_download"),
                "valid_ratio": float(row.get("valid_ratio", 1) or 1),
                "product": row.get("product_name", ""),
                "product_id": row.get("product_id", ""),
                "cloud_cover": row.get("cloud_cover", ""),
                "downloaded_at": row.get("downloaded_at", ""),
                "tci_path": path,
            }
            rows.setdefault(tile, []).append(item)
        for tile in rows:
            rows[tile].sort(key=lambda item: (str(item.get("date", "")), str(item.get("product", ""))), reverse=True)
        return rows

    def _load_active_imagery(self) -> dict[str, str]:
        if not ACTIVE_IMAGERY.exists():
            return {}
        try:
            payload = json.loads(ACTIVE_IMAGERY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key).upper().removeprefix("T"): str(value) for key, value in payload.items() if value}

    def _save_active_imagery(self) -> None:
        ACTIVE_IMAGERY.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_IMAGERY.write_text(json.dumps(self.active_imagery, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rebuild_effective_tci(self) -> None:
        effective = dict(self.base_tci_by_tile)
        for tile, rows in self.user_tci_rows.items():
            active_product = self.active_imagery.get(tile)
            selected = next((row for row in rows if row.get("product") == active_product), None)
            if selected is None and tile not in effective and rows:
                selected = rows[0]
            if selected is not None:
                effective[tile] = selected
        self.tci_by_tile = effective

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

    def _load_lakes(self) -> list[LakeRecord]:
        if LAKE_METADATA.exists():
            return self._load_lakes_from_metadata()
        return self._load_lakes_from_osm()

    def _load_lakes_from_metadata(self) -> list[LakeRecord]:
        data = pyogrio.read_dataframe(LAKE_METADATA, layer="lake_metadata").to_crs("EPSG:4326")
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
        data = pyogrio.read_dataframe(OSM_WATER, layer="osm_water_polygons", columns=columns)
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
        detail = {
            **self._summary(lake),
            "properties": attrs,
            "layers": {
                "osm": {
                    "source": "OSM",
                    "geometry": mapping(lake.geometry),
                    "properties": attrs,
                },
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
        candidate_tiles = [item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*lake.bbox))]
        if not candidate_tiles:
            raise FileNotFoundError(f"No downloaded TCI for lake {lake.object_id}")
        tci_rows = [self.tci_by_tile[tile] for tile in candidate_tiles]
        cache_key = image_cache_key(lake, size, padding, tci_rows)
        cache_png = CACHE_DIR / f"{cache_key}.png"
        cache_meta = CACHE_DIR / f"{cache_key}.json"
        if cache_png.exists() and cache_meta.exists():
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            meta["cached"] = True
            return cache_png.read_bytes(), meta

        png, meta = render_tci_mosaic_png(tci_rows, render_bounds, size=size)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_png.write_bytes(png)
        cache_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return png, meta

    def tile_meta_for_lake(self, lake: LakeRecord, padding: float = 0.8) -> dict:
        bounds = padded_bounds(lake.bbox, padding)
        rows = self._tci_rows_for_lake(lake)
        return {
            **mosaic_source_meta(rows),
            "bounds": list(bounds),
            "center": list(lake.center),
            "padding": padding,
            "tile_url": f"/api/lakes/{lake.object_id}/tiles/{{z}}/{{x}}/{{y}}.png?padding={padding}",
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
        render_bounds = padded_bounds(lake.bbox, padding)
        if not box(*bounds_4326).intersects(box(*render_bounds)):
            return blank_png(tile_size), {"empty": True, "bounds": list(bounds_4326)}

        cache_key = tile_cache_key(lake, z, x, y, padding, rows)
        cache_png = TILE_CACHE_DIR / f"{cache_key}.png"
        if cache_png.exists():
            return cache_png.read_bytes(), {"cached": True, "bounds": list(bounds_4326)}

        payload = render_tci_xyz_tile(rows, bounds_3857, tile_size=tile_size)
        TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_png.write_bytes(payload)
        return payload, {"cached": False, "bounds": list(bounds_4326)}

    def _tci_rows_for_lake(self, lake: LakeRecord) -> list[dict]:
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
                    "products": self.imagery_products_for_tile(tile["tile"]),
                }
                for tile in tiles
            ],
        }

    def imagery_products_for_tile(self, tile: str) -> list[dict]:
        tile = str(tile).upper().removeprefix("T")
        products = []
        base = self.base_tci_by_tile.get(tile)
        if base:
            products.append(
                {
                    "tile": tile,
                    "product": base.get("product", ""),
                    "date": base.get("date", ""),
                    "source": base.get("source", "preloaded"),
                    "valid_ratio": base.get("valid_ratio"),
                    "tci_path": display_path(base["tci_path"]),
                    "active": self.tci_by_tile.get(tile, {}).get("product") == base.get("product"),
                    "downloaded": True,
                    "preloaded": True,
                }
            )
        for row in self.user_tci_rows.get(tile, []):
            products.append(
                {
                    "tile": tile,
                    "product": row.get("product", ""),
                    "product_id": row.get("product_id", ""),
                    "date": row.get("date", ""),
                    "source": row.get("source", "user_download"),
                    "cloud_cover": row.get("cloud_cover", ""),
                    "downloaded_at": row.get("downloaded_at", ""),
                    "valid_ratio": row.get("valid_ratio"),
                    "tci_path": display_path(row["tci_path"]),
                    "active": self.tci_by_tile.get(tile, {}).get("product") == row.get("product"),
                    "downloaded": True,
                    "preloaded": False,
                }
            )
        return products

    def local_product_status(self, product_id: str | None, product_name: str | None) -> dict:
        product_id = clean_optional(product_id)
        product_name = clean_optional(product_name)
        for rows in self.user_tci_rows.values():
            for row in rows:
                if (product_id and clean_optional(row.get("product_id")) == product_id) or (
                    product_name and clean_optional(row.get("product")) == product_name
                ):
                    return {"downloaded": True, "source": "user_download", "tci_path": display_path(row["tci_path"])}
        for row in self.base_tci_by_tile.values():
            if product_name and clean_optional(row.get("product")) == product_name:
                return {"downloaded": True, "source": "preloaded", "tci_path": display_path(row["tci_path"])}
        return {"downloaded": False}

    def set_active_imagery(self, tile: str, product_name: str) -> dict:
        tile = str(tile).upper().removeprefix("T")
        product_name = str(product_name)
        with self._lock:
            candidates = []
            base = self.base_tci_by_tile.get(tile)
            if base:
                candidates.append(base)
            candidates.extend(self.user_tci_rows.get(tile, []))
            selected = next((row for row in candidates if row.get("product") == product_name), None)
            if selected is None:
                raise KeyError(f"Product not found for tile {tile}: {product_name}")
            if base and product_name == base.get("product"):
                self.active_imagery.pop(tile, None)
            else:
                self.active_imagery[tile] = product_name
            self._save_active_imagery()
            self._rebuild_effective_tci()
            self.tci_footprints = self._load_tci_footprints()
        return {
            "tile": tile,
            "product": product_name,
            "active": True,
            "imagery": self.imagery_products_for_tile(tile),
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
            "valid_ratio": 1.0,
        }
        with self._lock:
            upsert_csv_row(USER_SENTINEL_INDEX, row, key="product_name")
            self.user_tci_rows = self._load_user_tci_rows()
            if tile and tile not in self.active_imagery and tile not in self.base_tci_by_tile:
                self.active_imagery[tile] = row["product_name"]
                self._save_active_imagery()
            self._rebuild_effective_tci()
            self.tci_footprints = self._load_tci_footprints()
        return row

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
        tiles = metadata_tiles(lake.properties.get("sentinel_tiles")) or [
            item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*lake.bbox))
        ]
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
            "has_tci": bool(lake.properties.get("has_tci", self._has_tci(lake))),
        }

    def _has_tci(self, lake: LakeRecord) -> bool:
        return any(item["geometry"].intersects(box(*lake.bbox)) for item in self.tci_footprints)

    def _has_real_name(self, lake: LakeRecord) -> bool:
        return any(
            clean_optional(lake.properties.get(key))
            for key in ["name", "name_zh", "name_en"]
        )

    def sentinel_tiles_for_lake(self, lake: LakeRecord) -> dict:
        tiles = metadata_tiles(lake.properties.get("sentinel_tiles")) or [
            item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*lake.bbox))
        ]
        rows = []
        for tile in tiles:
            row = self.tci_by_tile.get(tile)
            rows.append(
                {
                    "tile": tile,
                    "downloaded": row is not None,
                    "date": row.get("date") if row else None,
                    "product": row.get("product") if row else None,
                    "valid_ratio": row.get("valid_ratio") if row else None,
                    "tci_path": display_path(row["tci_path"]) if row else None,
                }
            )
        return {"lake_id": lake.object_id, "tiles": rows}

    def _match_hydrolakes(self, lake: LakeRecord) -> dict | None:
        xmin, ymin, xmax, ymax = lake.bbox
        pad_x = max((xmax - xmin) * 0.2, 0.01)
        pad_y = max((ymax - ymin) * 0.2, 0.01)
        try:
            candidates = pyogrio.read_dataframe(
                HYDROLAKES,
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
        if lake.area_km2 > 250:
            return {
                "source": "ESA WorldCover 2021 water mask, smoothed",
                "geometry": None,
                "properties": {
                    "water_id": f"ESA_{lake.object_id}",
                    "skipped": True,
                    "reason": "object too large for on-demand ESA polygonization",
                },
            }
        try:
            geom = lake.geometry.buffer(max(lake.bbox[2] - lake.bbox[0], lake.bbox[3] - lake.bbox[1]) * 0.15)
            with rasterio.open(ESA_WATER_MASK) as src:
                clipped, transform = mask(src, [mapping(geom)], crop=True, filled=True)
                arr = clipped[0]
                geoms = []
                for geom_json, value in shapes(arr.astype("uint8"), mask=(arr == 1), transform=transform):
                    if int(value) != 1:
                        continue
                    poly = shape(geom_json)
                    if poly.is_empty:
                        continue
                    if not poly.intersects(lake.geometry):
                        continue
                    geoms.append(poly)
        except Exception:
            return None
        if not geoms:
            return None
        source = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
        smoothed = smooth_water_geometry(source)
        return {
            "source": "ESA WorldCover 2021 water mask, smoothed",
            "geometry": mapping(smoothed),
            "properties": {"water_id": f"ESA_{lake.object_id}"},
        }

    def _jrc_occurrence_layer(self, lake: LakeRecord, threshold: int = 75) -> dict | None:
        threshold = max(1, min(100, int(threshold)))
        cached = read_jrc_polygon_cache(lake.object_id, threshold)
        if cached is not None:
            return cached
        if lake.area_km2 > 250:
            available = available_jrc_thresholds(lake.object_id)
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
        return build_jrc_occurrence_layer(lake, threshold)


def _jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def jrc_polygon_cache_path(lake_id: str, threshold: int) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", lake_id)
    return JRC_POLYGON_DIR / safe_id / f"jrc_occurrence_ge{threshold}.geojson"


def read_jrc_polygon_cache(lake_id: str, threshold: int) -> dict | None:
    path = jrc_polygon_cache_path(lake_id, threshold)
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


def write_jrc_polygon_cache(lake_id: str, threshold: int, layer: dict) -> Path:
    path = jrc_polygon_cache_path(lake_id, threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(layer, ensure_ascii=False), encoding="utf-8")
    return path


def available_jrc_thresholds(lake_id: str) -> list[int]:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", lake_id)
    folder = JRC_POLYGON_DIR / safe_id
    if not folder.exists():
        return []
    thresholds = []
    for path in folder.glob("jrc_occurrence_ge*.geojson"):
        match = re.search(r"ge(\d+)\.geojson$", path.name)
        if match:
            thresholds.append(int(match.group(1)))
    return sorted(thresholds)


def build_jrc_occurrence_layer(lake: LakeRecord, threshold: int) -> dict | None:
    threshold = max(1, min(100, int(threshold)))
    try:
        geom = lake.geometry.buffer(max(lake.bbox[2] - lake.bbox[0], lake.bbox[3] - lake.bbox[1]) * 0.2)
        with rasterio.open(JRC_OCCURRENCE) as src:
            clipped, transform = mask(src, [mapping(geom)], crop=True, filled=True)
            arr = clipped[0]
            water = (arr >= threshold) & (arr <= 100)
            geoms = []
            for geom_json, value in shapes(water.astype("uint8"), mask=water, transform=transform):
                if int(value) != 1:
                    continue
                poly = shape(geom_json)
                if poly.is_empty:
                    continue
                if not poly.intersects(lake.geometry):
                    continue
                geoms.append(poly)
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


def smooth_jrc_geometry(geom, lake_area_km2: float):
    if lake_area_km2 > 250:
        metric = transform_geom(geom, "EPSG:4326", "EPSG:3857")
        metric = make_valid(metric)
        parts = list(metric.geoms) if isinstance(metric, MultiPolygon) else [metric]
        min_area_m2 = 100_000
        kept = [part for part in parts if part.area >= min_area_m2]
        if not kept:
            kept = parts
        simplified = [make_valid(part.simplify(30, preserve_topology=True)) for part in kept]
        result = simplified[0] if len(simplified) == 1 else MultiPolygon(simplified)
        return transform_geom(result, "EPSG:3857", "EPSG:4326")
    return smooth_water_geometry(geom)


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


def parse_float(value) -> float | None:
    text = clean_optional(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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

            safe_dir, tci_path = download_copernicus_product(product, SENTINEL_DOWNLOAD_DIR, progress=progress)
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


def resolve_data_path(value) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "data_download" and parts[1] == "downloads":
        return DATA_DIR.joinpath(*parts[2:])
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def transform_geom(geom, src_crs: str, dst_crs: str):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def _transform(x, y, z=None):
        return transformer.transform(x, y)

    from shapely.ops import transform as shapely_transform

    return shapely_transform(_transform, geom)


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
            data = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
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
            indexes=[1, 2, 3],
            nodata=0,
            method="first",
            resampling=Resampling.bilinear,
        )
    finally:
        for src in srcs:
            src.close()

    rgb = np.moveaxis(mosaic, 0, -1).astype(np.uint8)
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
    filled = np.any(mosaic != 0, axis=0)
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
            [1, 2, 3],
            window=window,
            out_shape=(3, out_height, out_width),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        rgb = np.moveaxis(data, 0, -1).astype(np.uint8)
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
    catalog: LakeCatalog
    downloads: DownloadManager

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                self._serve_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                self._serve_file(STATIC_DIR / path.removeprefix("/static/"))
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
                payload, meta = self.catalog.image_for_lake(lake, size=size, padding=padding)
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
                self._json(self.catalog.tile_meta_for_lake(lake, padding=padding))
            elif re.fullmatch(r"/api/lakes/[^/]+/tiles/\d+/\d+/\d+\.png", path):
                match = re.fullmatch(r"/api/lakes/([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.png", path)
                lake = self.catalog.get_lake(match.group(1))
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                params = parse_qs(parsed.query)
                padding = float(params.get("padding", ["0.8"])[0])
                payload, _meta = self.catalog.tile_png_for_lake(
                    lake,
                    z=int(match.group(2)),
                    x=int(match.group(3)),
                    y=int(match.group(4)),
                    padding=padding,
                )
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
            elif re.fullmatch(r"/api/lakes/[^/]+/imagery", path):
                lake_key = path.split("/")[-2]
                lake = self.catalog.get_lake(lake_key)
                if lake is None:
                    self._error(HTTPStatus.NOT_FOUND, "Lake not found")
                    return
                self._json(self.catalog.imagery_for_lake(lake))
            elif path == "/api/sentinel/products":
                params = parse_qs(parsed.query)
                tile = params.get("tile", [""])[0]
                if not tile:
                    self._error(HTTPStatus.BAD_REQUEST, "tile is required")
                    return
                start = params.get("start", ["2025-06-01"])[0]
                end = params.get("end", ["2025-08-31"])[0]
                cloud = float(params.get("cloud", ["20"])[0])
                product_type = params.get("product_type", ["MSIL1C"])[0]
                limit = int(params.get("limit", ["50"])[0])
                products = query_copernicus_tile_products(tile, start, end, cloud, product_type, limit)
                products = [
                    {
                        **product,
                        **self.catalog.local_product_status(product.get("product_id"), product.get("name")),
                    }
                    for product in products
                ]
                self._json({
                    "tile": str(tile).upper().removeprefix("T"),
                    "start": start,
                    "end": end,
                    "cloud": cloud,
                    "product_type": product_type,
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
                _, meta = self.catalog.image_for_lake(lake)
                self._json(meta)
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001 - surface local diagnostics in MVP.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
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
                    result = self.catalog.set_active_imagery(tile, product)
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
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    LakeHandler.catalog = LakeCatalog()
    LakeHandler.downloads = DownloadManager(LakeHandler.catalog)
    server = ThreadingHTTPServer((args.host, args.port), LakeHandler)
    print(f"Lake browser running: http://{args.host}:{args.port}")
    print(f"Loaded lakes: {len(LakeHandler.catalog.lakes)}")
    print(f"Loaded TCI tiles: {len(LakeHandler.catalog.tci_by_tile)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
