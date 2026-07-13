"""Local and downloaded imagery inventory, selection, and registration."""

import json
import time
from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import box

from lake_workbench.sentinel.download import (
    product_date,
    product_tile_name,
    product_type_from_name,
    upsert_csv_row,
    valid_ratio_for_tci as calculate_valid_ratio_for_tci,
)
from lake_workbench.utils import (
    clean_optional,
    display_path,
    parse_float_or_default,
    resolve_data_path,
)


def _lake_id(value) -> str:
    return clean_optional(getattr(value, "object_id", value)) or ""


class ImageryInventoryMixin:
    def _load_tci_index(self) -> dict[str, dict]:
        if not self.region.tci_index.exists():
            return {}
        rows = {}
        for row in pd.read_csv(self.region.tci_index).to_dict("records"):
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
        for row in pd.read_csv(self.region.user_sentinel_index).to_dict("records"):
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

    def _active_imagery_key(self, tile: str, lake=None) -> str:
        tile = str(tile).upper().removeprefix("T")
        lake_id = _lake_id(lake)
        if lake_id and any(row.get("lake_id") == lake_id for row in self.user_tci_rows.get(tile, [])):
            return f"{lake_id}:{tile}"
        return tile

    def _active_imagery_row(self, tile: str, lake=None) -> dict | None:
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
            asset_type, asset_scope, asset_label = "lake_native", "lake", "本体影像"
        elif source == "preloaded":
            asset_type, asset_scope, asset_label = "preloaded_tile", "tile", "预置 Sentinel tile"
        else:
            asset_type, asset_scope, asset_label = "sentinel_tile", "tile", "已下载 Sentinel tile"
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

    def imagery_inventory_summary(self) -> dict:
        product_tiles = set(self.base_tci_by_tile) | set(self.user_tci_rows)
        return {
            "tci_tile_count": len(product_tiles),
            "active_imagery_count": len(self.active_imagery),
            "active_tile_count": len({key.rsplit(":", 1)[-1] for key in self.active_imagery}),
            "product_count": sum(len(rows) for rows in self.user_tci_rows.values()) + len(self.base_tci_by_tile),
        }

    def imagery_products_for_tile(self, tile: str, lake=None) -> list[dict]:
        tile = str(tile).upper().removeprefix("T")
        active_key = self._active_imagery_key(tile, lake)
        lake_id = _lake_id(lake)
        has_lake_products = bool(lake_id and any(row.get("lake_id") == lake_id for row in self.user_tci_rows.get(tile, [])))
        products = []
        base = self.base_tci_by_tile.get(tile)
        if base and not has_lake_products:
            products.append(self._imagery_product_payload(base, tile, active_key, lake_id=lake_id, preloaded=True))
        for row in self.user_tci_rows.get(tile, []):
            if lake_id and row.get("lake_id") and row.get("lake_id") != lake_id:
                continue
            products.append(self._imagery_product_payload(row, tile, active_key, lake_id=lake_id))
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
        if key not in self._valid_ratio_cache:
            self._valid_ratio_cache[key] = calculate_valid_ratio_for_tci(tci_path)
        return self._valid_ratio_cache[key]

    def set_active_imagery(self, tile: str, product_name: str, lake=None) -> dict:
        tile = str(tile).upper().removeprefix("T")
        product_name = str(product_name)
        active_key = self._active_imagery_key(tile, lake)
        lake_id = _lake_id(lake)
        with self._lock:
            candidates = []
            base = self.base_tci_by_tile.get(tile)
            if base and ":" not in active_key:
                candidates.append(base)
            candidates.extend(
                row for row in self.user_tci_rows.get(tile, []) if not lake_id or not row.get("lake_id") or row.get("lake_id") == lake_id
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
