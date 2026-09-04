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
from lake_workbench.imagery.formats import is_local_imagery_source, local_label_path


def _site_id(value) -> str:
    return clean_optional(getattr(value, "site_id", value)) or ""


class ImageryInventoryMixin:
    def _load_tci_index(self) -> dict[str, dict]:
        if not self.region.tci_index.exists():
            return {}
        rows = {}
        for row in pd.read_csv(self.region.tci_index).to_dict("records"):
            path_text = clean_optional(row.get("tci_path"))
            path = resolve_data_path(path_text, self.region) if path_text else None
            tile = clean_optional(row.get("tile"))
            if not tile or not path:
                continue
            rows[tile.upper().removeprefix("T")] = {
                "tile": tile.upper().removeprefix("T"),
                "date": str(row["date"]),
                "source": row.get("source", ""),
                "valid_ratio": parse_float_or_default(row.get("valid_ratio"), 0.0),
                "product": row.get("product", ""),
                "cloud_cover": row.get("cloud_cover", ""),
                "tci_path": path,
                "missing": not path or not path.exists(),
            }
        return rows

    def _load_user_tci_rows(self) -> dict[str, list[dict]]:
        rows: dict[str, list[dict]] = {}
        if not self.region.user_sentinel_index.exists():
            return rows
        for row in pd.read_csv(self.region.user_sentinel_index).to_dict("records"):
            path_text = clean_optional(row.get("tci_path"))
            path = resolve_data_path(path_text, self.region) if path_text else None
            safe_path_text = clean_optional(row.get("safe_path")) or ""
            tile = str(row.get("tile", "")).upper().removeprefix("T")
            if not tile:
                continue
            site_id = clean_optional(row.get("site_id")) or ""
            item = {
                "tile": tile,
                "site_id": site_id,
                "date": clean_optional(row.get("date")) or product_date(row.get("product_name", "")),
                "source": clean_optional(row.get("source")) or "user_download",
                "valid_ratio": parse_float_or_default(row.get("valid_ratio"), 1.0),
                "product": clean_optional(row.get("product_name")) or "",
                "product_id": clean_optional(row.get("product_id")) or "",
                "cloud_cover": clean_optional(row.get("cloud_cover")) or "",
                "downloaded_at": clean_optional(row.get("downloaded_at")) or "",
                "safe_path": resolve_data_path(safe_path_text, self.region) if safe_path_text else None,
                "label_path": (
                    resolve_data_path(clean_optional(row.get("label_path")), self.region)
                    if clean_optional(row.get("label_path"))
                    else None
                ),
                "tci_path": path,
                "missing": not path or not path.exists(),
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
            if text.startswith("site:"):
                active[text] = str(value)
                continue
            if ":" in text:
                site_id, tile = text.split(":", 1)
                active[f"{site_id}:{tile.upper().removeprefix('T')}"] = str(value)
            else:
                active[text.upper().removeprefix("T")] = str(value)
        return active

    def _save_active_imagery(self) -> None:
        self.region.active_imagery.parent.mkdir(parents=True, exist_ok=True)
        self.region.active_imagery.write_text(json.dumps(self.active_imagery, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rebuild_effective_tci(self) -> None:
        effective = {}
        tiles = set(self.base_tci_by_tile) | set(self.user_tci_rows) | {
            key for key in self.active_imagery if not key.startswith("site:")
        }
        for tile in tiles:
            active_product = self.active_imagery.get(tile)
            selected = None
            base = self.base_tci_by_tile.get(tile)
            if base and base.get("product") == active_product and base.get("tci_path") and base["tci_path"].exists():
                selected = base
            if selected is None:
                selected = next(
                    (
                        row
                        for row in self.user_tci_rows.get(tile, [])
                        if row.get("product") == active_product
                        and row.get("tci_path")
                        and row["tci_path"].exists()
                    ),
                    None,
                )
            if selected is not None:
                effective[tile] = selected
        self.tci_by_tile = effective

    def _active_imagery_key(self, tile: str, site=None) -> str:
        tile = str(tile).upper().removeprefix("T")
        site_id = _site_id(site)
        if site_id and any(row.get("site_id") == site_id for row in self.user_tci_rows.get(tile, [])):
            return f"{site_id}:{tile}"
        return tile

    def _active_imagery_row(self, tile: str, site=None) -> dict | None:
        tile = str(tile).upper().removeprefix("T")
        active_key = self._active_imagery_key(tile, site)
        active_product = self.active_imagery.get(active_key)
        site_id = _site_id(site)
        site_product = self.active_imagery.get(f"site:{site_id}") if site_id else None
        if site_product:
            selected = next(
                (
                    row
                    for row in self.user_tci_rows.get(tile, [])
                    if row.get("product") == site_product
                    and (not row.get("site_id") or row.get("site_id") == site_id)
                    and row.get("tci_path")
                    and row["tci_path"].exists()
                ),
                None,
            )
            if selected is not None:
                return selected
        if not active_product:
            return None
        base = self.base_tci_by_tile.get(tile)
        if active_key == tile and base and base.get("product") == active_product and base.get("tci_path") and base["tci_path"].exists():
            return base
        site_id = active_key.split(":", 1)[0] if ":" in active_key else ""
        return next(
            (
                row
                for row in self.user_tci_rows.get(tile, [])
                if row.get("product") == active_product
                and (not site_id or not row.get("site_id") or row.get("site_id") == site_id)
                and row.get("tci_path")
                and row["tci_path"].exists()
            ),
            None,
        )

    def _imagery_asset_meta(self, row: dict, site_id: str = "") -> dict:
        source = clean_optional(row.get("source")) or "preloaded"
        row_site_id = clean_optional(row.get("site_id")) or ""
        if is_local_imagery_source(source):
            asset_type, asset_scope, asset_label = "site_native", "site", "区域影像"
        elif source == "preloaded":
            asset_type, asset_scope, asset_label = "preloaded_tile", "tile", "预置 Sentinel tile"
        else:
            asset_type, asset_scope, asset_label = "sentinel_tile", "tile", "已下载 Sentinel tile"
        return {
            "asset_type": asset_type,
            "asset_scope": asset_scope,
            "asset_label": asset_label,
            "is_site_native": asset_type == "site_native",
            "is_tile_product": asset_scope == "tile",
            "applies_to_site": not row_site_id or not site_id or row_site_id == site_id,
        }

    def _imagery_product_payload(self, row: dict, tile: str, active_key: str, site_id: str = "", preloaded: bool = False) -> dict:
        source = clean_optional(row.get("source")) or ("preloaded" if preloaded else "user_download")
        path = row.get("tci_path")
        available = bool(path and path.exists())
        payload = {
            "tile": tile,
            "site_id": row.get("site_id", ""),
            "product": row.get("product", ""),
            "product_id": row.get("product_id", ""),
            "date": row.get("date", ""),
            "source": source,
            "cloud_cover": row.get("cloud_cover", ""),
            "downloaded_at": row.get("downloaded_at", ""),
            "valid_ratio": row.get("valid_ratio"),
            "safe_path": display_path(row["safe_path"]) if row.get("safe_path") else "",
            "tci_path": display_path(path) if path else "",
            "active": self.active_imagery.get(active_key) == row.get("product"),
            "downloaded": available,
            "missing": bool(path) and not available,
            "missing_reason": "影像文件不存在" if path and not available else "",
            "preloaded": preloaded,
        }
        payload["asset_id"] = row.get("product_id") or row.get("product", "")
        if is_local_imagery_source(source) and path:
            label_path = row.get("label_path") or self._local_label_path(Path(path))
            if available and label_path.exists() and hasattr(self, "_local_label_item"):
                label = self._local_label_item(label_path)
                payload["label"] = label
                payload["label_id"] = label["id"]
            else:
                payload["label"] = None
                payload["label_id"] = ""
        payload.update(self._imagery_asset_meta(row, site_id=site_id))
        return payload

    @staticmethod
    def _local_label_path(image_path: Path) -> Path:
        """Find the label file for an IMG, including legacy suffix styles."""
        return local_label_path(image_path)

    def _local_imagery_rows_for_site(self, site) -> list[dict]:
        site_id = _site_id(site)
        rows = [
            row
            for tile_rows in self.user_tci_rows.values()
            for row in tile_rows
            if is_local_imagery_source(row.get("source")) and row.get("site_id") == site_id
        ]
        return sorted(rows, key=lambda item: (str(item.get("date", "")), str(item.get("product", ""))), reverse=True)

    def _active_local_imagery_row(self, site) -> dict | None:
        rows = [row for row in self._local_imagery_rows_for_site(site) if row.get("tci_path") and row["tci_path"].exists()]
        if not rows:
            return None
        active_product = self.active_imagery.get(f"site:{_site_id(site)}")
        selected = next((row for row in rows if row.get("product") == active_product or row.get("product_id") == active_product), None)
        if selected is not None:
            return selected
        # Preserve selections written by the older site/tile format.
        selected = next(
            (row for row in rows if self.active_imagery.get(f"{_site_id(site)}:{row.get('tile', '')}") == row.get("product")),
            None,
        )
        return selected or rows[0]

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

    def imagery_products_for_tile(self, tile: str, site=None) -> list[dict]:
        tile = str(tile).upper().removeprefix("T")
        active_key = self._active_imagery_key(tile, site)
        site_id = _site_id(site)
        has_site_products = bool(site_id and any(row.get("site_id") == site_id for row in self.user_tci_rows.get(tile, [])))
        products = []
        base = self.base_tci_by_tile.get(tile)
        if base and not has_site_products:
            products.append(self._imagery_product_payload(base, tile, active_key, site_id=site_id, preloaded=True))
        for row in self.user_tci_rows.get(tile, []):
            if site_id and row.get("site_id") and row.get("site_id") != site_id:
                continue
            products.append(self._imagery_product_payload(row, tile, active_key, site_id=site_id))
        return products

    def local_product_status(self, product_id: str | None, product_name: str | None) -> dict:
        product_id = clean_optional(product_id)
        product_name = clean_optional(product_name)
        for rows in self.user_tci_rows.values():
            for row in rows:
                if (product_id and clean_optional(row.get("product_id")) == product_id) or (
                    product_name and clean_optional(row.get("product")) == product_name
                ):
                    available = bool(row.get("tci_path") and row["tci_path"].exists())
                    return {
                        "downloaded": available,
                        "missing": not available,
                        "source": "user_download",
                        "tci_path": display_path(row["tci_path"]) if row.get("tci_path") else "",
                        "coverage_ratio": self.valid_ratio_for_tci(row["tci_path"]) if available else 0.0,
                        "coverage_basis": "pixels",
                    }
        for row in self.base_tci_by_tile.values():
            if product_name and clean_optional(row.get("product")) == product_name:
                available = bool(row.get("tci_path") and row["tci_path"].exists())
                return {
                    "downloaded": available,
                    "missing": not available,
                    "source": "preloaded",
                    "tci_path": display_path(row["tci_path"]) if row.get("tci_path") else "",
                    "coverage_ratio": self.valid_ratio_for_tci(row["tci_path"]) if available else 0.0,
                    "coverage_basis": "pixels",
                }
        return {"downloaded": False}

    def valid_ratio_for_tci(self, tci_path: Path) -> float:
        key = str(tci_path)
        if key not in self._valid_ratio_cache:
            self._valid_ratio_cache[key] = calculate_valid_ratio_for_tci(tci_path)
        return self._valid_ratio_cache[key]

    def prune_active_imagery(self) -> bool:
        """Drop active selections whose indexed product is no longer available."""
        valid = {}
        for key, product in self.active_imagery.items():
            if key.startswith("site:"):
                site_id = key.removeprefix("site:")
                found = any(
                    row.get("site_id") == site_id
                    and (row.get("product") == product or row.get("product_id") == product)
                    and row.get("tci_path")
                    and row["tci_path"].exists()
                    for rows in self.user_tci_rows.values()
                    for row in rows
                )
            elif ":" in key:
                site_id, tile = key.split(":", 1)
                found = any(
                    str(row.get("product")) == product
                    and (not site_id or not row.get("site_id") or row.get("site_id") == site_id)
                    and row.get("tci_path")
                    and row["tci_path"].exists()
                    for row in self.user_tci_rows.get(tile, [])
                ) or (
                    not site_id
                    and self.base_tci_by_tile.get(tile, {}).get("tci_path")
                    and self.base_tci_by_tile.get(tile, {})["tci_path"].exists()
                    and self.base_tci_by_tile.get(tile, {}).get("product") == product
                )
            else:
                base = self.base_tci_by_tile.get(key, {})
                found = (
                    base.get("product") == product
                    and base.get("tci_path")
                    and base["tci_path"].exists()
                ) or any(
                    row.get("product") == product and row.get("tci_path") and row["tci_path"].exists()
                    for row in self.user_tci_rows.get(key, [])
                )
            if found:
                valid[key] = product
        changed = valid != self.active_imagery
        if changed:
            self.active_imagery = valid
            self._save_active_imagery()
        return changed

    def set_active_imagery(self, tile: str, product_name: str, site=None) -> dict:
        tile = str(tile).upper().removeprefix("T")
        product_name = str(product_name)
        active_key = self._active_imagery_key(tile, site)
        site_id = _site_id(site)
        with self._lock:
            candidates = []
            base = self.base_tci_by_tile.get(tile)
            if base and ":" not in active_key:
                candidates.append(base)
            candidates.extend(
                row for row in self.user_tci_rows.get(tile, []) if not site_id or not row.get("site_id") or row.get("site_id") == site_id
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
            "site_id": site_id,
            "product": product_name,
            "active": True,
            "imagery": self.imagery_products_for_tile(tile, site),
        }

    def set_active_site_imagery(self, site, asset_id: str = "", product_name: str = "") -> dict:
        rows = self._local_imagery_rows_for_site(site)
        selected = next(
            (
                row
                for row in rows
                if (asset_id and (row.get("product_id") == asset_id or row.get("product") == asset_id))
                or (product_name and row.get("product") == product_name)
            ),
            None,
        )
        if selected is None:
            raise KeyError(f"Imagery asset not found for site {_site_id(site)}: {asset_id or product_name}")
        with self._lock:
            self.active_imagery[f"site:{_site_id(site)}"] = selected["product"]
            self._save_active_imagery()
            self._rebuild_effective_tci()
            self.tci_footprints = self._load_tci_footprints()
        payload = self._imagery_product_payload(selected, selected.get("tile", ""), f"site:{_site_id(site)}", site_id=_site_id(site))
        payload["active"] = True
        return {"site_id": _site_id(site), "asset": payload}

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
