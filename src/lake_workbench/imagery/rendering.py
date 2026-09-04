"""Site imagery mosaic metadata, PNG rendering, and XYZ tile rendering."""

import json

from rasterio.warp import transform_bounds
from shapely.geometry import box

from lake_workbench.geo import padded_bounds
from lake_workbench.imagery.display import blank_png
from lake_workbench.imagery.mosaic import render_tci_mosaic_png
from lake_workbench.imagery.tiles import (
    image_cache_key,
    mosaic_source_meta,
    render_tci_xyz_tile,
    rows_bounds,
    tile_cache_key,
    xyz_tile_bounds,
)


class ImageryRenderingMixin:
    def image_for_site(self, site, size: int = 900, padding: float = 0.3) -> tuple[bytes, dict]:
        render_bounds = padded_bounds(site.bbox, padding)
        tci_rows = self._tci_rows_for_site(site)
        cache_key = image_cache_key(site, size, padding, tci_rows)
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

    def tile_meta_for_site(self, site, padding: float = 0.8) -> dict:
        site_bounds = padded_bounds(site.bbox, padding)
        rows = self._tci_rows_for_site(site)
        tile_url_prefix = "" if self.region.key == self.default_region_key else f"/regions/{self.region.key}"
        return {
            **mosaic_source_meta(rows),
            "bounds": list(site_bounds),
            "site_bounds": list(site_bounds),
            "tile_bounds": list(rows_bounds(rows)),
            "center": list(site.center),
            "padding": padding,
            "tile_url": f"/api{tile_url_prefix}/sites/{site.site_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        }

    def tile_png_for_site(self, site, z: int, x: int, y: int, padding: float = 0.8, tile_size: int = 256) -> tuple[bytes, dict]:
        rows = self._tci_rows_for_site(site)
        bounds_3857 = xyz_tile_bounds(z, x, y)
        bounds_4326 = transform_bounds("EPSG:3857", "EPSG:4326", *bounds_3857, densify_pts=21)
        if not box(*bounds_4326).intersects(box(*rows_bounds(rows))):
            return blank_png(tile_size), {"empty": True, "bounds": list(bounds_4326)}
        cache_key = tile_cache_key(site, z, x, y, padding, rows)
        cache_png = self.region.tile_cache_dir / f"{cache_key}.png"
        if cache_png.exists():
            return cache_png.read_bytes(), {"cached": True, "bounds": list(bounds_4326)}
        payload = render_tci_xyz_tile(rows, bounds_3857, tile_size=tile_size)
        self.region.tile_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_png.write_bytes(payload)
        return payload, {"cached": False, "bounds": list(bounds_4326)}

    def _tci_rows_for_site(self, site) -> list[dict]:
        local_row = self._active_local_imagery_row(site)
        if local_row is not None:
            return [local_row]
        rows = []
        for item in self.sentinel_tiles_for_site(site)["tiles"]:
            row = self._active_imagery_row(item["tile"], site)
            if row is not None:
                rows.append(row)
        if rows:
            return rows
        candidate_tiles = [item["tile"] for item in self.tci_footprints if item["geometry"].intersects(box(*site.bbox))]
        if not candidate_tiles:
            raise FileNotFoundError(f"No active imagery for site {site.site_id}")
        return [self.tci_by_tile[tile] for tile in candidate_tiles]

    def imagery_for_site(self, site) -> dict:
        local_rows = self._local_imagery_rows_for_site(site)
        active = self._active_local_imagery_row(site)
        if local_rows:
            assets = []
            for row in local_rows:
                asset = self._imagery_product_payload(row, row.get("tile", ""), f"site:{site.site_id}", site_id=site.site_id)
                asset["active"] = bool(active and row.get("product") == active.get("product"))
                assets.append(asset)
            return {"site_id": site.site_id, "assets": assets, "selection_mode": "site_imagery"}

        tiles = self.sentinel_tiles_for_site(site)["tiles"]
        return {
            "site_id": site.site_id,
            "tiles": [{**tile, "products": self.imagery_products_for_tile(tile["tile"], site)} for tile in tiles],
            "selection_mode": "tile_fallback",
        }
