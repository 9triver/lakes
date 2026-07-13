"""Sentinel MGRS tile matching and product coverage enrichment."""

import pyogrio
from shapely.geometry import box, mapping

from lake_workbench.geo import geometry_coverage_ratio, lake_aoi_geometry, product_geometry
from lake_workbench.utils import display_path, metadata_tiles, parse_float_or_default


class SentinelCatalogMixin:
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

    def sentinel_tiles_for_lake(self, lake) -> dict:
        target = lake.geometry
        rows = []
        for tile in self._required_sentinel_tiles_for_lake(lake):
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
        return {"lake_id": lake.object_id, "aoi_bounds": list(lake.bbox), "tiles": rows}

    def _required_sentinel_tiles_for_lake(self, lake) -> list[str]:
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

    def enrich_products_for_lake(self, lake, products: list[dict]) -> list[dict]:
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
            )
        )
        return enriched
