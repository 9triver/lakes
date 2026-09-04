"""Logical-patch lookup and source-image preview operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

from lake_workbench.geo import transform_geom
from lake_workbench.imagery.display import blank_png
from lake_workbench.imagery.tiles import render_tci_xyz_tile, xyz_tile_bounds
from lake_workbench.utils import (
    clean_optional,
    display_path,
    read_csv_records,
    resolve_data_path,
    truthy_flag,
)


class TrainingPatchCatalogMixin:
    """Read and render logical patches stored for a region."""

    region: Any

    def list_logical_patches(
        self,
        manifest_path: Path,
        include: str = "",
        site_id: str = "",
        sample_id: str = "",
        image_index: str = "",
    ) -> dict:
        rows = []
        for row in read_csv_records(manifest_path):
            item = self._logical_patch_summary(row, manifest_path)
            if include == "included" and not item["included"]:
                continue
            if include == "excluded" and item["included"]:
                continue
            if site_id and item.get("site_id") != site_id:
                continue
            if sample_id and item.get("sample_id") != sample_id:
                continue
            if image_index != "" and str(item.get("image_index")) != str(image_index):
                continue
            rows.append(item)
        rows.sort(
            key=lambda item: (
                item.get("sample_id", ""),
                int(item.get("row_off") or 0),
                int(item.get("col_off") or 0),
            )
        )
        included_count = sum(1 for item in rows if item["included"])
        return {
            "total": len(rows),
            "included_count": included_count,
            "excluded_count": len(rows) - included_count,
            "items": rows,
        }

    def logical_patch_by_id(self, patch_id: str, manifest_path: Path) -> dict:
        patch_id = str(patch_id)
        for row in read_csv_records(manifest_path):
            if row.get("logical_patch_id") == patch_id:
                return self._logical_patch_summary(row, manifest_path)
        raise KeyError(f"logical patch not found: {patch_id}")

    def _logical_patch_summary(self, row: dict, manifest_path: Path) -> dict:
        patch_id = row.get("logical_patch_id", "")
        preview_path = (
            resolve_data_path(row.get("preview_path", ""), self.region)
            if row.get("preview_path")
            else None
        )
        preview_base_path = (
            resolve_data_path(row.get("preview_base_path", ""), self.region)
            if row.get("preview_base_path")
            else None
        )
        image_path = (
            resolve_data_path(row.get("image_path", ""), self.region)
            if row.get("image_path")
            else None
        )
        label_path = (
            resolve_data_path(row.get("label_path", ""), self.region)
            if row.get("label_path")
            else None
        )
        review_status = clean_optional(row.get("review_status"))
        include_value = (
            "true"
            if review_status == "included"
            else (
                "false"
                if review_status == "excluded"
                else clean_optional(row.get("include") or row.get("included"))
            )
        )
        included = (
            True if include_value is None else truthy_flag(include_value, default=True)
        )
        site_id = row.get("site_id", "")
        site = self.get_site(site_id) if site_id else None
        display_name = site.display_name if site else row.get("site_name") or site_id
        geometry = None
        try:
            source_box = box(
                float(row["bounds_left"]),
                float(row["bounds_bottom"]),
                float(row["bounds_right"]),
                float(row["bounds_top"]),
            )
            geometry = mapping(
                transform_geom(source_box, row.get("crs") or "EPSG:4326", "EPSG:4326")
            )
        except (KeyError, TypeError, ValueError):
            geometry = None
        return {
            **row,
            "patch_id": patch_id,
            "logical_patch_id": patch_id,
            "site_id": site_id,
            "site_display_name": display_name,
            "included": included,
            "include": "true" if included else "false",
            "manifest_path": display_path(manifest_path),
            "geometry": geometry,
            "preview_exists": bool(preview_path and preview_path.exists()),
            "overlay_available": bool(
                (
                    preview_path
                    and preview_base_path
                    and preview_path.exists()
                    and preview_base_path.exists()
                )
                or (
                    image_path
                    and label_path
                    and image_path.exists()
                    and label_path.exists()
                )
            ),
            "preview_url": "",
        }

    def logical_patch_source_meta(self, patch_id: str, manifest_path: Path) -> dict:
        patch = self.logical_patch_by_id(patch_id, manifest_path)
        path = resolve_data_path(patch.get("image_path", ""), self.region)
        if not path.exists():
            raise FileNotFoundError(f"logical patch source image not found: {patch_id}")
        with rasterio.open(path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        return {
            "patch_id": patch_id,
            "bounds": list(bounds),
            "tile_url": f"/api/regions/{self.region.key}/sites/{patch.get('site_id')}/logical-patch-source/tiles/{{z}}/{{x}}/{{y}}.png?patch_id={patch_id}",
        }

    def logical_patch_source_tile(
        self, patch_id: str, z: int, x: int, y: int, manifest_path: Path
    ) -> bytes:
        patch = self.logical_patch_by_id(patch_id, manifest_path)
        path = resolve_data_path(patch.get("image_path", ""), self.region)
        if not path.exists():
            raise FileNotFoundError(f"logical patch source image not found: {patch_id}")
        bounds_3857 = xyz_tile_bounds(z, x, y)
        with rasterio.open(path) as src:
            source_bounds = transform_bounds(
                src.crs, "EPSG:3857", *src.bounds, densify_pts=21
            )
        if not box(*source_bounds).intersects(box(*bounds_3857)):
            return blank_png(256)
        return render_tci_xyz_tile(
            [{"tci_path": path, "source": "local_imagery", "valid_ratio": 1}],
            bounds_3857,
        )
