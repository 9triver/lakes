"""Render one or more raster products as a site image."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

from lake_workbench.imagery.display import display_band_indexes, to_display_rgb
from lake_workbench.imagery.tiles import mosaic_source_meta
from lake_workbench.imagery.validity import valid_pixel_mask


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
            png, meta = render_tci_png(
                ordered_rows[0]["tci_path"],
                bounds_wgs84,
                size=size,
                padding=0,
                row=ordered_rows[0],
            )
            meta.update(mosaic_fallback_meta(ordered_rows))
            return png, meta

        crs = srcs[0].crs
        west, south, east, north = bounds_wgs84
        left, bottom, right, top = transform_bounds(
            "EPSG:4326",
            crs,
            west,
            south,
            east,
            north,
            densify_pts=21,
        )
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

    buffer = io.BytesIO()
    image = Image.fromarray(np.moveaxis(to_display_rgb(mosaic), 0, -1), "RGB")
    image.save(buffer, format="PNG", optimize=True)
    height, width = mosaic.shape[1], mosaic.shape[2]
    left = out_transform.c
    top = out_transform.f
    right = left + out_transform.a * width
    bottom = top + out_transform.e * height
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = inverse.transform(left, bottom)
    east, north = inverse.transform(right, top)
    filled = valid_pixel_mask(mosaic, nodata=0)
    source = mosaic_source_meta(ordered_rows)
    return buffer.getvalue(), {
        "bounds": [west, south, east, north],
        "width": width,
        "height": height,
        "crs": str(crs),
        "tile": ",".join(source["tiles"]),
        "tiles": source["tiles"],
        "date": ",".join(source["dates"]),
        "dates": source["dates"],
        "product": ",".join(source["products"]),
        "products": source["products"],
        "valid_ratio": float(np.count_nonzero(filled) / filled.size) if filled.size else 0.0,
        "tci_path": source["tci_path"],
        "mosaic": True,
        "cached": False,
    }


def mosaic_fallback_meta(tci_rows: list[dict]) -> dict:
    source = mosaic_source_meta(tci_rows)
    return {
        "tile": source["tiles"][0] if source["tiles"] else "",
        "tiles": source["tiles"],
        "date": ",".join(source["dates"]),
        "dates": source["dates"],
        "product": ",".join(source["products"]),
        "products": source["products"],
        "valid_ratio": float(tci_rows[0].get("valid_ratio", 0) or 0) if tci_rows else 0.0,
        "tci_path": source["tci_path"],
        "mosaic": False,
        "mosaic_fallback": "mixed CRS",
    }


def render_tci_png(
    tci_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
    size: int,
    padding: float,
    row: dict | None = None,
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
        left = max(left, src.bounds.left)
        right = min(right, src.bounds.right)
        bottom = max(bottom, src.bounds.bottom)
        top = min(top, src.bounds.top)
        if right <= left or top <= bottom:
            raise ValueError(f"Site bbox does not overlap raster {tci_path}")
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        aspect = (right - left) / max(top - bottom, 1)
        out_width = size
        out_height = max(240, min(1200, round(size / max(aspect, 0.1))))
        if out_height > size:
            out_height = size
            out_width = max(240, min(1200, round(size * aspect)))
        indexes = display_band_indexes(src, row or {})
        data = src.read(
            indexes,
            window=window,
            out_shape=(3, out_height, out_width),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        masks = src.read_masks(
            indexes,
            window=window,
            out_shape=(3, out_height, out_width),
            resampling=Resampling.nearest,
            boundless=True,
        )
        nodata = tuple(src.nodatavals[index - 1] for index in indexes)
        valid = valid_pixel_mask(data, nodata, masks)
        data = np.where(valid[None, :, :], data, 0)
        image = Image.fromarray(np.moveaxis(to_display_rgb(data), 0, -1), "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        inverse = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        west, south = inverse.transform(left, bottom)
        east, north = inverse.transform(right, top)
        return buffer.getvalue(), {
            "bounds": [west, south, east, north],
            "width": out_width,
            "height": out_height,
            "crs": str(src.crs),
        }
