"""Raster rendering and model-prediction helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.warp import reproject, transform_bounds, transform_geom as rasterio_transform_geom
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from lake_workbench.geo import boxes_intersect, transform_geom
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.models.runtime import predict_array
from lake_workbench.paths import PROJECT_ROOT


WEB_MERCATOR_LIMIT = 20037508.342789244
MODEL_VALIDATION_MAX_DIM = max(256, int(os.environ.get("LAKES_MODEL_VALIDATION_MAX_DIM", "2048")))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def predict_water_geojson(
    path: Path,
    model: Any,
    threshold: float = 0.5,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[dict, dict]:
    threshold = float(threshold)
    with rasterio.open(path) as src:
        if src.count < model.in_channels:
            raise ValueError(f"Raster has {src.count} bands, model needs {model.in_channels}: {display_path(path)}")
        window = None
        if bounds is not None:
            source_bounds = bounds
            if src.crs and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                source_bounds = transform_bounds("EPSG:4326", src.crs, *bounds, densify_pts=21)
            raster_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            if not boxes_intersect(source_bounds, raster_bounds):
                stats = {
                    "threshold": threshold,
                    "valid_pixels": 0,
                    "predicted_pixels": 0,
                    "predicted_ratio": 0.0,
                    "polygon_count": 0,
                    "area_km2": 0.0,
                    "mean_probability": 0.0,
                    "max_probability": 0.0,
                }
                return {"type": "FeatureCollection", "features": []}, stats
            left = max(source_bounds[0], raster_bounds[0])
            bottom = max(source_bounds[1], raster_bounds[1])
            right = min(source_bounds[2], raster_bounds[2])
            top = min(source_bounds[3], raster_bounds[3])
            window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        out_shape = None
        transform_scale = Affine.identity()
        read_height = int(window.height) if window is not None else src.height
        read_width = int(window.width) if window is not None else src.width
        if max(read_height, read_width) > MODEL_VALIDATION_MAX_DIM:
            scale = MODEL_VALIDATION_MAX_DIM / max(read_height, read_width)
            out_height = max(1, int(round(read_height * scale)))
            out_width = max(1, int(round(read_width * scale)))
            out_shape = (model.in_channels, out_height, out_width)
            transform_scale = Affine.scale(read_width / out_width, read_height / out_height)
        indexes = list(range(1, model.in_channels + 1))
        read_kwargs = {"indexes": indexes, "window": window, "resampling": Resampling.bilinear}
        if out_shape is not None:
            read_kwargs["out_shape"] = out_shape
        image = src.read(**read_kwargs).astype(np.float32)
        mask_kwargs = {"indexes": indexes, "window": window, "resampling": Resampling.nearest}
        if out_shape is not None:
            mask_kwargs["out_shape"] = (model.in_channels, image.shape[1], image.shape[2])
        masks = src.read_masks(**mask_kwargs)
        if image.shape[1] == 0 or image.shape[2] == 0:
            stats = {
                "threshold": threshold,
                "valid_pixels": 0,
                "predicted_pixels": 0,
                "predicted_ratio": 0.0,
                "polygon_count": 0,
                "area_km2": 0.0,
                "mean_probability": 0.0,
                "max_probability": 0.0,
            }
            return {"type": "FeatureCollection", "features": []}, stats
        nodata = tuple(src.nodatavals[index - 1] for index in indexes)
        valid = valid_pixel_mask(image, nodata, masks)
        probability = predict_array(model, image, valid=valid)
        predicted = (probability >= threshold) & valid
        transform = window_transform(window, src.transform) if window is not None else src.transform
        transform = transform * transform_scale
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


def image_cache_key(site: Any, size: int, padding: float, tci_rows: list[dict]) -> str:
    payload = {
        "site_id": site.site_id,
        "bbox": [round(value, 8) for value in site.bbox],
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


def tile_cache_key(site: Any, z: int, x: int, y: int, padding: float, tci_rows: list[dict]) -> str:
    payload = {
        "render_version": 2,
        "site_id": site.site_id,
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
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def xyz_tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    tiles = 2**int(z)
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
            indexes = display_band_indexes(src, row)
            raw = np.zeros((3, tile_size, tile_size), dtype=np.float32)
            reproject(
                source=rasterio.band(src, indexes),
                destination=raw,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:3857",
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
            masks = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            for mask_index, source_index in enumerate(indexes):
                source_mask = src.read_masks(indexes=[source_index])[0]
                reproject(
                    source=source_mask,
                    destination=masks[mask_index],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:3857",
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
            nodata = tuple(src.nodatavals[index - 1] for index in indexes)
            data = to_display_rgb(raw)
        valid = valid_pixel_mask(raw, nodata, masks) & ~filled
        if np.any(valid):
            output[:, valid] = data[:, valid]
            filled |= valid
        if np.all(filled):
            break
    rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.moveaxis(output, 0, -1)
    rgba[:, :, 3] = np.where(filled, 255, 0).astype(np.uint8)
    image = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def blank_png(tile_size: int = 256) -> bytes:
    image = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def display_band_indexes(src: Any, row: dict) -> list[int]:
    if src.count < 3:
        raise ValueError(f"Raster must contain at least three display bands: {src.name}")
    descriptions = [str(value or "").lower() for value in (src.descriptions or ())]

    def find_band(*tokens: str) -> int | None:
        for index, description in enumerate(descriptions, start=1):
            if any(token in description for token in tokens):
                return index
        return None

    blue = find_band("rhot_492", "b02", "blue")
    green = find_band("rhot_560", "b03", "green")
    red = find_band("rhot_665", "b04", "red")
    if blue and green and red and len({blue, green, red}) == 3:
        return [red, green, blue]
    if row.get("source") in {"local_img", "local_imagery"}:
        return [3, 2, 1]
    return [1, 2, 3]


def to_display_rgb(data: Any) -> np.ndarray:
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
        left, bottom, right, top = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
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
    image = Image.fromarray(np.moveaxis(display, 0, -1), "RGB")
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
    filled = valid_pixel_mask(mosaic, nodata=0)
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
