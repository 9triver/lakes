"""Fetch and align cached XYZ map tiles for automatic label evidence."""

from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject


WEB_MERCATOR_LIMIT = 20037508.342789244
MAX_LATITUDE = 85.05112878
DEFAULT_OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_OSM_PROXY = "http://192.168.30.107:7897"


@dataclass(frozen=True)
class BasemapEvidence:
    rgb: np.ndarray
    valid: np.ndarray
    zoom: int
    tile_count: int
    provider: str
    tile_url: str


def aligned_osm_rgb(
    bounds_wgs84: tuple[float, float, float, float],
    requested_zoom: int,
    target_crs: object,
    target_transform: Affine,
    target_shape: tuple[int, int],
    cache_dir: Path,
    *,
    max_tiles: int = 64,
) -> BasemapEvidence:
    """Download and align standard OSM map tiles for water evidence."""
    tile_url = os.environ.get("LAKES_OSM_TILE_URL", DEFAULT_OSM_TILE_URL).strip()
    proxy = os.environ.get("LAKES_OSM_PROXY", DEFAULT_OSM_PROXY).strip()
    return _aligned_xyz_rgb(
        bounds_wgs84,
        requested_zoom,
        target_crs,
        target_transform,
        target_shape,
        cache_dir,
        tile_url=tile_url,
        cache_key="osm_standard",
        cache_suffix="png",
        provider="osm_standard",
        max_tiles=max_tiles,
        proxy=proxy,
    )


def osm_water_mask(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Identify the blue water palette used by the standard OSM renderer."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if valid.shape != rgb.shape[:2]:
        raise ValueError("valid must match the first two rgb dimensions")
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    return valid & (blue >= 145) & (green >= 135) & (blue - red >= 20) & (green - red >= 15)


def _aligned_xyz_rgb(
    bounds_wgs84: tuple[float, float, float, float],
    requested_zoom: int,
    target_crs: object,
    target_transform: Affine,
    target_shape: tuple[int, int],
    cache_dir: Path,
    *,
    tile_url: str,
    cache_key: str,
    cache_suffix: str,
    provider: str,
    max_tiles: int,
    proxy: str = "",
) -> BasemapEvidence:
    zoom, tiles = bounded_tile_range(bounds_wgs84, requested_zoom, max_tiles=max_tiles)
    xs = [tile[0] for tile in tiles]
    ys = [tile[1] for tile in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mosaic = np.zeros(
        ((max_y - min_y + 1) * 256, (max_x - min_x + 1) * 256, 3),
        dtype=np.uint8,
    )
    source_valid = np.zeros(mosaic.shape[:2], dtype=np.uint8)
    session = requests.Session()
    session.trust_env = False
    if proxy:
        normalized_proxy = _normalize_proxy(proxy)
        session.proxies = {"http": normalized_proxy, "https": normalized_proxy}
    for x, y in tiles:
        payload = _cached_tile(
            session, tile_url, cache_dir, zoom, x, y, cache_key, cache_suffix
        )
        with Image.open(io.BytesIO(payload)) as image:
            tile = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.uint8)
        row = (y - min_y) * 256
        column = (x - min_x) * 256
        mosaic[row : row + 256, column : column + 256] = tile
        source_valid[row : row + 256, column : column + 256] = 1

    tile_span = (WEB_MERCATOR_LIMIT * 2) / (2**zoom)
    pixel_size = tile_span / 256
    source_transform = Affine(
        pixel_size,
        0,
        -WEB_MERCATOR_LIMIT + min_x * tile_span,
        0,
        -pixel_size,
        WEB_MERCATOR_LIMIT - min_y * tile_span,
    )
    height, width = target_shape
    aligned = np.zeros((height, width, 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=mosaic[..., band],
            destination=aligned[..., band],
            src_transform=source_transform,
            src_crs="EPSG:3857",
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    aligned_valid = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=source_valid,
        destination=aligned_valid,
        src_transform=source_transform,
        src_crs="EPSG:3857",
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    return BasemapEvidence(
        rgb=aligned,
        valid=aligned_valid > 0,
        zoom=zoom,
        tile_count=len(tiles),
        provider=provider,
        tile_url=tile_url,
    )


def _normalize_proxy(proxy: str) -> str:
    return proxy if "://" in proxy else f"http://{proxy}"


def bounded_tile_range(
    bounds_wgs84: tuple[float, float, float, float],
    requested_zoom: int,
    *,
    max_tiles: int,
) -> tuple[int, list[tuple[int, int]]]:
    west, south, east, north = bounds_wgs84
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid WGS84 map extent")
    zoom = min(19, max(4, int(round(requested_zoom))))
    while True:
        x0, y0 = _tile_coordinate(west, north, zoom)
        x1, y1 = _tile_coordinate(
            math.nextafter(east, west), math.nextafter(south, north), zoom
        )
        tiles = [
            (x, y)
            for y in range(min(y0, y1), max(y0, y1) + 1)
            for x in range(min(x0, x1), max(x0, x1) + 1)
        ]
        if len(tiles) <= max_tiles or zoom <= 4:
            return zoom, tiles
        zoom -= 1


def _tile_coordinate(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    latitude = max(-MAX_LATITUDE, min(MAX_LATITUDE, latitude))
    count = 2**zoom
    x = int(math.floor((longitude + 180.0) / 360.0 * count))
    radians = math.radians(latitude)
    y = int(
        math.floor(
            (1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * count
        )
    )
    return max(0, min(count - 1, x)), max(0, min(count - 1, y))


def _cached_tile(
    session: requests.Session,
    tile_url: str,
    cache_dir: Path,
    zoom: int,
    x: int,
    y: int,
    cache_key: str,
    cache_suffix: str,
) -> bytes:
    path = cache_dir / "basemap" / cache_key / str(zoom) / str(x) / f"{y}.{cache_suffix}"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    response = session.get(
        tile_url.format(z=zoom, x=x, y=y),
        timeout=(5, 20),
        headers={"User-Agent": "LakesWorkbench/0.1"},
    )
    response.raise_for_status()
    payload = response.content
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return payload
