"""Convert raster bands into web-displayable RGB and PNG payloads."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def blank_png(tile_size: int = 256) -> bytes:
    image = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


DISPLAY_STYLES = {'rgb': [2, 1, 0], 'nir': [3, 2, 1], 'swir': [4, 3, 2]}

def display_band_indexes(src: Any, row: dict, style: str = 'rgb') -> list[int]:
    if style not in DISPLAY_STYLES:
        raise ValueError('未知影像配色')
    if style != 'rgb':
        if src.count != 5 or row.get('source') not in {'local_img', 'local_imagery'}:
            raise ValueError('假彩色底图需要已核对的蓝绿红NIR-SWIR1五波段影像')
        return [band + 1 for band in DISPLAY_STYLES[style]]
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


HUNAN_RGB_RANGE = (262, 6040)


def hunan_rgb_enabled(region: str) -> bool:
    from os import environ

    return region == "hunan" and environ.get("LAKES_HUNAN_RGB") == "1"


def hunan_display_rgb(data: np.ndarray) -> np.ndarray:
    """Apply one shared RGB curve to local Hunan optical imagery."""
    low, high = HUNAN_RGB_RANGE
    values = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0)
    scaled = np.sqrt(np.clip((values - low) / (high - low), 0, 1))
    return np.rint(scaled * 255).astype(np.uint8)
