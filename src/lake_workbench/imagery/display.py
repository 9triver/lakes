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


def encode_rgb_png(data: Any) -> bytes:
    image = Image.fromarray(np.moveaxis(to_display_rgb(data), 0, -1), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
