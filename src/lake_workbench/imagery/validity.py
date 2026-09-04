"""Shared rules for deciding whether raster pixels are usable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def valid_pixel_mask(
    image: np.ndarray,
    nodata: Any = None,
    masks: np.ndarray | None = None,
    *,
    zero_is_invalid: bool = True,
) -> np.ndarray:
    """Return pixels for which every input band contains usable data.

    ``nodata`` may be one value shared by all bands or one value per band, as
    returned by Rasterio's ``DatasetReader.nodatavals``.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (bands, height, width)")
    valid = np.all(np.isfinite(image), axis=0)
    if nodata is not None:
        values = _band_nodata_values(nodata, image.shape[0])
        for band, value in zip(image, values, strict=True):
            if value is None:
                continue
            try:
                if np.isnan(value):
                    valid &= ~np.isnan(band)
                    continue
            except TypeError:
                pass
            valid &= band != value
    if masks is not None:
        if masks.shape != image.shape:
            raise ValueError("masks must have the same shape as image")
        valid &= np.all(masks > 0, axis=0)
    if zero_is_invalid:
        valid &= np.any(image != 0, axis=0)
    return valid


def _band_nodata_values(nodata: Any, band_count: int) -> list[Any]:
    if isinstance(nodata, np.ndarray) and nodata.ndim == 1:
        values = nodata.tolist()
        if len(values) != band_count:
            raise ValueError("nodata must contain one value per image band")
        return values
    if isinstance(nodata, Sequence) and not isinstance(nodata, (str, bytes)):
        values = list(nodata)
        if len(values) != band_count:
            raise ValueError("nodata must contain one value per image band")
        return values
    return [nodata] * band_count
