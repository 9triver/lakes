"""Helpers for discovering supported local raster formats."""

from __future__ import annotations

from pathlib import Path


LOCAL_RASTER_SUFFIXES = (".tif", ".tiff", ".img")
_FORMAT_PRIORITY = {suffix: index for index, suffix in enumerate(LOCAL_RASTER_SUFFIXES)}
LOCAL_LABEL_SUFFIXES = (".Swater.shp", "_Swater.shp")


def is_local_imagery_source(value: object) -> bool:
    """Accept the legacy source name and the normalized local name."""
    return str(value or "").strip().lower() in {"local_img", "local_imagery"}


def local_imagery_paths(directory: Path) -> list[Path]:
    """Return one raster per product stem, preferring GeoTIFF over ENVI IMG."""
    selected: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in _FORMAT_PRIORITY:
            continue
        key = path.stem.casefold()
        current = selected.get(key)
        if current is None or _FORMAT_PRIORITY[path.suffix.lower()] < _FORMAT_PRIORITY[current.suffix.lower()]:
            selected[key] = path
    return sorted(selected.values(), key=lambda path: path.name.casefold())


def local_imagery_paths_from_roots(roots: list[Path]) -> list[Path]:
    """Merge local imagery roots, preferring earlier roots and GeoTIFF within a root."""
    selected: dict[str, tuple[tuple[int, int], Path]] = {}
    for root_index, root in enumerate(roots):
        if not root.is_dir():
            continue
        for path in local_imagery_paths(root):
            key = path.stem.casefold()
            candidate_rank = (root_index, _FORMAT_PRIORITY[path.suffix.lower()])
            current = selected.get(key)
            if current is None or candidate_rank < current[0]:
                selected[key] = (candidate_rank, path)
    return sorted((path for _rank, path in selected.values()), key=lambda path: path.name.casefold())


def local_label_path(image_path: Path, fallback_dirs: list[Path] | None = None) -> Path:
    """Find a same-stem local label beside an image or in fallback dirs."""
    directories = [image_path.parent, *(fallback_dirs or [])]
    candidates = [
        directory / f"{image_path.stem}{suffix}"
        for directory in directories
        for suffix in LOCAL_LABEL_SUFFIXES
    ]
    return next((path for path in candidates if path.exists()), candidates[0])
