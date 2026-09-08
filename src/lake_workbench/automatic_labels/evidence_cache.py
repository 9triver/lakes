"""Disk-backed caches for reusable automatic-label evidence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from affine import Affine

from lake_workbench.automatic_labels.basemap import BasemapEvidence, osm_water_mask
from lake_workbench.automatic_labels.spectral import (
    SPECTRAL_ALGORITHM_VERSION,
    read_spectral_evidence,
)


EVIDENCE_CACHE_SCHEMA = "automatic_label_evidence_v2"
OSM_WATER_MASK_VERSION = "standard_osm_blue_palette_v1"
INCOMPLETE_OSM_CACHE_SECONDS = 300


@dataclass(frozen=True)
class OsmMaskEvidence:
    """OSM water and coverage masks aligned to one spectral grid."""

    water: np.ndarray
    valid: np.ndarray
    zoom: int
    tile_count: int
    available_tile_count: int
    missing_tile_count: int
    provider: str
    tile_url: str


@dataclass(frozen=True)
class SpectralLabelEvidence:
    """Minimal spectral evidence required by generated label variants."""

    water_score: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    bands: dict[str, int]
    diagnostics: dict[str, Any]
    transform: Affine
    crs: Any

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape


def cached_spectral_evidence(
    cache_dir: Path,
    image_path: Path,
    bounds: tuple[float, float, float, float],
    *,
    max_dimension: int,
    threshold: float,
    quality_path: Path | None,
) -> tuple[SpectralLabelEvidence, str, bool]:
    """Return reusable spectral evidence and its deterministic cache key."""
    identity = {
        "schema": EVIDENCE_CACHE_SCHEMA,
        "kind": "spectral",
        "algorithm": SPECTRAL_ALGORITHM_VERSION,
        "image": _path_identity(image_path),
        "quality": _path_identity(quality_path) if quality_path else None,
        "bounds": [float(value) for value in bounds],
        "max_dimension": int(max_dimension),
        "threshold": float(threshold),
    }
    key = _identity_key(identity)
    path = cache_dir / "automatic_labels" / "spectral" / f"{key}.npz"
    cached = _load_spectral(path)
    if cached is not None:
        return cached, key, True
    generated = read_spectral_evidence(
        image_path,
        bounds,
        max_dimension=max_dimension,
        threshold=threshold,
        quality_path=quality_path,
    )
    evidence = SpectralLabelEvidence(
        water_score=generated.water_score,
        labels=generated.labels,
        valid=generated.valid,
        bands=generated.bands,
        diagnostics=generated.diagnostics,
        transform=generated.transform,
        crs=generated.crs,
    )
    _write_spectral(path, evidence)
    return evidence, key, False


def cached_osm_mask(
    cache_dir: Path,
    spectral_key: str,
    bounds: tuple[float, float, float, float],
    requested_zoom: int,
    tile_url: str,
    producer: Callable[[], BasemapEvidence],
) -> tuple[OsmMaskEvidence, bool]:
    """Return one aligned OSM water mask shared by all fusion variants."""
    identity = {
        "schema": EVIDENCE_CACHE_SCHEMA,
        "kind": "osm_water_mask",
        "mask_version": OSM_WATER_MASK_VERSION,
        "spectral_key": spectral_key,
        "bounds": [float(value) for value in bounds],
        "requested_zoom": int(requested_zoom),
        "tile_url": tile_url,
    }
    key = _identity_key(identity)
    path = cache_dir / "automatic_labels" / "osm" / f"{key}.npz"
    cached = _load_osm(path)
    if cached is not None:
        return cached, True
    basemap = producer()
    evidence = OsmMaskEvidence(
        water=osm_water_mask(basemap.rgb, basemap.valid),
        valid=basemap.valid,
        zoom=basemap.zoom,
        tile_count=basemap.tile_count,
        available_tile_count=getattr(
            basemap, "available_tile_count", basemap.tile_count
        ),
        missing_tile_count=getattr(basemap, "missing_tile_count", 0),
        provider=basemap.provider,
        tile_url=basemap.tile_url,
    )
    _write_osm(path, evidence)
    return evidence, False


def _path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _identity_key(identity: dict[str, Any]) -> str:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_spectral(path: Path) -> SpectralLabelEvidence | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            return SpectralLabelEvidence(
                water_score=np.asarray(data["water_score"], dtype=np.float32),
                labels=np.asarray(data["labels"], dtype=np.uint8),
                valid=np.asarray(data["valid"], dtype=bool),
                bands={key: int(value) for key, value in metadata["bands"].items()},
                diagnostics=metadata["diagnostics"],
                transform=Affine(*np.asarray(data["transform"], dtype=float).tolist()),
                crs=str(data["crs"].item()),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def _write_spectral(path: Path, evidence: SpectralLabelEvidence) -> None:
    metadata = {
        "bands": evidence.bands,
        "diagnostics": evidence.diagnostics,
    }
    arrays = {
        "water_score": evidence.water_score,
        "labels": evidence.labels,
        "valid": evidence.valid.astype(np.uint8, copy=False),
        "transform": np.asarray(tuple(evidence.transform)[:6], dtype=np.float64),
        "crs": np.asarray(str(evidence.crs)),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    _write_npz(path, arrays)


def _load_osm(path: Path) -> OsmMaskEvidence | None:
    try:
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if (
                int(metadata["missing_tile_count"]) > 0
                and time.time() - path.stat().st_mtime
                > INCOMPLETE_OSM_CACHE_SECONDS
            ):
                return None
            water = np.asarray(data["water"], dtype=bool)
            valid = np.asarray(data["valid"], dtype=bool)
            if water.shape != valid.shape or water.ndim != 2:
                raise ValueError("cached OSM masks have incompatible dimensions")
            return OsmMaskEvidence(
                water=water,
                valid=valid,
                zoom=int(metadata["zoom"]),
                tile_count=int(metadata["tile_count"]),
                available_tile_count=int(metadata["available_tile_count"]),
                missing_tile_count=int(metadata["missing_tile_count"]),
                provider=str(metadata["provider"]),
                tile_url=str(metadata["tile_url"]),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def _write_osm(path: Path, evidence: OsmMaskEvidence) -> None:
    metadata = {
        "zoom": evidence.zoom,
        "tile_count": evidence.tile_count,
        "available_tile_count": evidence.available_tile_count,
        "missing_tile_count": evidence.missing_tile_count,
        "provider": evidence.provider,
        "tile_url": evidence.tile_url,
    }
    _write_npz(
        path,
        {
            "water": evidence.water.astype(np.uint8, copy=False),
            "valid": evidence.valid.astype(np.uint8, copy=False),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        },
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
