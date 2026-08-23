"""Model discovery, inference, and validation behavior for site catalogs."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Any

import rasterio

from lake_workbench.geo import padded_bounds
from lake_workbench.imagery import mosaic_source_meta, predict_water_geojson
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.utils import display_path, safe_filename


MODEL_INFERENCE_SEMAPHORE = threading.BoundedSemaphore(1)


class ModelInferenceBusy(RuntimeError):
    """Raised when a model inference request is already running."""


class ModelValidationMixin:
    """Model validation operations that require a site catalog instance."""

    region: Any
    sites: list[Any]

    def model_validation_random(self, threshold: float = 0.5, model: Any = None) -> dict:
        if model is None:
            raise ValueError("A loaded workspace model is required")
        candidates = list(self.sites)
        random.shuffle(candidates)
        skipped = []
        for site in candidates:
            rows = self._model_validation_rows(site, model.in_channels)
            if not rows:
                continue
            try:
                prediction = self.model_prediction_for_site(site, threshold=threshold, rows=rows, model=model)
            except ModelInferenceBusy:
                raise
            except Exception as exc:  # noqa: BLE001 - keep looking for a usable validation target.
                skipped.append(f"{site.site_id}: {type(exc).__name__}: {exc}")
                continue
            return {
                "region": self.region.key,
                "site_id": site.site_id,
                "site": self._summary(site),
                "model": prediction["model"],
                "prediction": prediction["prediction"],
                "stats": prediction["stats"],
                "imagery": prediction["imagery"],
                "skipped_count": len(skipped),
            }
        raise FileNotFoundError(
            f"No observation site with active imagery matching model bands ({model.in_channels}) "
            f"for {self.region.key}: {display_path(model.path)}"
        )

    def model_prediction_for_site(
        self,
        site: Any,
        threshold: float = 0.5,
        rows: list[dict] | None = None,
        model: Any = None,
    ) -> dict:
        if model is None:
            raise ValueError("A loaded workspace model is required")
        rows = rows or self._model_validation_rows(site, model.in_channels)
        if not rows:
            raise FileNotFoundError(f"No active imagery matching model bands for site {site.site_id}")
        rows = sorted(rows, key=lambda row: float(row.get("valid_ratio", 0) or 0), reverse=True)
        prediction_bounds = padded_bounds(site.bbox, 0.8)
        cache_path = self._model_prediction_cache_path(site, model.path, rows, threshold, prediction_bounds)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return payload

        row = rows[0]
        if not MODEL_INFERENCE_SEMAPHORE.acquire(blocking=False):
            raise ModelInferenceBusy("模型推理正在运行，请稍后再试")
        try:
            prediction, stats = predict_water_geojson(
                row["tci_path"], model, threshold=threshold, bounds=prediction_bounds
            )
        finally:
            MODEL_INFERENCE_SEMAPHORE.release()
        payload = {
            "region": self.region.key,
            "site_id": site.site_id,
            "cached": False,
            "model": {
                "key": display_path(model.path),
                "name": model.path.parent.name,
                "path": display_path(model.path),
                "device": str(model.device),
                "epoch": model.epoch,
                "in_channels": model.in_channels,
                "base_channels": model.base_channels,
                "model_type": model.model_type,
                "model_options": model.model_options,
                "architecture_label": model.architecture_label,
                "threshold": threshold,
            },
            "imagery": {
                **mosaic_source_meta([row]),
                "source": row.get("source") or "",
                "asset": self._imagery_asset_meta(row, site_id=site.site_id),
            },
            "stats": stats,
            "prediction": prediction,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def _model_validation_cache_dir(self, model_path: Path) -> Path:
        try:
            relative = model_path.resolve().relative_to((PROJECT_ROOT / "data" / "models").resolve())
            return PROJECT_ROOT / "data" / "model_predictions" / relative.parent
        except ValueError:
            return self.region.processed_dir / "model_predictions" / "legacy" / model_path.parent.name

    def _model_prediction_cache_path(
        self,
        site: Any,
        model_path: Path,
        rows: list[dict],
        threshold: float,
        prediction_bounds: tuple[float, float, float, float],
    ) -> Path:
        payload = {
            "site_id": site.site_id,
            "threshold": round(float(threshold), 4),
            "bounds": [round(value, 8) for value in prediction_bounds],
            "model": display_path(model_path),
            "model_mtime": model_path.stat().st_mtime,
            "rows": [
                {
                    "tile": row.get("tile"),
                    "product": row.get("product"),
                    "path": display_path(row["tci_path"]),
                    "mtime": row["tci_path"].stat().st_mtime,
                }
                for row in rows
            ],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        return self._model_validation_cache_dir(model_path) / f"{safe_filename(site.site_id)}_{digest}.geojson"

    def _model_validation_rows(self, site: Any, in_channels: int) -> list[dict]:
        rows = []
        for tile in self._model_validation_tiles_for_site(site):
            row = self._active_imagery_row(tile, site)
            if row is None or row.get("tci_path") is None or not row["tci_path"].exists():
                continue
            try:
                with rasterio.open(row["tci_path"]) as src:
                    if src.count < in_channels:
                        continue
            except Exception:
                continue
            rows.append(row)
        return rows

    def _model_validation_tiles_for_site(self, site: Any) -> list[str]:
        tiles = [item["tile"] for item in self.sentinel_tiles_for_site(site)["tiles"]]
        active_site_tiles = [
            tile
            for tile, rows in self.user_tci_rows.items()
            if any(row.get("site_id") == site.site_id for row in rows)
        ]
        seen = set()
        result = []
        for tile in active_site_tiles + tiles:
            tile = str(tile).upper().removeprefix("T")
            if tile and tile not in seen:
                seen.add(tile)
                result.append(tile)
        return result
