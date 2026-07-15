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
from lake_workbench.models.metadata import (
    global_model_key,
    global_model_path_from_key,
    iter_global_model_paths,
    model_sort_key,
    model_training_metadata,
)
from lake_workbench.models.runtime import load_model_checkpoint
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.utils import clean_optional, display_path, safe_filename


GLOBAL_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "all"
MODEL_INFERENCE_SEMAPHORE = threading.BoundedSemaphore(1)


class ModelInferenceBusy(RuntimeError):
    """Raised when a model inference request is already running."""


class ModelValidationMixin:
    """Model validation operations that require a site catalog instance."""

    region: Any
    sites: list[Any]

    def model_validation_models(self) -> dict:
        items = []
        default_key = ""
        default_path = self._default_model_path()
        for path, legacy in self._iter_model_paths():
            key = self._model_key(path)
            label = f"{path.parent.name}/{path.name}" if not legacy else f"旧目录 / {path.parent.name}/{path.name}"
            try:
                model = load_model_checkpoint(path)
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "epoch": model.epoch,
                    "in_channels": model.in_channels,
                    "base_channels": model.base_channels,
                    "model_type": model.model_type,
                    "model_options": model.model_options,
                    "architecture_label": model.architecture_label,
                    "scope": self.region.key,
                    "legacy": legacy,
                    "default": path.resolve() == default_path.resolve(),
                    **model_training_metadata(path, self.region.key),
                }
            except Exception as exc:  # noqa: BLE001 - broken checkpoints remain visible in the UI.
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "scope": self.region.key,
                    "legacy": legacy,
                    "error": f"{type(exc).__name__}: {exc}",
                    "default": path.resolve() == default_path.resolve(),
                }
            if item["default"]:
                default_key = item["key"]
            items.append(item)

        for path in iter_global_model_paths():
            key = global_model_key(path)
            label = f"全部区域 / {path.parent.name}/{path.name}"
            try:
                model = load_model_checkpoint(path)
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "epoch": model.epoch,
                    "in_channels": model.in_channels,
                    "base_channels": model.base_channels,
                    "model_type": model.model_type,
                    "model_options": model.model_options,
                    "architecture_label": model.architecture_label,
                    "scope": "all",
                    "legacy": False,
                    "default": False,
                    **model_training_metadata(path, "all"),
                }
            except Exception as exc:  # noqa: BLE001 - broken checkpoints remain visible in the UI.
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "scope": "all",
                    "legacy": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "default": False,
                }
            items.append(item)
        items.sort(key=model_sort_key)
        if items:
            default_key = next((item["key"] for item in items if not item.get("error")), items[0]["key"])
        return {"region": self.region.key, "default": default_key, "items": items}

    def model_validation_random(self, threshold: float = 0.5, model_key: str = "") -> dict:
        model = self._load_validation_model(model_key)
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
        model_key: str = "",
    ) -> dict:
        model = model or self._load_validation_model(model_key)
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
                "key": self._model_key(model.path),
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

    def _load_validation_model(self, model_key: str = "") -> Any:
        model_path = self._model_path_from_key(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {self.region.key}: {display_path(model_path)}")
        return load_model_checkpoint(model_path)

    def _default_model_path(self) -> Path:
        preferred = self.region.model_dir / "unet_current_v1" / "best.pt"
        if preferred.exists():
            return preferred
        return self.region.legacy_model_dir / "unet_current_v1" / "best.pt"

    def _iter_model_paths(self) -> list[tuple[Path, bool]]:
        paths: list[tuple[Path, bool]] = []
        if self.region.model_dir.exists():
            paths.extend((path, False) for path in sorted(self.region.model_dir.glob("*/*.pt")))
        if self.region.legacy_model_dir.exists():
            paths.extend((path, True) for path in sorted(self.region.legacy_model_dir.glob("*/*.pt")))
        return paths

    def _model_path_from_key(self, model_key: str = "") -> Path:
        key = clean_optional(model_key) or ""
        if not key:
            default_path = self._default_model_path()
            if default_path.exists():
                return default_path
            candidates = [path for path, _legacy in self._iter_model_paths()]
            return candidates[0] if candidates else default_path
        path = Path(key)
        if path.is_absolute():
            raise ValueError("absolute model paths are not allowed")
        parts = path.parts
        if len(parts) == 3 and parts[0] == "all":
            return global_model_path_from_key(key)
        if len(parts) == 3 and parts[0] == "legacy":
            if parts[1] in {"", ".", ".."} or parts[2] in {"", ".", ".."}:
                raise ValueError(f"invalid model key: {key}")
            return self.region.legacy_model_dir / parts[1] / parts[2]
        if len(parts) != 2 or parts[0] in {"", ".", ".."} or parts[1] in {"", ".", ".."}:
            raise ValueError(f"invalid model key: {key}")
        return self.region.model_dir / path

    def _model_key(self, path: Path) -> str:
        path = path.resolve()
        try:
            return f"all/{path.relative_to(GLOBAL_MODEL_DIR.resolve())}"
        except ValueError:
            pass
        try:
            return str(path.relative_to(self.region.model_dir.resolve()))
        except ValueError:
            pass
        try:
            return f"legacy/{path.relative_to(self.region.legacy_model_dir.resolve())}"
        except ValueError:
            return path.name

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
