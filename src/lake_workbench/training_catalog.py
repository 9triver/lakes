"""Training sample and patch persistence mixed into the lake catalog."""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from lake_workbench.geo import lake_aoi_geometry
from lake_workbench.sentinel_download import upsert_csv_row
from lake_workbench.training import bbox_from_row, bbox_iou, training_view_signature
from lake_workbench.utils import (
    clean_optional,
    count_values,
    display_path,
    parse_float_or_default,
    parse_int_or_default,
    read_csv_records,
    resolve_data_path,
    split_commas,
    split_semicolon,
    truthy_flag,
    write_csv_records,
    write_training_label,
)


class TrainingCatalogMixin:
    """Persistence operations that require access to a lake catalog instance."""

    region: Any

    def create_training_sample(self, lake: Any, payload: dict) -> dict:
        readiness = self.training_sample_readiness(lake)
        label_source = clean_optional(payload.get("label_source")) or "osm"
        label_threshold = clean_optional(payload.get("label_threshold")) or ""
        view_state = payload.get("view_state") if isinstance(payload.get("view_state"), dict) else {}
        model_validation = view_state.get("model_validation") if isinstance(view_state.get("model_validation"), dict) else {}
        model_prediction_excluded = truthy_flag(view_state.get("model_prediction_excluded"), default=True)
        is_current_view = label_source == "current_view" or bool(view_state)
        label_scope = clean_optional(payload.get("label_scope")) or ("current_view" if is_current_view else "target_only")
        mask_policy = clean_optional(payload.get("mask_policy")) or ("current_view" if is_current_view else "other_water_ignore")
        context_sources = clean_optional(payload.get("context_sources")) or ("" if is_current_view else "osm,hydrolakes")
        ignore_sources = clean_optional(payload.get("ignore_sources")) or ("" if is_current_view else "osm,hydrolakes,esa,jrc")
        buffer_ratio = parse_float_or_default(payload.get("buffer_ratio"), 0.8)
        aoi = lake_aoi_geometry(lake, padding=buffer_ratio)
        if is_current_view:
            label_source = "current_view"
            label_layer = self.current_view_training_label_layer(lake, view_state, buffer_ratio=buffer_ratio)
            label_threshold = clean_optional(label_layer.get("properties", {}).get("jrc_threshold")) or label_threshold
            context_sources = clean_optional(label_layer.get("properties", {}).get("visible_sources")) or context_sources
        else:
            label_layer = self.training_label_layer(lake, label_source, label_threshold)
        has_label_geometry = bool(
            label_layer
            and (label_layer.get("geometry") or (label_layer.get("type") == "FeatureCollection" and label_layer.get("features")))
        )
        if not has_label_geometry:
            raise ValueError(f"label source has no geometry: {label_source}")

        products = readiness["products"]
        product_names = [item["product_name"] for item in products]
        tile_names = [item["tile"] for item in products]
        product_key = ",".join(product_names)
        fingerprint, base_fingerprint, view_extent = training_view_signature(
            lake.object_id,
            product_key,
            label_source,
            label_threshold,
            label_scope,
            mask_policy,
            view_state,
        )
        existing_rows = read_csv_records(self.region.training_samples)
        exact_existing = next((row for row in existing_rows if row.get("training_fingerprint") == fingerprint), None)
        similar_samples = []
        for existing in existing_rows:
            if exact_existing and existing.get("sample_id") == exact_existing.get("sample_id"):
                continue
            if existing.get("lake_id") != lake.object_id or existing.get("training_base_fingerprint") != base_fingerprint:
                continue
            overlap = bbox_iou(view_extent, bbox_from_row(existing))
            if overlap >= 0.9:
                similar_samples.append(
                    {
                        "sample_id": existing.get("sample_id", ""),
                        "lake_id": existing.get("lake_id", ""),
                        "overlap": overlap,
                        "created_at": existing.get("created_at", ""),
                        "notes": existing.get("notes", ""),
                    }
                )
        sample_id = (exact_existing or {}).get("sample_id") or f"{lake.object_id}_{fingerprint[:12]}"
        label_path = write_training_label(
            self.region,
            sample_id,
            label_layer,
            {
                "label_scope": label_scope,
                "mask_policy": mask_policy,
                "context_sources": context_sources,
                "ignore_sources": ignore_sources,
                "model_prediction_excluded": model_prediction_excluded,
            },
        )
        asset_types = [item.get("asset_type", "") for item in products]
        asset_scopes = [item.get("asset_scope", "") for item in products]
        asset_labels = [item.get("asset_label", "") for item in products]
        required_tiles = readiness.get("required_tiles") or []
        missing_tiles = readiness.get("missing_tiles") or []
        row = {
            "sample_id": sample_id,
            "lake_id": lake.object_id,
            "lake_name": lake.name or "",
            "tile": ",".join(tile_names),
            "tiles": ",".join(tile_names),
            "required_tiles": ",".join(required_tiles),
            "ready_tiles": ",".join(tile_names),
            "missing_tiles": ",".join(missing_tiles),
            "imagery_ready": "true" if readiness.get("ready") else "false",
            "product_id": ",".join(item.get("product_id", "") for item in products),
            "product_name": product_key,
            "products": product_key,
            "product_source": ",".join(item.get("source", "") for item in products),
            "imagery_asset_type": ",".join(asset_types),
            "imagery_asset_types": ",".join(asset_types),
            "imagery_asset_scope": ",".join(asset_scopes),
            "imagery_asset_scopes": ",".join(asset_scopes),
            "imagery_asset_label": ",".join(asset_labels),
            "imagery_asset_labels": ",".join(asset_labels),
            "product_date": ",".join(item.get("date", "") for item in products),
            "safe_path": ";".join(item.get("safe_path", "") for item in products),
            "tci_path": ";".join(item.get("tci_path", "") for item in products),
            "label_source": label_source,
            "label_threshold": label_threshold,
            "label_scope": label_scope,
            "mask_policy": mask_policy,
            "context_sources": context_sources,
            "ignore_sources": ignore_sources,
            "label_path": display_path(label_path),
            "aoi_west": aoi.bounds[0],
            "aoi_south": aoi.bounds[1],
            "aoi_east": aoi.bounds[2],
            "aoi_north": aoi.bounds[3],
            "buffer_ratio": buffer_ratio,
            "quality": clean_optional(payload.get("quality")) or "",
            "split": clean_optional(payload.get("split")) or "",
            "training_fingerprint": fingerprint,
            "training_base_fingerprint": base_fingerprint,
            "view_west": view_extent[0] if view_extent else "",
            "view_south": view_extent[1] if view_extent else "",
            "view_east": view_extent[2] if view_extent else "",
            "view_north": view_extent[3] if view_extent else "",
            "view_state_json": json.dumps(view_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if view_state else "",
            "model_prediction_excluded": "true" if model_prediction_excluded else "false",
            "diagnostic_model_key": clean_optional(model_validation.get("model_key")) or "",
            "diagnostic_model_name": clean_optional(model_validation.get("model_name")) or "",
            "diagnostic_model_path": clean_optional(model_validation.get("model_path")) or "",
            "diagnostic_model_weight": clean_optional(model_validation.get("model_weight")) or "",
            "diagnostic_model_threshold": clean_optional(model_validation.get("threshold")) or "",
            "diagnostic_prediction_area_km2": clean_optional(model_validation.get("predicted_area_km2")) or "",
            "diagnostic_prediction_ratio": clean_optional(model_validation.get("predicted_ratio")) or "",
            "diagnostic_prediction_feature_count": clean_optional(model_validation.get("prediction_feature_count")) or "",
            "diagnostic_model_device": clean_optional(model_validation.get("device")) or "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "notes": clean_optional(payload.get("notes")) or "",
        }
        upsert_csv_row(self.region.training_samples, row, key="sample_id")
        return {
            **row,
            "action": "updated_existing" if exact_existing else "created",
            "duplicate": bool(exact_existing),
            "similar_samples": similar_samples[:5],
        }

    def current_view_training_label_layer(self, lake: Any, view_state: dict, buffer_ratio: float = 0.8) -> dict:
        visible = dict(view_state.get("visible_layers")) if isinstance(view_state.get("visible_layers"), dict) else {}
        model_prediction_visible = truthy_flag(visible.pop("model_prediction", False), default=False)
        features = []
        sources = []

        def add_layer(key: str, source_name: str, layer: dict | None) -> None:
            if not visible.get(key) or not layer or not layer.get("geometry"):
                return
            props = {**(layer.get("properties") or {})}
            props.update({"training_layer": key, "source": props.get("source") or layer.get("source") or source_name})
            features.append({"type": "Feature", "geometry": layer["geometry"], "properties": props})
            sources.append(key)

        def add_collection(key: str, collection: dict | None) -> None:
            if not visible.get(key) or not collection:
                return
            added = 0
            for feature in collection.get("features") or []:
                if not feature.get("geometry"):
                    continue
                props = {**(feature.get("properties") or {}), "training_layer": key}
                features.append({"type": "Feature", "geometry": feature["geometry"], "properties": props})
                added += 1
            if added:
                sources.append(key)

        add_layer("osm", "OSM", self._osm_layer(lake))
        add_layer("hydrolakes", "HydroLAKES", self._match_hydrolakes(lake))
        add_layer("esa", "ESA", self._esa_smoothed_layer(lake))
        threshold = parse_int_or_default(view_state.get("jrc_threshold"), 75)
        if visible.get("jrc"):
            add_layer("jrc", "JRC", self._jrc_occurrence_layer(lake, threshold=threshold))
        if visible.get("context_osm") or visible.get("context_hydrolakes"):
            context = self.context_water_for_lake(lake, padding=buffer_ratio, min_area_km2=10, limit=500)
            add_collection("context_osm", context.get("sources", {}).get("osm"))
            add_collection("context_hydrolakes", context.get("sources", {}).get("hydrolakes"))
        local_label = view_state.get("selected_local_label") if isinstance(view_state.get("selected_local_label"), dict) else {}
        local_label_id = clean_optional(local_label.get("id"))
        if visible.get("local_label") and local_label_id:
            add_collection("local_label", self.local_label_geojson(lake, local_label_id).get("geojson"))
        if not features:
            raise ValueError("current view has no visible label geometry")
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "source": "current_view",
                "visible_sources": ",".join(sources),
                "jrc_threshold": threshold,
                "model_prediction_visible": model_prediction_visible,
                "model_prediction_excluded": True,
                "view_state": view_state,
            },
        }

    def list_training_samples(self) -> dict:
        rows = read_csv_records(self.region.training_samples)
        samples = [self._training_sample_summary(row) for row in rows]
        samples.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {
            "total": len(samples),
            "items": samples,
            "quality_counts": count_values(item.get("quality") for item in samples),
            "split_counts": count_values(item.get("split") for item in samples),
        }

    def update_training_sample(self, sample_id: str, payload: dict) -> dict:
        rows = read_csv_records(self.region.training_samples)
        sample_id = str(sample_id)
        updated = None
        allowed = {"quality", "split", "notes", "label_scope", "mask_policy", "context_sources", "ignore_sources"}
        for row in rows:
            if row.get("sample_id") != sample_id:
                continue
            for key in allowed:
                if key in payload:
                    row[key] = clean_optional(payload.get(key)) or ""
            updated = row
            break
        if updated is None:
            raise KeyError(f"training sample not found: {sample_id}")
        write_csv_records(self.region.training_samples, rows)
        return self._training_sample_summary(updated)

    def delete_training_sample(self, sample_id: str) -> dict:
        rows = read_csv_records(self.region.training_samples)
        sample_id = str(sample_id)
        kept = [row for row in rows if row.get("sample_id") != sample_id]
        if len(kept) == len(rows):
            raise KeyError(f"training sample not found: {sample_id}")
        write_csv_records(self.region.training_samples, kept)
        return {"sample_id": sample_id, "deleted": True}

    def list_training_patches(self, include: str = "") -> dict:
        rows = []
        for manifest_path in self.training_patch_manifest_paths():
            for row in read_csv_records(manifest_path):
                item = self._training_patch_summary(row, manifest_path)
                if include == "included" and not item["included"]:
                    continue
                if include == "excluded" and item["included"]:
                    continue
                rows.append(item)
        rows.sort(key=lambda item: (item.get("sample_id", ""), int(item.get("row_off") or 0), int(item.get("col_off") or 0)))
        included_count = sum(1 for item in rows if item["included"])
        return {
            "total": len(rows),
            "included_count": included_count,
            "excluded_count": len(rows) - included_count,
            "items": rows,
        }

    def update_training_patch(self, patch_id: str, payload: dict) -> dict:
        patch_id = str(patch_id)
        for manifest_path in self.training_patch_manifest_paths():
            rows = read_csv_records(manifest_path)
            updated = None
            for row in rows:
                if row.get("patch_id") != patch_id:
                    continue
                if "include" in payload or "included" in payload:
                    value = payload.get("include") if "include" in payload else payload.get("included")
                    row["include"] = "true" if truthy_flag(value, default=False) else "false"
                if "patch_notes" in payload:
                    row["patch_notes"] = clean_optional(payload.get("patch_notes")) or ""
                updated = row
                break
            if updated is None:
                continue
            write_csv_records(manifest_path, rows)
            return self._training_patch_summary(updated, manifest_path)
        raise KeyError(f"training patch not found: {patch_id}")

    def training_patch_preview(self, patch_id: str) -> tuple[bytes, str]:
        patch = self.training_patch_by_id(patch_id)
        preview_path = resolve_data_path(patch.get("preview_path", ""), self.region)
        if not preview_path.exists():
            raise FileNotFoundError(f"patch preview not found: {patch_id}")
        return preview_path.read_bytes(), mimetypes.guess_type(preview_path.name)[0] or "image/png"

    def training_patch_by_id(self, patch_id: str) -> dict:
        patch_id = str(patch_id)
        for manifest_path in self.training_patch_manifest_paths():
            for row in read_csv_records(manifest_path):
                if row.get("patch_id") == patch_id:
                    return self._training_patch_summary(row, manifest_path)
        raise KeyError(f"training patch not found: {patch_id}")

    def training_patch_manifest_paths(self) -> list[Path]:
        root = self.region.processed_dir / "training_patches"
        if not root.exists():
            return []
        return sorted(root.glob("*/manifest.csv"), key=lambda path: path.stat().st_mtime, reverse=True)

    def _training_patch_summary(self, row: dict, manifest_path: Path) -> dict:
        patch_id = row.get("patch_id", "")
        preview_path = resolve_data_path(row.get("preview_path", ""), self.region) if row.get("preview_path") else None
        npz_path = resolve_data_path(row.get("npz_path", ""), self.region) if row.get("npz_path") else None
        include_value = clean_optional(row.get("include") or row.get("included"))
        included = True if include_value is None else truthy_flag(include_value, default=True)
        lake_id = row.get("lake_id", "")
        lake = self.get_lake(lake_id) if lake_id else None
        return {
            **row,
            "included": included,
            "include": "true" if included else "false",
            "lake_display_name": (lake.properties.get("display_name") if lake else None) or row.get("lake_name") or lake_id,
            "manifest_path": display_path(manifest_path),
            "preview_exists": bool(preview_path and preview_path.exists()),
            "npz_exists": bool(npz_path and npz_path.exists()),
            "preview_url": f"/api/regions/{self.region.key}/training-patches/{patch_id}/preview.png"
            if patch_id and preview_path and preview_path.exists()
            else "",
        }

    def _training_sample_summary(self, row: dict) -> dict:
        lake_id = row.get("lake_id", "")
        lake = self.get_lake(lake_id) if lake_id else None
        tci_paths = split_semicolon(row.get("tci_path"))
        safe_paths = split_semicolon(row.get("safe_path"))
        label_path = resolve_data_path(row.get("label_path", ""), self.region) if row.get("label_path") else None
        missing_tci = [path for path in tci_paths if not resolve_data_path(path, self.region).exists()]
        missing_safe = [path for path in safe_paths if path and not resolve_data_path(path, self.region).exists()]
        label_exists = bool(label_path and label_path.exists())
        return {
            **row,
            "lake_display_name": (lake.properties.get("display_name") if lake else None) or row.get("lake_name") or lake_id,
            "tile_count": len(split_commas(row.get("tiles") or row.get("tile"))),
            "product_count": len(split_commas(row.get("products") or row.get("product_name"))),
            "imagery_asset_labels": split_commas(row.get("imagery_asset_labels") or row.get("imagery_asset_label")),
            "imagery_asset_types": split_commas(row.get("imagery_asset_types") or row.get("imagery_asset_type")),
            "label_exists": label_exists,
            "missing_tci": missing_tci,
            "missing_safe": missing_safe,
            "status": "missing_files" if missing_tci or not label_exists else "ok",
        }

    def training_sample_readiness(self, lake: Any, buffer_ratio: float = 0.8) -> dict:
        aoi = lake_aoi_geometry(lake, padding=buffer_ratio)
        tiles = self._required_sentinel_tiles_for_lake(lake)
        products = []
        missing_tiles = []
        for tile in tiles:
            row = self._active_imagery_row(tile, lake)
            if row is None:
                missing_tiles.append(tile)
                continue
            products.append(
                {
                    "tile": tile,
                    "product_name": row.get("product", ""),
                    "product_id": row.get("product_id", ""),
                    "date": row.get("date", ""),
                    "source": row.get("source", ""),
                    **self._imagery_asset_meta(row, lake_id=lake.object_id),
                    "valid_ratio": row.get("valid_ratio"),
                    "safe_path": display_path(row["safe_path"]) if row.get("safe_path") else "",
                    "tci_path": display_path(row["tci_path"]) if row.get("tci_path") else "",
                }
            )
        return {
            "lake_id": lake.object_id,
            "ready": bool(tiles) and not missing_tiles,
            "required_tiles": tiles,
            "missing_tiles": missing_tiles,
            "ready_tiles": [item["tile"] for item in products],
            "required_count": len(tiles),
            "ready_count": len(products),
            "missing_count": len(missing_tiles),
            "products": products,
            "buffer_ratio": buffer_ratio,
            "aoi_bounds": list(aoi.bounds),
            "message": "ready" if tiles and not missing_tiles else "active imagery selection is incomplete",
        }

    def training_label_layer(self, lake: Any, label_source: str, label_threshold: str = "") -> dict | None:
        source = str(label_source).lower()
        if source == "osm":
            return self._osm_layer(lake)
        if source == "hydrolakes":
            return self._match_hydrolakes(lake)
        if source == "esa":
            return self._esa_smoothed_layer(lake)
        if source == "jrc":
            return self._jrc_occurrence_layer(lake, threshold=int(label_threshold or 75))
        raise ValueError(f"unknown label_source: {label_source}")
