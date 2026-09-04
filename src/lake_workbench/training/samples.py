"""Training-sample creation and annotation snapshot operations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lake_workbench.geo import site_aoi_geometry
from lake_workbench.sentinel.download import upsert_csv_row
from lake_workbench.training.identity import (
    bbox_from_row,
    bbox_iou,
    training_view_signature,
)
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


class TrainingSampleCatalogMixin:
    """Persistence operations for samples and their label snapshots."""

    region: Any

    def create_training_sample(
        self,
        site: Any,
        payload: dict,
        *,
        samples_path: Path | None = None,
        label_dir: Path | None = None,
    ) -> dict:
        readiness = self.training_sample_readiness(site)
        label_source = clean_optional(payload.get("label_source")) or "osm"
        label_threshold = clean_optional(payload.get("label_threshold")) or ""
        view_state = (
            payload.get("view_state")
            if isinstance(payload.get("view_state"), dict)
            else {}
        )
        model_validation = (
            view_state.get("model_validation")
            if isinstance(view_state.get("model_validation"), dict)
            else {}
        )
        model_prediction_excluded = truthy_flag(
            view_state.get("model_prediction_excluded"), default=True
        )
        is_current_view = label_source == "current_view" or bool(view_state)
        label_scope = clean_optional(payload.get("label_scope")) or (
            "current_view" if is_current_view else "target_only"
        )
        mask_policy = clean_optional(payload.get("mask_policy")) or (
            "current_view" if is_current_view else "other_water_ignore"
        )
        context_sources = clean_optional(payload.get("context_sources")) or (
            "" if is_current_view else "osm,hydrolakes"
        )
        ignore_sources = clean_optional(payload.get("ignore_sources")) or (
            "" if is_current_view else "osm,hydrolakes,esa,jrc"
        )
        buffer_ratio = parse_float_or_default(payload.get("buffer_ratio"), 0.8)
        aoi = site_aoi_geometry(site, padding=buffer_ratio)
        if is_current_view:
            label_source = "current_view"
            label_layer = self.current_view_training_label_layer(
                site, view_state, buffer_ratio=buffer_ratio
            )
            label_threshold = (
                clean_optional(label_layer.get("properties", {}).get("jrc_threshold"))
                or label_threshold
            )
            context_sources = (
                clean_optional(label_layer.get("properties", {}).get("visible_sources"))
                or context_sources
            )
        else:
            label_layer = self.training_label_layer(site, label_source, label_threshold)
        has_label_geometry = bool(
            label_layer
            and (
                label_layer.get("geometry")
                or (
                    label_layer.get("type") == "FeatureCollection"
                    and label_layer.get("features")
                )
            )
        )
        if not has_label_geometry:
            raise ValueError(f"label source has no geometry: {label_source}")
        products = readiness["products"]
        product_names = [item["product_name"] for item in products]
        tile_names = [item["tile"] for item in products]
        product_key = ",".join(product_names)
        fingerprint, base_fingerprint, view_extent = training_view_signature(
            site.site_id,
            product_key,
            label_source,
            label_threshold,
            label_scope,
            mask_policy,
            view_state,
        )
        samples_path = samples_path or self.region.training_samples
        existing_rows = read_csv_records(samples_path)
        exact_existing = next(
            (
                row
                for row in existing_rows
                if row.get("training_fingerprint") == fingerprint
            ),
            None,
        )
        similar_samples = []
        for existing in existing_rows:
            if exact_existing and existing.get("sample_id") == exact_existing.get(
                "sample_id"
            ):
                continue
            if (
                existing.get("site_id") != site.site_id
                or existing.get("training_base_fingerprint") != base_fingerprint
            ):
                continue
            overlap = bbox_iou(view_extent, bbox_from_row(existing))
            if overlap >= 0.9:
                similar_samples.append(
                    {
                        "sample_id": existing.get("sample_id", ""),
                        "site_id": existing.get("site_id") or "",
                        "overlap": overlap,
                        "created_at": existing.get("created_at", ""),
                        "notes": existing.get("notes", ""),
                    }
                )
        sample_id = (exact_existing or {}).get(
            "sample_id"
        ) or f"{site.site_id}_{fingerprint[:12]}"
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
            output_dir=label_dir,
        )
        asset_types = [item.get("asset_type", "") for item in products]
        asset_scopes = [item.get("asset_scope", "") for item in products]
        asset_labels = [item.get("asset_label", "") for item in products]
        row = {
            "sample_id": sample_id,
            "site_id": site.site_id,
            "site_name": site.display_name or "",
            "tile": ",".join(tile_names),
            "tiles": ",".join(tile_names),
            "required_tiles": ",".join(readiness.get("required_tiles") or []),
            "ready_tiles": ",".join(tile_names),
            "missing_tiles": ",".join(readiness.get("missing_tiles") or []),
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
            "view_state_json": json.dumps(
                view_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if view_state
            else "",
            "model_prediction_excluded": "true"
            if model_prediction_excluded
            else "false",
            "diagnostic_model_key": clean_optional(model_validation.get("model_key"))
            or "",
            "diagnostic_model_name": clean_optional(model_validation.get("model_name"))
            or "",
            "diagnostic_model_path": clean_optional(model_validation.get("model_path"))
            or "",
            "diagnostic_model_weight": clean_optional(
                model_validation.get("model_weight")
            )
            or "",
            "diagnostic_model_threshold": clean_optional(
                model_validation.get("threshold")
            )
            or "",
            "diagnostic_prediction_area_km2": clean_optional(
                model_validation.get("predicted_area_km2")
            )
            or "",
            "diagnostic_prediction_ratio": clean_optional(
                model_validation.get("predicted_ratio")
            )
            or "",
            "diagnostic_prediction_feature_count": clean_optional(
                model_validation.get("prediction_feature_count")
            )
            or "",
            "diagnostic_model_device": clean_optional(model_validation.get("device"))
            or "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "notes": clean_optional(payload.get("notes")) or "",
        }
        upsert_csv_row(samples_path, row, key="sample_id")
        return {
            **row,
            "action": "updated_existing" if exact_existing else "created",
            "duplicate": bool(exact_existing),
            "similar_samples": similar_samples[:5],
        }

    def current_view_training_label_layer(
        self, site: Any, view_state: dict, buffer_ratio: float = 0.8
    ) -> dict:
        visible = (
            dict(view_state.get("visible_layers"))
            if isinstance(view_state.get("visible_layers"), dict)
            else {}
        )
        model_prediction_visible = truthy_flag(
            visible.pop("model_prediction", False), default=False
        )
        features = []
        sources = []

        def add_layer(key: str, source_name: str, layer: dict | None) -> None:
            if not visible.get(key) or not layer or not layer.get("geometry"):
                return
            props = {
                **(layer.get("properties") or {}),
                "training_layer": key,
                "source": (layer.get("properties") or {}).get("source")
                or layer.get("source")
                or source_name,
            }
            features.append(
                {"type": "Feature", "geometry": layer["geometry"], "properties": props}
            )
            sources.append(key)

        def add_collection(key: str, collection: dict | None) -> None:
            if not visible.get(key) or not collection:
                return
            added = 0
            for feature in collection.get("features") or []:
                if not feature.get("geometry"):
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": feature["geometry"],
                        "properties": {
                            **(feature.get("properties") or {}),
                            "training_layer": key,
                        },
                    }
                )
                added += 1
            if added:
                sources.append(key)

        add_layer("osm", "OSM", self._annotation_layer(site, "osm"))
        add_layer(
            "hydrolakes", "HydroLAKES", self._annotation_layer(site, "hydrolakes")
        )
        add_layer("esa", "ESA", self._annotation_layer(site, "esa"))
        threshold = parse_int_or_default(view_state.get("jrc_threshold"), 75)
        if visible.get("jrc"):
            add_layer(
                "jrc",
                "JRC",
                self._annotation_layer(site, "jrc", {"threshold": threshold}),
            )
        if visible.get("context_osm") or visible.get("context_hydrolakes"):
            context = self.context_water_for_site(
                site, padding=buffer_ratio, min_area_km2=10, limit=500
            )
            add_collection("context_osm", context.get("sources", {}).get("osm"))
            add_collection(
                "context_hydrolakes", context.get("sources", {}).get("hydrolakes")
            )
        local_label = (
            view_state.get("selected_local_label")
            if isinstance(view_state.get("selected_local_label"), dict)
            else {}
        )
        if visible.get("local_label") and clean_optional(local_label.get("id")):
            add_collection(
                "local_label",
                self._annotation_layer(
                    site, "local", {"label_id": clean_optional(local_label.get("id"))}
                ),
            )
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

    def list_training_samples(self, samples_path: Path | None = None) -> dict:
        samples = [
            self._training_sample_summary(row)
            for row in read_csv_records(samples_path or self.region.training_samples)
        ]
        samples.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {
            "total": len(samples),
            "items": samples,
            "quality_counts": count_values(item.get("quality") for item in samples),
            "split_counts": count_values(item.get("split") for item in samples),
        }

    def update_training_sample(
        self, sample_id: str, payload: dict, samples_path: Path | None = None
    ) -> dict:
        samples_path = samples_path or self.region.training_samples
        rows = read_csv_records(samples_path)
        updated = None
        allowed = {
            "quality",
            "split",
            "notes",
            "label_scope",
            "mask_policy",
            "context_sources",
            "ignore_sources",
        }
        for row in rows:
            if row.get("sample_id") != str(sample_id):
                continue
            for key in allowed:
                if key in payload:
                    row[key] = clean_optional(payload.get(key)) or ""
            updated = row
            break
        if updated is None:
            raise KeyError(f"training sample not found: {sample_id}")
        write_csv_records(samples_path, rows)
        return self._training_sample_summary(updated)

    def delete_training_sample(
        self, sample_id: str, samples_path: Path | None = None
    ) -> dict:
        samples_path = samples_path or self.region.training_samples
        rows = read_csv_records(samples_path)
        kept = [row for row in rows if row.get("sample_id") != str(sample_id)]
        if len(kept) == len(rows):
            raise KeyError(f"training sample not found: {sample_id}")
        write_csv_records(samples_path, kept)
        return {"sample_id": str(sample_id), "deleted": True}

    def _training_sample_summary(self, row: dict) -> dict:
        site_id = row.get("site_id", "")
        site = self.get_site(site_id) if site_id else None
        label_path = (
            resolve_data_path(row.get("label_path", ""), self.region)
            if row.get("label_path")
            else None
        )
        missing_tci = [
            path
            for path in split_semicolon(row.get("tci_path"))
            if not resolve_data_path(path, self.region).exists()
        ]
        missing_safe = [
            path
            for path in split_semicolon(row.get("safe_path"))
            if path and not resolve_data_path(path, self.region).exists()
        ]
        return {
            **row,
            "site_id": site_id,
            "site_display_name": site.display_name
            if site
            else row.get("site_name") or site_id,
            "tile_count": len(split_commas(row.get("tiles") or row.get("tile"))),
            "product_count": len(
                split_commas(row.get("products") or row.get("product_name"))
            ),
            "imagery_asset_labels": split_commas(
                row.get("imagery_asset_labels") or row.get("imagery_asset_label")
            ),
            "imagery_asset_types": split_commas(
                row.get("imagery_asset_types") or row.get("imagery_asset_type")
            ),
            "label_exists": bool(label_path and label_path.exists()),
            "missing_tci": missing_tci,
            "missing_safe": missing_safe,
            "status": "missing_files"
            if missing_tci or not (label_path and label_path.exists())
            else "ok",
        }

    def training_sample_readiness(self, site: Any, buffer_ratio: float = 0.8) -> dict:
        aoi = site_aoi_geometry(site, padding=buffer_ratio)
        tiles = self._required_sentinel_tiles_for_site(site)
        products = []
        missing_tiles = []
        for tile in tiles:
            row = self._active_imagery_row(tile, site)
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
                    **self._imagery_asset_meta(row, site_id=site.site_id),
                    "valid_ratio": row.get("valid_ratio"),
                    "safe_path": display_path(row["safe_path"])
                    if row.get("safe_path")
                    else "",
                    "tci_path": display_path(row["tci_path"])
                    if row.get("tci_path")
                    else "",
                }
            )
        return {
            "site_id": site.site_id,
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
            "message": "ready"
            if tiles and not missing_tiles
            else "active imagery selection is incomplete",
        }

    def training_label_layer(
        self, site: Any, label_source: str, label_threshold: str = ""
    ) -> dict | None:
        source = str(label_source).lower()
        options = {"threshold": label_threshold or 75} if source == "jrc" else None
        try:
            return self._annotation_layer(site, source, options)
        except ValueError as exc:
            raise ValueError(f"unknown label_source: {label_source}") from exc

    def _annotation_layer(
        self, site: Any, source: str, options: dict | None = None
    ) -> dict | None:
        return self.annotation_for_site(site, source, options).get("annotation")
