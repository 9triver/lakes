"""Cross-region views used by the HTTP API."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from lake_workbench.models.metadata import (
    global_model_key,
    global_model_path_from_key,
    iter_global_model_paths,
    model_sort_key,
    model_training_metadata,
)
from lake_workbench.models.runtime import load_model_checkpoint
from lake_workbench.paths import PROJECT_ROOT


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def split_global_model_key(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    parts = text.split("/", 1)
    if len(parts) != 2:
        return "", text
    return parts[0], parts[1]


class RegionService:
    def __init__(
        self,
        catalogs: dict[str, Any],
        default_region_key: str,
        busy_error: type[Exception],
    ) -> None:
        self.catalogs = catalogs
        self.default_region_key = default_region_key
        self.busy_error = busy_error

    def regions_payload(self) -> dict:
        items = []
        for key, catalog in self.catalogs.items():
            region = catalog.region
            items.append(
                {
                    "key": key,
                    "name": region.name,
                    "default": key == self.default_region_key,
                    "bounds": list(region.bounds) if region.bounds else None,
                    "ready": catalog.load_error is None,
                    "load_error": catalog.load_error,
                    "site_count": len(catalog.sites),
                    **catalog.imagery_inventory_summary(),
                    "has_metadata": region.site_metadata.exists(),
                    "has_osm_water": region.osm_water.exists(),
                    "metadata_path": display_path(region.site_metadata),
                    "osm_water_path": display_path(region.osm_water),
                }
            )
        return {"default": self.default_region_key, "items": items}

    def all_sites_payload(self, query: str, limit: int, offset: int, filters: dict, profile_store=None, profile_id: str | None = None) -> dict:
        merged = []
        total = 0
        for key, catalog in self.catalogs.items():
            payload = catalog.list_sites(query=query, limit=10**9, offset=0, filters=filters)
            total += payload["total"]
            counts = profile_store.patch_counts(profile_id, key) if profile_store else {}
            merged.extend({
                **item,
                "region": key,
                "region_name": catalog.region.name,
                **({"included_logical_patch_count": counts.get(item.get("site_id", ""), 0)} if profile_store else {}),
            } for item in payload["items"])
        merged.sort(key=lambda item: (-(item.get("coverage_area_km2") or 0), item.get("region", ""), item.get("site_id", "")))
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": merged[offset : offset + limit],
            "all_regions": True,
        }

    def all_training_samples_payload(self) -> dict:
        items = []
        total = 0
        for key, catalog in self.catalogs.items():
            payload = catalog.list_training_samples()
            total += payload.get("total", 0)
            items.extend(
                {**item, "region": key, "region_name": catalog.region.name}
                for item in payload.get("items", [])
            )
        items.sort(key=lambda item: (item.get("region", ""), item.get("created_at", ""), item.get("sample_id", "")), reverse=True)
        return {"total": total, "items": items, "all_regions": True}

    def all_training_patches_payload(self, include: str = "") -> dict:
        items = []
        included_count = 0
        excluded_count = 0
        for key, catalog in self.catalogs.items():
            payload = catalog.list_training_patches(include=include)
            included_count += payload.get("included_count", 0)
            excluded_count += payload.get("excluded_count", 0)
            items.extend(
                {**item, "region": key, "region_name": catalog.region.name}
                for item in payload.get("items", [])
            )
        items.sort(
            key=lambda item: (
                item.get("region", ""),
                item.get("sample_id", ""),
                int(item.get("row_off") or 0),
                int(item.get("col_off") or 0),
            )
        )
        return {
            "total": len(items),
            "included_count": included_count,
            "excluded_count": excluded_count,
            "items": items,
            "all_regions": True,
        }

    def all_logical_patches_payload(self, include: str = "", site_id: str = "") -> dict:
        items = []
        for key, catalog in self.catalogs.items():
            payload = catalog.list_logical_patches(include=include, site_id=site_id)
            items.extend({**item, "region": key, "region_name": catalog.region.name} for item in payload["items"])
        items.sort(key=lambda item: (item.get("region", ""), item.get("sample_id", ""), int(item.get("image_index") or 0), int(item.get("row_off") or 0), int(item.get("col_off") or 0)))
        included_count = sum(1 for item in items if item["included"])
        return {"total": len(items), "included_count": included_count, "excluded_count": len(items) - included_count, "items": items, "all_regions": True}

    def all_profile_logical_patches_payload(
        self,
        profile_store,
        profile_id: str,
        include: str = "",
        site_id: str = "",
    ) -> dict:
        items = []
        for key, catalog in self.catalogs.items():
            manifest = profile_store.ensure_profile_logical_patch_manifest(profile_id, key)
            payload = catalog.list_logical_patches(include=include, site_id=site_id, manifest_path=manifest)
            for item in payload["items"]:
                included = bool(item.get("included"))
                items.append({
                    **item,
                    "included": included,
                    "include": "true" if included else "false",
                    "region": key,
                    "region_name": catalog.region.name,
                    "profile_id": profile_id,
                    "preview_url": f"/api/profiles/{profile_id}/regions/{key}/logical-patches/{item.get('logical_patch_id', '')}/preview.png",
                })
        items.sort(key=lambda item: (item.get("region", ""), item.get("sample_id", ""), int(item.get("image_index") or 0), int(item.get("row_off") or 0), int(item.get("col_off") or 0)))
        included_count = sum(1 for item in items if item["included"])
        return {"profile_id": profile_id, "total": len(items), "included_count": included_count, "excluded_count": len(items) - included_count, "items": items, "all_regions": True}

    def all_training_dataset_statuses(self) -> dict:
        configs = {}
        for key, catalog in self.catalogs.items():
            for item in catalog.training_dataset_statuses()["items"]:
                aggregate = configs.setdefault(item["config_id"], {"config": item["config"], "config_id": item["config_id"], "regions": [], "patches": 0})
                aggregate["regions"].append({"region": key, "status": item["status"], "patches": item["patches"]})
                aggregate["patches"] += item["patches"]
        items = []
        for aggregate in configs.values():
            statuses = {item["status"] for item in aggregate["regions"] if item["status"] != "missing_logical"}
            status = "ready" if statuses and statuses == {"ready"} else "stale" if "stale" in statuses else "missing"
            items.append({**aggregate, "status": status, "ready": status == "ready"})
        return {"region": "all", "items": sorted(items, key=lambda item: item["config_id"])}

    def all_profile_training_dataset_statuses(self, profile_store, profile_id: str) -> dict:
        from lake_workbench.training.logical_patches import dataset_configs, profile_training_dataset_status

        items = []
        for config_id, config in dataset_configs().items():
            regions = [
                profile_training_dataset_status(catalog.region, config_id, profile_store, profile_id)
                for catalog in self.catalogs.values()
            ]
            relevant = [item for item in regions if item["status"] != "missing_selection"]
            statuses = {item["status"] for item in relevant}
            if not relevant:
                status = "missing_selection"
            elif statuses == {"ready"}:
                status = "ready"
            elif "needs_resolution" in statuses:
                status = "needs_resolution"
            elif "stale" in statuses:
                status = "stale"
            else:
                status = "missing"
            items.append({
                "profile_id": profile_id,
                "config_id": config_id,
                "config": config,
                "regions": regions,
                "patches": sum(item["patches"] for item in relevant),
                "status": status,
                "ready": status == "ready",
            })
        return {"profile_id": profile_id, "region": "all", "items": items}

    def all_model_validation_models_payload(self) -> dict:
        items = []
        default_key = ""
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
                    "region": "all",
                    "region_name": "全部区域",
                    "scope": "all",
                    "default": path.name == "best.pt" and not default_key,
                    **model_training_metadata(path, "all"),
                }
            except Exception as exc:  # noqa: BLE001 - expose broken checkpoints in the model list.
                item = {
                    "key": key,
                    "label": label,
                    "name": path.parent.name,
                    "weight": path.name,
                    "path": display_path(path),
                    "region": "all",
                    "region_name": "全部区域",
                    "scope": "all",
                    "error": f"{type(exc).__name__}: {exc}",
                    "default": False,
                }
            if item["default"] and not default_key:
                default_key = key
            items.append(item)

        for key, catalog in self.catalogs.items():
            for item in catalog.model_validation_models().get("items", []):
                if item.get("scope") == "all":
                    continue
                global_key = f"{key}/{item['key']}"
                if item.get("default") and not default_key:
                    default_key = global_key
                items.append(
                    {
                        **item,
                        "key": global_key,
                        "label": f"{catalog.region.name} / {item['label']}",
                        "region": key,
                        "region_name": catalog.region.name,
                        "scope": key,
                    }
                )
        items.sort(key=model_sort_key)
        default_key = next((item["key"] for item in items if not item.get("error")), items[0]["key"] if items else "")
        return {"region": "all", "default": default_key, "items": items, "all_regions": True}

    def all_model_validation_random(self, threshold: float = 0.5, model_key: str = "") -> dict:
        region_key, local_model_key = split_global_model_key(model_key)
        if not region_key:
            payload = self.all_model_validation_models_payload()
            if not payload["default"]:
                raise FileNotFoundError("No model weights found")
            region_key, local_model_key = split_global_model_key(payload["default"])
        if region_key == "all":
            return self._all_model_validation_random_global(threshold=threshold, model_key=local_model_key)
        if region_key not in self.catalogs:
            raise FileNotFoundError(f"Region not found for model: {region_key}")
        catalog = self.catalogs[region_key]
        result = catalog.model_validation_random(threshold=threshold, model_key=local_model_key)
        result["model"]["key"] = f"{region_key}/{result['model']['key']}"
        result["model"]["scope"] = region_key
        result["model"]["region_name"] = catalog.region.name
        result["site"]["region"] = region_key
        result["site"]["region_name"] = catalog.region.name
        return result

    def all_model_validation_random_loaded(self, model, threshold: float, model_key: str) -> dict:
        catalogs = list(self.catalogs.items())
        random.shuffle(catalogs)
        errors = []
        for region_key, catalog in catalogs:
            try:
                result = catalog.model_validation_random(threshold=threshold, model=model)
            except FileNotFoundError as exc:
                errors.append(str(exc))
                continue
            result["model"]["key"] = model_key
            result["model"]["profile_model"] = True
            result["site"]["region"] = region_key
            result["site"]["region_name"] = catalog.region.name
            result["region"] = region_key
            return result
        raise FileNotFoundError("No observation site can validate the selected model" + (f": {errors[0]}" if errors else ""))

    def _all_model_validation_random_global(self, threshold: float, model_key: str) -> dict:
        model_path = global_model_path_from_key(model_key)
        if not model_path.exists():
            raise FileNotFoundError(f"Global model not found: {display_path(model_path)}")
        model = load_model_checkpoint(model_path)
        catalogs = list(self.catalogs.items())
        random.shuffle(catalogs)
        skipped = []
        for region_key, catalog in catalogs:
            candidates = list(catalog.sites)
            random.shuffle(candidates)
            for site in candidates:
                rows = catalog._model_validation_rows(site, model.in_channels)
                if not rows:
                    continue
                try:
                    prediction = catalog.model_prediction_for_site(site, threshold=threshold, rows=rows, model=model)
                except self.busy_error:
                    raise
                except Exception as exc:  # noqa: BLE001 - continue looking for a usable validation target.
                    skipped.append(f"{region_key}/{site.site_id}: {type(exc).__name__}: {exc}")
                    continue
                prediction["model"]["key"] = global_model_key(model_path)
                prediction["model"]["scope"] = "all"
                prediction["model"]["region_name"] = "全部区域"
                return {
                    "region": region_key,
                    "site_id": site.site_id,
                    "site": {**catalog._summary(site), "region": region_key, "region_name": catalog.region.name},
                    "model": prediction["model"],
                    "prediction": prediction["prediction"],
                    "stats": prediction["stats"],
                    "imagery": prediction["imagery"],
                    "skipped_count": len(skipped),
                }
        raise FileNotFoundError(
            f"No observation site with active imagery matching global model bands ({model.in_channels}): {display_path(model_path)}"
        )
