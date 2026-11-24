"""Cross-region views used by the HTTP API."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from lake_workbench.paths import PROJECT_ROOT


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _random_site_candidates(catalogs, *, site_loader=None):
    """Yield sites uniformly across provinces, loading only selected catalogs.

    Counts come from cached lists or metadata headers, without loading every
    province's geometry. Sparse Fisher-Yates removes attempted sites lazily;
    there is no full-site copy, shuffle, or imagery scan per request.
    """
    pools = []
    total = 0
    for region_key, catalog in catalogs.items():
        count = catalog.site_count()
        if count:
            pools.append([region_key, catalog, count, None, {}])
            total += count
    while total:
        slot = random.randrange(total)
        for pool in pools:
            if slot < pool[2]:
                break
            slot -= pool[2]
        region_key, catalog, remaining, sites, swaps = pool
        if sites is None:
            sites = site_loader(catalog) if site_loader is not None else catalog.sites
            pool[3] = sites
            if len(sites) != remaining:
                # Loading may reveal unavailable/stale metadata. Reweight before
                # drawing any site from this catalog, preserving equal odds.
                total += len(sites) - remaining
                pool[2] = len(sites)
                continue
        index = swaps.get(slot, slot)
        remaining -= 1
        replacement = swaps.pop(remaining, remaining)
        if slot != remaining:
            swaps[slot] = replacement
        pool[2] = remaining
        total -= 1
        yield region_key, catalog, sites[index]


@lru_cache(maxsize=32)
def _cached_validation_metadata(path_string, mtime_ns, size):
    """Read Site geometry only; do not initialize the province raster inventory."""
    from types import SimpleNamespace
    from lake_workbench.catalog import SiteCatalog
    holder = SimpleNamespace(region=SimpleNamespace(site_metadata=Path(path_string)))
    return SiteCatalog._load_sites(holder)

def _metadata_validation_sites(catalog):
    path = catalog.region.site_metadata
    stat = path.stat()
    return _cached_validation_metadata(str(path), stat.st_mtime_ns, stat.st_size)


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
            site_count = catalog.site_count()
            items.append(
                {
                    "key": key,
                    "name": region.name,
                    "default": key == self.default_region_key,
                    "bounds": list(region.bounds) if region.bounds else None,
                    "ready": catalog.load_error is None,
                    "load_error": catalog.load_error,
                    "site_count": site_count,
                    **catalog.imagery_inventory_summary(),
                    "has_metadata": region.site_metadata.exists(),
                    "has_osm_water": region.osm_water.exists(),
                    "metadata_path": display_path(region.site_metadata),
                    "osm_water_path": display_path(region.osm_water),
                }
            )
        return {"default": self.default_region_key, "items": items}

    def all_sites_payload(self, query: str, limit: int, offset: int, filters: dict, workspace_store=None, workspace_id: str | None = None) -> dict:
        merged = []
        total = 0
        for key, catalog in self.catalogs.items():
            payload = catalog.list_sites(query=query, limit=10**9, offset=0, filters=filters)
            total += payload["total"]
            counts = workspace_store.patch_counts(workspace_id, key) if workspace_store else {}
            merged.extend({
                **item,
                "region": key,
                "region_name": catalog.region.name,
                **({"included_logical_patch_count": counts.get(item.get("site_id", ""), 0)} if workspace_store else {}),
            } for item in payload["items"])
        merged.sort(key=lambda item: (-(item.get("coverage_area_km2") or 0), item.get("region", ""), item.get("site_id", "")))
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": merged[offset : offset + limit],
            "all_regions": True,
        }

    def all_training_samples_payload(self, workspace_store=None, workspace_id: str | None = None) -> dict:
        items = []
        total = 0
        for key, catalog in self.catalogs.items():
            samples_path = (
                workspace_store.ensure_workspace_training_samples(workspace_id, key)
                if workspace_store and workspace_id
                else None
            )
            payload = catalog.list_training_samples(samples_path)
            total += payload.get("total", 0)
            items.extend(
                {**item, "region": key, "region_name": catalog.region.name}
                for item in payload.get("items", [])
            )
        items.sort(key=lambda item: (item.get("region", ""), item.get("created_at", ""), item.get("sample_id", "")), reverse=True)
        return {"total": total, "items": items, "all_regions": True}

    def all_workspace_logical_patches_payload(
        self,
        workspace_store,
        workspace_id: str,
        include: str = "",
        site_id: str = "",
        sample_id: str = "",
    ) -> dict:
        items = []
        for key, catalog in self.catalogs.items():
            manifest = workspace_store.ensure_workspace_logical_patch_manifest(workspace_id, key)
            payload = catalog.list_logical_patches(
                include=include,
                site_id=site_id,
                sample_id=sample_id,
                manifest_path=manifest,
            )
            for item in payload["items"]:
                included = bool(item.get("included"))
                items.append({
                    **item,
                    "included": included,
                    "include": "true" if included else "false",
                    "region": key,
                    "region_name": catalog.region.name,
                    "workspace_id": workspace_id,
                    "preview_url": f"/api/workspaces/{workspace_id}/regions/{key}/logical-patches/{item.get('logical_patch_id', '')}/preview.png",
                })
        items.sort(key=lambda item: (item.get("region", ""), item.get("sample_id", ""), int(item.get("image_index") or 0), int(item.get("row_off") or 0), int(item.get("col_off") or 0)))
        included_count = sum(1 for item in items if item["included"])
        return {"workspace_id": workspace_id, "total": len(items), "included_count": included_count, "excluded_count": len(items) - included_count, "items": items, "all_regions": True}

    def all_workspace_training_dataset_statuses(self, workspace_store, workspace_id: str) -> dict:
        from lake_workbench.training.logical_patches import dataset_configs, workspace_training_dataset_status

        items = []
        for config_id, config in dataset_configs().items():
            regions = [
                workspace_training_dataset_status(catalog.region, config_id, workspace_store, workspace_id)
                for catalog in self.catalogs.values()
            ]
            relevant = [item for item in regions if item["status"] != "missing_selection"]
            statuses = {item["status"] for item in relevant}
            if not relevant:
                status = "missing_selection"
            elif statuses == {"ready"}:
                status = "ready"
            elif "stale" in statuses:
                status = "stale"
            else:
                status = "missing"
            items.append({
                "workspace_id": workspace_id,
                "config_id": config_id,
                "config": config,
                "regions": regions,
                "patches": sum(item["patches"] for item in relevant),
                "status": status,
                "ready": status == "ready",
            })
        return {"workspace_id": workspace_id, "region": "all", "items": items}

    def random_model_validation_target(self, region="") -> dict:
        """Draw from the materialized index, rechecking only sampled sites."""
        from shapely.geometry import mapping
        from lake_workbench.models.preparation import _slope_ready
        from lake_workbench.models.random_site_index import indexed_sites

        index = indexed_sites()
        scope = (region,) if region and region != "all" else self.catalogs
        candidates = [
            (key, site_id)
            for key in scope if key in self.catalogs
            for site_id in index.get(key, ())
        ]
        random.shuffle(candidates)
        metadata = {}
        for region_key, site_id in candidates:
            catalog = self.catalogs[region_key]
            if region_key not in metadata:
                metadata[region_key] = {site.site_id: site for site in _metadata_validation_sites(catalog)}
            site = metadata[region_key].get(site_id)
            if site is None:
                continue
            rows = catalog._model_validation_rows(site, 5)
            if not rows:
                continue
            active = max(rows, key=lambda row: float(row.get("valid_ratio", 0) or 0))
            try:
                if not _slope_ready(Path(active["tci_path"])):
                    continue
            except (OSError, ValueError, EOFError):
                continue
            return dict(region=region_key, site_id=site.site_id, site=dict(
                site_id=site.site_id, region=region_key, region_name=catalog.region.name,
                display_name=site.display_name, coverage_area_km2=site.area_km2,
                bbox=list(site.bbox), geometry=mapping(site.geometry),
                image_count=int(site.properties.get("image_count") or 0)))
        raise FileNotFoundError("所选区域没有当前影像已缓存坡度的站点")

    def all_model_validation_random_loaded(self, model, threshold: float, model_key: str) -> dict:
        errors = []
        for region_key, catalog, site in _random_site_candidates(self.catalogs):
            try:
                result = catalog.model_validation_random(
                    threshold=threshold, model=model, candidates=(site,),
                )
            except FileNotFoundError as exc:
                errors.append(str(exc))
                continue
            result["model"]["key"] = model_key
            result["model"]["workspace_model"] = True
            result["site"]["region"] = region_key
            result["site"]["region_name"] = catalog.region.name
            result["region"] = region_key
            return result
        raise FileNotFoundError("No observation site can validate the selected model" + (f": {errors[0]}" if errors else ""))
