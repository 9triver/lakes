#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local observation imagery, water annotation, and model-training workbench.

This is intentionally dependency-light on the web side: the HTTP server uses
Python's standard library, while GIS IO uses the project environment's
geopandas/rasterio stack.
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from lake_workbench.sentinel.download import disable_proxy_env
from lake_workbench.jobs import DownloadManager, PatchExportManager, TrainingManager
from lake_workbench.catalog import SiteCatalog
from lake_workbench.http_handler import create_site_handler
from lake_workbench.models.metadata import persisted_training_job
from lake_workbench.models.validation import ModelInferenceBusy
from lake_workbench.profiles import ProfileStore
from lake_workbench.regions.config import load_region_configs
from lake_workbench.regions.service import RegionService
from lake_workbench.training.datasets import current_profile_training_dataset_summary
from lake_workbench.training.runner import run_dataset_build, run_patch_export, run_training_job
from lake_workbench.utils import parse_int_or_default


PROJECT_ROOT = Path(__file__).resolve().parents[2]


REGIONS, DEFAULT_REGION_KEY = load_region_configs()




def main() -> None:
    disable_proxy_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    catalogs = {
        key: SiteCatalog(region, DEFAULT_REGION_KEY) for key, region in REGIONS.items()
    }
    profile_store = ProfileStore(REGIONS, model_root=PROJECT_ROOT / "data" / "models")
    default_profile = profile_store.ensure_default_profile()
    downloads_by_region = {
        key: DownloadManager(catalog) for key, catalog in catalogs.items()
    }
    patch_exports_by_region = {
        key: PatchExportManager(catalog, exporter=lambda region, options: run_patch_export(region, options, profile_store))
        for key, catalog in catalogs.items()
    }
    all_patch_exports = PatchExportManager(catalogs=catalogs, exporter=lambda region, options: run_patch_export(region, options, profile_store))
    dataset_builds_by_region = {
        key: PatchExportManager(catalog, exporter=lambda region, options: run_dataset_build(region, options, profile_store))
        for key, catalog in catalogs.items()
    }
    all_dataset_builds = PatchExportManager(catalogs=catalogs, exporter=lambda region, options: run_dataset_build(region, options, profile_store))
    training_managers: dict[tuple[str, str], TrainingManager] = {}

    def training_manager(profile_id: str, scope: str) -> TrainingManager:
        key = (profile_id, scope)
        if key not in training_managers:
            training_managers[key] = TrainingManager(
                scope,
                profile_id=profile_id,
                model_root=PROJECT_ROOT / "data" / "models" / "profiles" / profile_id,
                legacy_model_dir=(PROJECT_ROOT / "data" / "models" / scope) if profile_id == "default" else None,
                dataset_summary=lambda selected_scope, config_id, selected_profile=profile_id: current_profile_training_dataset_summary(profile_store, selected_profile, selected_scope, config_id),
                runner=lambda selected_scope, options, progress_callback=None, cancel_event=None: run_training_job(
                    selected_scope,
                    options,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    profile_store=profile_store,
                ),
                persisted_job_loader=persisted_training_job,
                parse_epochs=parse_int_or_default,
            )
        return training_managers[key]

    handler = create_site_handler(
        catalogs=catalogs,
        downloads_by_region=downloads_by_region,
        patch_exports_by_region=patch_exports_by_region,
        dataset_builds_by_region=dataset_builds_by_region,
        all_patch_exports=all_patch_exports,
        all_dataset_builds=all_dataset_builds,
        default_region_key=DEFAULT_REGION_KEY,
        region_service=RegionService(catalogs, DEFAULT_REGION_KEY, ModelInferenceBusy),
        profile_store=profile_store,
        training_manager_factory=training_manager,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Lakes Workbench running: http://{args.host}:{args.port}")
    print(f"Default Profile: {default_profile['name']} ({default_profile['status']}, {default_profile['selected_patch_count']} patches)")
    for key, catalog in catalogs.items():
        status = "ready" if catalog.load_error is None else catalog.load_error
        imagery_summary = catalog.imagery_inventory_summary()
        print(
            f"Region {key}: {len(catalog.sites)} sites, "
            f"{imagery_summary['tci_tile_count']} imagery tiles, "
            f"{imagery_summary['active_imagery_count']} active imagery selections, {status}"
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
