#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local observation imagery, water annotation, and model-training workbench.

This is intentionally dependency-light on the web side: the HTTP server uses
Python's standard library, while GIS IO uses the project environment's
geopandas/rasterio stack.
"""

from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from lake_workbench.sentinel.download import disable_proxy_env
from lake_workbench.jobs import DownloadManager, PatchExportManager, TrainingManager
from lake_workbench.catalog import SiteCatalog
from lake_workbench.http_handler import create_site_handler
from lake_workbench.models.metadata import persisted_training_job
from lake_workbench.models.validation import ModelInferenceBusy
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.regions.config import load_region_configs
from lake_workbench.regions.service import RegionService
from lake_workbench.training.datasets import current_global_training_dataset_summary, current_workspace_training_dataset_summary
from lake_workbench.training.runner import run_dataset_build, run_global_dataset_build, run_patch_export, run_training_job
from lake_workbench.training.registry import DatasetRegistry
from lake_workbench.users import UserStore
from lake_workbench.auth import AuthenticationService
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
    workspace_store = WorkspaceStore(REGIONS, model_root=PROJECT_ROOT / "data" / "models")
    dataset_registry = DatasetRegistry(PROJECT_ROOT / "data" / "global_datasets")
    default_workspace = workspace_store.ensure_default_workspace()
    user_store = UserStore(workspace_store)
    default_user = user_store.ensure_default_user()
    auth_service = AuthenticationService.from_env()
    bootstrap_email = os.environ.get("LAKES_BOOTSTRAP_EMAIL", "").strip().lower()
    admin_emails = {
        value.strip().lower()
        for value in os.environ.get("LAKES_ADMIN_EMAILS", "").split(",")
        if value.strip()
    }
    downloads_by_region = {
        key: DownloadManager(catalog) for key, catalog in catalogs.items()
    }
    patch_exports_by_region = {
        key: PatchExportManager(catalog, exporter=lambda region, options: run_patch_export(region, options, workspace_store))
        for key, catalog in catalogs.items()
    }
    all_patch_exports = PatchExportManager(catalogs=catalogs, exporter=lambda region, options: run_patch_export(region, options, workspace_store))
    dataset_builds_by_region = {
        key: PatchExportManager(catalog, exporter=lambda region, options: run_dataset_build(region, options, workspace_store))
        for key, catalog in catalogs.items()
    }
    all_dataset_builds = PatchExportManager(catalogs=catalogs, exporter=lambda region, options: run_dataset_build(region, options, workspace_store))
    global_dataset_builds = PatchExportManager(
        catalog=catalogs[DEFAULT_REGION_KEY],
        exporter=lambda _region, options: run_global_dataset_build(options, workspace_store, dataset_registry),
    )
    training_managers: dict[tuple[str, str], TrainingManager] = {}

    def training_manager(workspace_id: str, scope: str) -> TrainingManager:
        key = (workspace_id, scope)
        if key not in training_managers:
            training_managers[key] = TrainingManager(
                scope,
                workspace_id=workspace_id,
                model_root=PROJECT_ROOT / "data" / "models" / "workspaces" / workspace_id,
                dataset_summary=lambda selected_scope, config_id, source, selected_workspace=workspace_id: (
                    current_global_training_dataset_summary(dataset_registry, selected_scope, config_id)
                    if source == "global"
                    else current_workspace_training_dataset_summary(workspace_store, selected_workspace, selected_scope, config_id)
                ),
                runner=lambda selected_scope, options, progress_callback=None, cancel_event=None: run_training_job(
                    selected_scope,
                    options,
                    workspace_store,
                    dataset_registry,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
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
        user_store=user_store,
        workspace_store=workspace_store,
        auth_service=auth_service,
        bootstrap_email=bootstrap_email,
        admin_emails=admin_emails,
        training_manager_factory=training_manager,
        dataset_registry=dataset_registry,
        global_dataset_builds=global_dataset_builds,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Lakes Workbench running: http://{args.host}:{args.port}")
    print(f"Default Workspace: {default_workspace['name']} ({default_workspace['status']}, {default_workspace['selected_patch_count']} patches)")
    print(f"Default User: {default_user['name']} -> {default_user['default_workspace_id']}")
    print(f"Authentication: {auth_service.mode}")
    for key, catalog in catalogs.items():
        site_count = catalog.site_count()
        status = "ready" if catalog.load_error is None else catalog.load_error
        imagery_summary = catalog.imagery_inventory_summary()
        print(
            f"Region {key}: {site_count} sites, "
            f"{imagery_summary['tci_tile_count']} imagery tiles, "
            f"{imagery_summary['active_imagery_count']} active imagery selections, {status}"
        )
    server.serve_forever()


if __name__ == "__main__":
    main()
