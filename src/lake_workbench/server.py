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
from lake_workbench.regions.config import load_region_configs
from lake_workbench.regions.service import RegionService
from lake_workbench.training.datasets import current_training_dataset_summary
from lake_workbench.training.runner import run_patch_export, run_training_job
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
    downloads_by_region = {
        key: DownloadManager(catalog) for key, catalog in catalogs.items()
    }
    patch_exports_by_region = {
        key: PatchExportManager(catalog, exporter=run_patch_export)
        for key, catalog in catalogs.items()
    }
    all_patch_exports = PatchExportManager(catalogs=catalogs, exporter=run_patch_export)
    training_manager_options = {
        "model_root": PROJECT_ROOT / "data" / "models",
        "dataset_summary": current_training_dataset_summary,
        "runner": run_training_job,
        "persisted_job_loader": persisted_training_job,
        "parse_epochs": parse_int_or_default,
    }
    training_runs_by_scope = {
        **{key: TrainingManager(key, **training_manager_options) for key in catalogs},
        "all": TrainingManager("all", **training_manager_options),
    }
    handler = create_site_handler(
        catalogs=catalogs,
        downloads_by_region=downloads_by_region,
        patch_exports_by_region=patch_exports_by_region,
        training_runs_by_scope=training_runs_by_scope,
        all_patch_exports=all_patch_exports,
        default_region_key=DEFAULT_REGION_KEY,
        region_service=RegionService(catalogs, DEFAULT_REGION_KEY, ModelInferenceBusy),
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Lakes Workbench running: http://{args.host}:{args.port}")
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
