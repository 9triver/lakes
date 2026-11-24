"""Build the server's random-site pool from materialized slope caches."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import rasterio
from lake_workbench.imagery.formats import is_local_imagery_source
from lake_workbench.imagery.inventory import ImageryInventoryMixin
from lake_workbench.models.preparation import _slope_ready
from lake_workbench.models.random_site_index import INDEX, indexed_sites
from lake_workbench.workspaces.file_lock import process_lock
from lake_workbench.regions.config import load_region_configs
from lake_workbench.regions.service import _metadata_validation_sites


class LocalInventory(ImageryInventoryMixin):
    def __init__(self, region):
        self.region = region
        self.active_imagery = self._load_active_imagery()
        grouped = defaultdict(list)
        for rows in self._load_user_tci_rows().values():
            for row in rows:
                if row.get("site_id") and is_local_imagery_source(row.get("source")):
                    grouped[row["site_id"]].append(row)
        self.by_site = {
            site_id: sorted(rows, key=lambda row: (str(row.get("date", "")), str(row.get("product", ""))), reverse=True)
            for site_id, rows in grouped.items()
        }

    def _local_imagery_rows_for_site(self, site):
        return self.by_site.get(site.site_id, [])


def main():
    regions, _ = load_region_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regions", nargs="*", metavar="REGION")
    args = parser.parse_args()
    for key in args.regions or regions:
        region = regions[key]
        inventory = LocalInventory(region)
        selected = []
        if inventory.by_site and region.site_metadata.is_file():
            for site in _metadata_validation_sites(inventory):
                row = inventory._active_local_imagery_row(site)
                if row is None:
                    continue
                path = Path(row["tci_path"])
                try:
                    with rasterio.open(path) as image:
                        eligible = image.count >= 5
                    if eligible and _slope_ready(path):
                        selected.append(site.site_id)
                except (OSError, ValueError, EOFError):
                    continue
        with process_lock(INDEX.with_suffix(".lock")):
            index = indexed_sites()
            index[key] = sorted(set(index.get(key, ())) | set(selected))
            temp = INDEX.with_suffix(".build.tmp")
            temp.write_text(json.dumps(index, ensure_ascii=False) + "\n")
            temp.replace(INDEX)
        print(key, len(selected), flush=True)


if __name__ == "__main__":
    main()
