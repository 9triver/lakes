"""Disk-backed index for sites with an existing aligned slope raster."""
import json

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.workspaces.file_lock import process_lock

INDEX = PROJECT_ROOT / "data/terrain/random_validation_sites.json"


def indexed_sites():
    try:
        return json.loads(INDEX.read_text())
    except FileNotFoundError:
        return {}


def remember_slope_ready(region, site_id):
    with process_lock(INDEX.with_suffix(".lock")):
        document = indexed_sites()
        sites = document.setdefault(region, [])
        if site_id in sites:
            return
        sites.append(site_id)
        temp = INDEX.with_suffix(".update.tmp")
        temp.write_text(json.dumps(document, ensure_ascii=False) + "\n")
        temp.replace(INDEX)
