# Lakes

Local Web GIS prototype for browsing Hunan lake and reservoir imagery.

The app serves a lightweight browser UI backed by local GIS data:

- OSM water polygons for the selectable lake list and main boundary overlay.
- Sentinel-2 TCI imagery clipped around the selected water body.
- HydroLAKES reference polygons.
- ESA WorldCover-derived smoothed water polygons.

## Layout

```text
lakes/
  src/lakes_browser/        Python server and static web UI
  data/raw/                 Local data copies, ignored by Git
  data/manifests/           Data inventory and setup notes
  scripts/                  Local helper commands
```

## Data

Large data is intentionally not tracked in Git. In this workspace, `data/raw` contains local copies of the browser data.

See `data/manifests/hunan_browser_data.md` for the expected files.

## Run

From this directory:

```bash
export PYTHONPATH=src
python -m lakes_browser.server --host 0.0.0.0 --port 8765
```

Or:

```bash
PYTHONPATH=src HOST=0.0.0.0 PORT=8765 scripts/run_dev.sh
```

Then open:

```text
http://127.0.0.1:8765
```

## Build Metadata

Generate the unified Hunan lake metadata table:

```bash
PYTHONPATH=src python scripts/build_lake_metadata.py
```

Outputs:

- `data/processed/hunan_lake_metadata.gpkg`
- `data/processed/hunan_lake_metadata.csv`
