# Lakes

Local Web GIS prototype for browsing Hunan lake and reservoir imagery.

The app serves a lightweight browser UI backed by local GIS data:

- OSM water polygons for the selectable lake list and main boundary overlay.
- Sentinel-2 TCI imagery served as local XYZ raster tiles.
- HydroLAKES reference polygons.
- ESA WorldCover-derived smoothed water polygons.
- JRC Global Surface Water occurrence polygons.

The browser map uses OpenLayers. Sentinel imagery is a raster tile layer, while
OSM, HydroLAKES, ESA, and JRC boundaries are rendered as independent vector
layers.

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

## Sentinel Download

The Lakes repo has its own Copernicus query/download helper:

```bash
PYTHONPATH=src python scripts/download_sentinel.py query \
  --tile 49RFN \
  --start 2025-06-01 \
  --end 2025-08-31 \
  --cloud 20
```

Download one product from the query output:

```bash
PYTHONPATH=src python scripts/download_sentinel.py download \
  --product-id PRODUCT_UUID \
  --name PRODUCT_NAME.SAFE \
  --cloud-cover 12.3
```

Set Copernicus credentials in `COPERNICUS_USERNAME` and
`COPERNICUS_PASSWORD`, or put them in `.env`. Copernicus requests explicitly
ignore proxy environment variables.
