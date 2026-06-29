# Lakes

Local Web GIS prototype for browsing regional lake and reservoir imagery.

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
  data/regions/<region>/    Region-scoped raw and processed data, ignored by Git
  config/regions.toml       Region definitions and normalized data paths
  scripts/                  Local helper commands
```

## Data

Large data is intentionally not tracked in Git. Region data is organized under
`data/regions/<region>/raw` and `data/regions/<region>/processed`.

Each processed region directory contains the metadata and user-managed browser
state for that region:

- `lake_metadata.gpkg`
- `lake_metadata.csv`
- `esa_polygons/`
- `jrc_polygons/`
- `sentinel_products.csv`
- `active_imagery.json`
- `training_samples.csv`
- `training_labels/`

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

Generate the lake metadata table for a configured region:

```bash
PYTHONPATH=src python scripts/build_lake_metadata.py --region hunan
PYTHONPATH=src python scripts/build_lake_metadata.py --region gansu
```

Outputs:

- `data/regions/<region>/processed/lake_metadata.gpkg`
- `data/regions/<region>/processed/lake_metadata.csv`

To prepare raw data and then build metadata:

```bash
PYTHONPATH=src python scripts/prepare_data.py --region hunan all
PYTHONPATH=src python scripts/prepare_data.py --region gansu all
```

## Sentinel Download

The Lakes repo has its own Copernicus query/download helper:

```bash
PYTHONPATH=src python scripts/download_sentinel.py query \
  --region hunan \
  --tile 49RFN \
  --start 2025-06-01 \
  --end 2025-08-31 \
  --cloud 20
```

Download one product from the query output:

```bash
PYTHONPATH=src python scripts/download_sentinel.py download \
  --region hunan \
  --product-id PRODUCT_UUID \
  --name PRODUCT_NAME.SAFE \
  --cloud-cover 12.3
```

Set Copernicus credentials in `COPERNICUS_USERNAME` and
`COPERNICUS_PASSWORD`, or put them in `.env`. Copernicus requests explicitly
ignore proxy environment variables.
