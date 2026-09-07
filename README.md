# Lakes

Lakes is a local Web GIS for multi-region satellite observation, water annotation, and semantic-segmentation training. It combines local imagery, external water providers, training-area capture, Patch review, GPU training, and model validation in one workbench. Sentinel-2 query and download remain available through backend APIs and command-line tools.

## Capabilities

- Browse observation Sites defined by local imagery directories across all configured regions. The current configuration contains 17 provincial regions.
- Display OSM, HydroLAKES, ESA WorldCover, JRC GSW, and local Shapefile annotations independently.
- Generate independent automatic labels from multispectral water evidence or conservative multispectral/OSM map consensus.
- Use local imagery as the primary Site imagery; query and download Sentinel-2 SAFE/TCI products through the API or CLI when needed.
- Generate Workspace-owned Patches directly from the current map extent, imagery, and visible annotations.
- Review Patch include/exclude state and contribute selected included Patches to versioned shared Datasets.
- Train U-Net or Pixel MLP models from Workspace or shared data by region or across all regions.
- Inspect run history and validate selected model weights on random or selected Sites in the unified Model view.

Model predictions are diagnostic overlays and are never captured as training truth.

## Domain Model

The main ownership chain is:

```text
External identity -> User -> default Workspace

Shared observations
  Region / Site / Imagery / Annotation

Workspace-owned training resources
  Source record (provenance) / Logical Patch / Dataset / Run / Model

Shared training resources
  Global Dataset / Dataset version / contributed Patch snapshot
```

A User is an authenticated operator. Production identities come from a verified Cloudflare Access JWT; local development uses a fixed identity. Each User maps one-way to exactly one default Workspace. Workspace remains independent and does not contain `user_id`.

New Workspaces are empty. The default Workspace lazily migrates legacy region-level source records once. A source record preserves the captured view for provenance, while Logical Patches are the user-facing training-data units.

## Repository Layout

```text
config/regions.toml                 Region definitions and paths
frontend/                           React, Vite, TypeScript, MUI, OpenLayers
src/lake_workbench/
  server.py                         Runtime dependency assembly and HTTP entry point
  http_handler.py                   Request context, dispatch, and static serving
  catalog.py                        Regional Site catalog facade
  routes/                           User, Workspace, Site, training, model, Sentinel APIs
  users/                            User registry and default Workspace mapping
  workspaces/                       Workspace state and training artifact ownership
  regions/                          Region configuration and cross-region queries
  imagery/                          Inventory, mosaics, raster rendering, XYZ tiles
  water/                            Unified annotation providers and local labels
  sentinel/                         Product catalog, query, and download
  training/                         Samples, Logical Patches, datasets, and runners
  models/                           Architectures, checkpoints, metadata, validation
scripts/                            Data preparation, metadata, Patch, and training CLIs
data/                               Large local data, ignored by Git
```

## Data Layout

Region data contains shared observation inputs and generated regional metadata. A local-imagery directory represents an observation area of interest; it is not required to be a formally named lake:

```text
data/shared/
  external_water/hydrolakes/
  sentinel_2_tiles/

data/regions/<region>/
  raw/
    local_imagery/<directory-id>/       .img or lossless tiled .tif/.tiff
    external_water/{osm,esa_worldcover,jrc_gsw}/
    sentinel_products/
  processed/
    site_metadata.gpkg
    site_metadata.csv
    sentinel_products.csv
    active_imagery.json
    training_samples.csv
    training_labels/
    esa_polygons/
    jrc_polygons/
```

User and Workspace state is separate:

```text
data/users/users.json
data/workspaces/workspaces.json
data/workspaces/<workspace>/
  samples/<region>/manifest.csv
  samples/<region>/labels/
  site_sources.json
  logical_patches/<region>/
  training_patch_cache/<region>/<dataset-config>/
  training_datasets/<region>/<dataset-config>/
data/global_datasets/<scope>/
  manifest.csv
  versions/<version>.csv
data/models/workspaces/<workspace>/<region-or-all>/<run>/
```

A model run normally contains `config.json`, `history.json`, `manifest.csv`, `best.pt`, and `last.pt`. The validation UI lists only `best.pt` and orders available models by validation IoU.

## Install And Run

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/pip install -e .
cd frontend && npm install && npm run build && cd ..
PYTHONPATH=src .venv/bin/python -m lake_workbench.server --host 0.0.0.0 --port 18765
```

Open `http://127.0.0.1:18765`. The Python server serves the built React application and API. The production frontend caches only basemap tiles actually requested while browsing: Service Worker Cache Storage is used on HTTPS, `localhost`, and `127.0.0.1`; an IndexedDB fallback is used for HTTP access through a machine IP such as `192.168.30.134`. Neither path caches Lakes API responses, local imagery tiles, or training data. The basemap cache is limited to 5,000 tiles and evicts the least recently used entries.

When no authentication variables are supplied, local startup defaults to development authentication and provisions a fixed local identity. The installed user service may instead load Cloudflare settings from `.env`. Development identity values can be overridden with `LAKES_DEV_USER_SUBJECT`, `LAKES_DEV_USER_EMAIL`, and `LAKES_DEV_USER_NAME`.

### Production authentication

Put the application hostname behind a Cloudflare Access self-hosted application, then configure the service environment:

```text
LAKES_AUTH_MODE=cloudflare
LAKES_CF_TEAM_DOMAIN=https://<team>.cloudflareaccess.com
LAKES_CF_AUD=<application-audience-tag>
LAKES_BOOTSTRAP_EMAIL=owner@example.com
LAKES_ADMIN_EMAILS=owner@example.com
LAKES_LOCAL_AUTH_BYPASS=true
LAKES_LOCAL_NETWORK=192.168.30.0/24
LAKES_LOCAL_USER_ID=default
```

`LAKES_CF_AUD` is the Application Audience tag shown by Cloudflare Zero Trust. `LAKES_BOOTSTRAP_EMAIL` binds that first verified identity to the existing default User and Workspace; it can also replace a previous development identity binding. Additional first-time identities receive a new empty Workspace. `LAKES_ADMIN_EMAILS` is a comma-separated list and is evaluated on every request.

The optional local bypass above is intentionally limited to the trusted LAN CIDR and the configured existing User. It applies only when a request has no Cloudflare Access token; requests through the public hostname still use JWT authentication. Any device that can reach the configured LAN address and port can use this bypass, so disable it on an untrusted network.

In Cloudflare Zero Trust, create an Access application under **Access > Applications > Add an application > Self-hosted**, attach the desired identity provider and allow policy, and protect the actual Lakes hostname. Keep the origin reachable only through Cloudflare Tunnel or an equivalent firewall rule; otherwise requests could bypass Access entirely. Lakes verifies the JWT signature, issuer, and audience and does not trust plain identity headers.

The repository service unit reads optional values from `%h/lakes/.env`. Reload it after changing the unit or environment:

```bash
systemctl --user daemon-reload
systemctl --user restart lakes.service
```

For frontend development:

```bash
cd frontend
npm run dev
```

Vite listens on `http://127.0.0.1:5173/` and proxies `/api` to port `18765`. Production builds write `index.html` and `assets/` to `src/lake_workbench/static/`; the Python service exposes that application at its root URL.

The user-level service can be managed with:

```bash
systemctl --user restart lakes.service
systemctl --user status lakes.service
journalctl --user -u lakes.service -f
```

## Prepare Region Data

Download public inputs and build Site metadata:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu all
```

Individual stages are `osm`, `hydrolakes`, `esa`, `jrc`, `sentinel-grid`, and `metadata`. HydroLAKES and the Sentinel grid are shared and need to be downloaded only once. Sentinel SAFE/TCI products are not downloaded by `prepare_data.py`; users select them on demand.

JRC Google Storage can use an explicit proxy during preparation:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py \
  --region gansu --proxy 192.168.30.107:7897 jrc
```

Copernicus requests made by the running Lakes service explicitly ignore proxy environment variables. Credentials are read from `.env` or environment variables:

```text
COPERNICUS_USERNAME=...
COPERNICUS_PASSWORD=...
```

The `光谱 + OSM 一致` automatic label source downloads standard OSM tiles through the explicit proxy in `LAKES_OSM_PROXY` (default `http://192.168.30.107:7897`) and caches them under the region cache. Set `LAKES_OSM_PROXY=` to disable it. This setting is isolated to OSM evidence downloads; other service downloads continue to ignore proxy environment variables.

Rebuild one Site catalog manually with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_site_metadata.py --region gansu --workers 4
```

`site_metadata.gpkg` contains `sites`, `site_coverage_core`, `imagery_assets`, `local_label_features`, and `external_water_features`. Site identity is directory-based: `site_id = <region>_<directory-id>`, for example `gansu_17407`.

`build_site_metadata.py` scans local imagery with four workers by default. Label files are read sequentially because concurrent reads contend on typical data disks. Use `--workers 1` on slower disks or to minimize resource usage; the option is limited to 16 workers.

The preparation scripts have separate responsibilities:

| Script | Purpose |
| --- | --- |
| `prepare_data.py` | Run regional public-data preparation and metadata stages. |
| `download_sentinel_tile_index.py` | Download the global Sentinel-2 MGRS tile grid. |
| `download_sentinel.py` | Query Copernicus products or download a selected SAFE/TCI product. |
| `build_site_metadata.py` | Match local imagery bounds with external water sources and write the regional Site catalog. |
| `convert_local_imagery.py` | Convert local `.img` rasters to lossless tiled BigTIFF with ZSTD; source files are retained. |
| `precompute_esa_polygons.py`, `precompute_jrc_polygons.py` | Precompute vector annotation layers from regional raster inputs. |
| `build_logical_patches.py` | Build Workspace-owned Logical Patches from recorded samples. |
| `build_training_dataset.py` | Materialize a Workspace training dataset for a selected configuration. |
| `train_model.py` | Start a model experiment from the command line. |

For a new region, add its paths, bounds, source imagery root, external-data tiles, and Geofabrik URL to `config/regions.toml`, place local imagery under the configured `raw/local_imagery/`, then run `prepare_data.py` and build the metadata. Large inputs and generated outputs remain under `data/` and are intentionally ignored by Git.

Local imagery can be stored as the original ENVI `.img` product or as a lossless tiled GeoTIFF. When both files have the same product stem, metadata construction uses the GeoTIFF and does not create a duplicate asset; without a GeoTIFF it falls back to `.img`.

To convert a region without cropping, resampling, or deleting the source files:

```bash
PYTHONPATH=src .venv/bin/python scripts/convert_local_imagery.py \
  --region-root /mnt/sda1/lakes-data/seasonal_3period_v1/regions/gansu
PYTHONPATH=src .venv/bin/python scripts/build_site_metadata.py --region gansu
```

The converter writes lossless TIFF files alongside the source files in `raw/local_imagery/` by default, using `Tiled + BigTIFF + ZSTD + PREDICTOR=2`. It preserves dimensions, CRS, transform, dtype, bands, NoData, band descriptions, and tags. Both `.img` and `.tif/.tiff` may coexist in the directory; when they have the same product stem, the TIFF is selected. This compression is lossless and is suitable for patch generation, training, inference, and map rendering.

## Training Workflow

1. In **Observation Sites**, select a Site, imagery, map extent, and trusted visible annotations, then choose **Generate training data**.
2. Lakes preserves an internal source record, generates Logical Patches, and opens **Training Data** filtered to the new batch.
3. Include or exclude individual Patches. Included Patches form the current Workspace training selection and can optionally be contributed to a regional or `all` shared Dataset.
4. In **Models > Train**, select the current Workspace or shared Dataset and start a U-Net or Pixel MLP run. Required NPZ data is materialized automatically.
5. In **Models > Validate**, select model weights and validate on a random or selected Site. A validation view can generate another Patch batch for iterative improvement.

Exact duplicates and high-overlap Patches require replacement confirmation. Excluding a Workspace Patch and withdrawing its shared snapshot are independent operations.

CLI equivalents:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_logical_patches.py \
  --workspace default --region yunnan --sample <sample-id>

PYTHONPATH=src .venv/bin/python scripts/build_training_dataset.py \
  --workspace default --region yunnan --config resize256_v1

PYTHONPATH=src .venv/bin/python scripts/train_model.py \
  --workspace default --region yunnan --model-type unet \
  --dataset-config resize256_v1 --epochs 30 --batch-size 8 --device cuda
```

Logical Patch rebuilds process only Training Samples present in the selected Workspace, plus an explicitly supplied `--sample`. A Patch identity includes its source image, native raster window, window size, label snapshot, and preprocessing inputs; changing the Patch size creates a distinct Patch variant.

## API Shape

Shared observation APIs:

```text
GET  /api/regions
GET  /api/regions/<region-or-all>/sites
GET  /api/regions/<region>/sites/<site_id>
GET  /api/regions/<region>/sites/<site_id>/annotations/<source>
GET  /api/regions/<region>/sites/<site_id>/local-labels
GET  /api/regions/<region>/sites/<site_id>/imagery
```

User APIs:

```text
GET   /api/users
GET   /api/users/<user_id>
POST  /api/users
PATCH /api/users/<user_id>
POST  /api/users/<user_id>/archive
POST  /api/users/<user_id>/restore
```

`GET /api/auth/session` returns the current authenticated User and Workspace mapping. User listing, creation, archival, and restoration are admin-only; a regular User can read or rename only itself.

Workspace-owned APIs:

```text
GET /api/workspaces/<workspace>/source-conflicts
GET /api/workspaces/<workspace>/regions/<region-or-all>/sites
GET /api/workspaces/<workspace>/regions/<region>/training-samples
GET /api/workspaces/<workspace>/regions/<region>/sites/<site_id>/training-samples
GET /api/workspaces/<workspace>/regions/<region-or-all>/logical-patches
GET /api/workspaces/<workspace>/regions/<region-or-all>/training-datasets
GET /api/workspaces/<workspace>/regions/<region-or-all>/training-runs
GET /api/workspaces/<workspace>/regions/<region-or-all>/global-dataset
GET /api/workspaces/<workspace>/regions/<region-or-all>/global-dataset/patches
GET /api/workspaces/<workspace>/regions/<region-or-all>/model-validation/models
GET /api/workspaces/<workspace>/regions/<region-or-all>/model-validation/random
POST /api/workspaces/<workspace>/regions/<region>/sites/<site_id>/training-samples
```

Frontend deep links use `#/users/<user>/workspaces/<workspace>/regions/...` and validate that the Workspace is the User's `default_workspace_id`. Old Profile URLs and APIs are unsupported. Model keys must be fully qualified:

```text
workspaces/<workspace>/<region-or-all>/<run>/<weight>
```

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts
PYTHONPATH=src .venv/bin/ruff check src scripts tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
(cd frontend && npm run typecheck && npm run build)
(cd frontend && LAKES_E2E_BASE_URL=http://127.0.0.1:18765/ npm run test:e2e)
git diff --check
```

`data/`, `attic/`, `.env`, virtual environments, generated frontend assets, imagery, Patches, datasets, and model weights are not committed to Git.
