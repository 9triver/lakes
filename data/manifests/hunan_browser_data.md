# Hunan Browser Data

The browser expects data under `data/raw`. Large GIS and Sentinel assets are not tracked by Git.

Current local data copies:

| Path | Source | Purpose |
| --- | --- | --- |
| `data/raw/hunan_osm_water` | Local copy from `../data_download/downloads/hunan_osm_water` | OSM water polygons used for the lake list and OSM overlay. |
| `data/raw/hunan_single_tiles` | Local copy from `../data_download/downloads/hunan_single_tiles` | Sentinel-2 TCI products and the selected tile index. |
| `data/raw/hydrolakes` | Local copy from `../data_download/downloads/hydrolakes` | HydroLAKES reference polygons. |
| `data/raw/hunan_external_water` | Local copy from `../data_download/downloads/hunan_external_water` | ESA WorldCover and JRC GSW derived water rasters. |

Required files:

- `data/raw/hunan_osm_water/hunan_water_raw.gpkg`
- `data/raw/hunan_single_tiles/hunan_selected_best_coverage_tci_valid.csv`
- `data/raw/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp`
- `data/raw/hunan_external_water/esa_worldcover/hunan_esa_water_mask.tif`

If data lives elsewhere, set `LAKES_DATA_DIR` to the directory containing these four subdirectories.
