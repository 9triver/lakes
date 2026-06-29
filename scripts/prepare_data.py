#!/usr/bin/env python3
"""Download and prepare large local datasets for the Lakes browser.

The generated layout matches the paths used by ``lakes_browser.server`` and
``scripts/build_lake_metadata.py``. Large data stays under each configured
region's ``raw`` directory and is ignored by Git.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lakes_browser.region_config import RegionConfig, load_region_configs  # noqa: E402


REGIONS, DEFAULT_REGION_KEY = load_region_configs()

HYDROLAKES_ZIP_URL = "https://data.hydrosheds.org/file/HydroLAKES/HydroLAKES_polys_v10_shp.zip"
ESA_TILE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
JRC_TILE_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/"
    "{layer}/{layer}_{tile}v1_4_2021.tif"
)

WATER_COLUMN_CANDIDATES = ["fclass", "class", "type", "natural", "water", "landuse"]
OSM_WATER_VALUES = {"water", "reservoir", "lake", "pond", "basin", "wetland", "riverbank", "dock"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--data-dir", type=Path, default=None, help="override configured raw data directory")
    parser.add_argument("--processed-dir", type=Path, default=None, help="override configured processed data directory")
    parser.add_argument("--force", action="store_true", help="overwrite existing generated files")
    parser.add_argument("--proxy", default="", help="proxy URL or host:port for downloads that need it")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("osm", help="download Geofabrik water polygons for the selected region")
    sub.add_parser("hydrolakes", help="download HydroLAKES polygons")
    sub.add_parser("esa", help="download ESA WorldCover tiles and build a regional water mask")
    sub.add_parser("jrc", help="download JRC GSW tiles and build regional clips")
    sub.add_parser("sentinel-grid", help="download Sentinel-2 MGRS tile grid, not SAFE imagery")
    sub.add_parser("sentinel-tiles", help="alias for sentinel-grid; downloads the tile grid only")
    sub.add_parser("metadata", help="generate processed lake metadata for the selected region")
    sub.add_parser("all", help="run public base-data prep and metadata; does not download Sentinel SAFE imagery")

    args = parser.parse_args()
    region = REGIONS[args.region]
    if args.data_dir or args.processed_dir:
        region = replace(
            region,
            data_dir=(args.data_dir.expanduser().resolve() if args.data_dir else region.data_dir),
            processed_dir=(args.processed_dir.expanduser().resolve() if args.processed_dir else region.processed_dir),
        )
    data_dir = region.data_dir
    processed_dir = region.processed_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "osm":
        prepare_osm(region, args.force, args.proxy)
    elif args.command == "hydrolakes":
        prepare_hydrolakes(region, args.force, args.proxy)
    elif args.command == "esa":
        prepare_esa(region, args.force, args.proxy)
    elif args.command == "jrc":
        prepare_jrc(region, args.force, args.proxy)
    elif args.command in {"sentinel-grid", "sentinel-tiles"}:
        prepare_sentinel_tiles(region, args.force)
    elif args.command == "metadata":
        prepare_metadata(region)
    elif args.command == "all":
        prepare_osm(region, args.force, args.proxy)
        prepare_hydrolakes(region, args.force, args.proxy)
        prepare_esa(region, args.force, args.proxy)
        prepare_jrc(region, args.force, args.proxy)
        prepare_sentinel_tiles(region, args.force)
        prepare_metadata_if_possible(region)


def prepare_osm(region: RegionConfig, force: bool, proxy: str = "") -> None:
    out_path = region.osm_water
    if out_path.exists() and not force:
        print(f"exists {display(out_path)}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lakes-osm-") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / f"{region.key}-latest-free.gpkg.zip"
        download_file(region.geofabrik, zip_path, force=True, proxy=proxy)
        extract_zip(zip_path, tmp_dir)
        gpkg = next(tmp_dir.glob("*.gpkg"), None)
        if gpkg is None:
            raise FileNotFoundError("Geofabrik archive did not contain a .gpkg file")
        water = read_osm_water(gpkg)
        if water.empty:
            raise RuntimeError(f"No water polygons found in Geofabrik {region.name} GPKG")
        water = water.to_crs("EPSG:4326")
        water = water[water.geometry.notna() & ~water.geometry.is_empty].copy()
        water = normalize_osm_water_schema(water)
        write_layer(out_path, "osm_water_polygons", water, force=True)
    print(f"wrote {display(out_path)} ({len(water)} features)")


def prepare_hydrolakes(region: RegionConfig, force: bool, proxy: str = "") -> None:
    out_dir = region.data_dir / "hydrolakes"
    shp_path = region.hydrolakes
    if shp_path.exists() and not force:
        print(f"exists {display(shp_path)}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "HydroLAKES_polys_v10_shp.zip"
    download_file(HYDROLAKES_ZIP_URL, zip_path, force=force, proxy=proxy)
    extract_zip(zip_path, out_dir)
    if not shp_path.exists():
        found = next(out_dir.rglob("HydroLAKES_polys_v10.shp"), None)
        if found is None:
            raise FileNotFoundError("HydroLAKES shapefile not found after extraction")
        print(f"wrote {display(found)}")
        return
    print(f"wrote {display(shp_path)}")


def prepare_esa(region: RegionConfig, force: bool, proxy: str = "") -> None:
    out_dir = region.esa_worldcover_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = region.esa_worldcover_clip
    mask_path = region.esa_water_mask
    if clip_path.exists() and mask_path.exists() and not force:
        print(f"exists {display(clip_path)}")
        print(f"exists {display(mask_path)}")
        return
    tile_paths = []
    for tile in region.esa_tiles:
        path = out_dir / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        download_file(ESA_TILE_URL.format(tile=tile), path, force=force, proxy=proxy)
        tile_paths.append(path)
    if region.external_raster_mode == "tiles":
        print(f"wrote {len(tile_paths)} ESA WorldCover source tiles under {display(out_dir)}")
        print("skip regional 10m mosaic; the browser reads intersecting source tiles on demand")
        return
    clip_rasters(tile_paths, clip_path, region_geom(region), force=True)
    build_esa_water_mask(clip_path, mask_path)
    print(f"wrote {display(clip_path)}")
    print(f"wrote {display(mask_path)}")


def prepare_jrc(region: RegionConfig, force: bool, proxy: str = "") -> None:
    out_dir = region.jrc_gsw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = {
        "occurrence": region.jrc_occurrence,
        "seasonality": region.jrc_seasonality,
    }
    if all(path.exists() for path in layers.values()) and not force:
        for path in layers.values():
            print(f"exists {display(path)}")
        return
    for layer, clip_path in layers.items():
        tile_paths = []
        for tile in region.jrc_tiles:
            path = out_dir / f"{layer}_{tile}v1_4_2021.tif"
            download_file(JRC_TILE_URL.format(layer=layer, tile=tile), path, force=force, proxy=proxy)
            tile_paths.append(path)
        if region.external_raster_mode == "tiles":
            print(f"wrote {len(tile_paths)} JRC {layer} source tiles under {display(out_dir)}")
            continue
        clip_rasters(tile_paths, clip_path, region_geom(region), force=True)
        print(f"wrote {display(clip_path)}")


def prepare_sentinel_tiles(region: RegionConfig, force: bool) -> None:
    out_dir = region.data_dir / "sentinel_2_tiles"
    geojson_path, shp_path = region.sentinel_tile_index_paths
    if geojson_path.exists() and shp_path.exists() and not force:
        print(f"exists {display(geojson_path)}")
        print(f"exists {display(shp_path)}")
        return
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "download_sentinel_tile_index.py"), "--out-dir", str(out_dir)]
    run(cmd)


def prepare_metadata(region: RegionConfig) -> None:
    require_paths(
        [
            region.osm_water,
            region.sentinel_tile_index_paths[0],
            region.hydrolakes,
        ]
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_lake_metadata.py"),
        "--region",
        region.key,
    ]
    run(cmd)


def prepare_metadata_if_possible(region: RegionConfig) -> None:
    missing = [
        path
        for path in [
            region.osm_water,
            region.sentinel_tile_index_paths[0],
            region.hydrolakes,
        ]
        if not path.exists()
    ]
    if missing:
        print("skip metadata: missing required inputs")
        for path in missing:
            print(f"missing {display(path)}")
        return
    prepare_metadata(region)


def read_osm_water(gpkg: Path):
    import geopandas as gpd
    import pyogrio

    layers = pyogrio.list_layers(gpkg)
    layer_names = [str(item[0]) for item in layers]
    for layer in preferred_osm_layers(layer_names):
        try:
            gdf = gpd.read_file(gpkg, layer=layer)
        except Exception:
            continue
        if gdf.empty or "geometry" not in gdf:
            continue
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if gdf.empty:
            continue
        water = filter_osm_water(gdf)
        if not water.empty:
            return water
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")


def preferred_osm_layers(layer_names: list[str]) -> list[str]:
    preferred = [
        name
        for name in ["gis_osm_water_a_free", "gis_osm_water_a_free_1", "water", "water_a"]
        if name in layer_names
    ]
    rest = [name for name in layer_names if name not in preferred]
    return preferred + rest


def filter_osm_water(gdf):
    import numpy as np

    columns = set(gdf.columns)
    mask_values = np.zeros(len(gdf), dtype=bool)
    for column in WATER_COLUMN_CANDIDATES:
        if column not in columns:
            continue
        values = gdf[column].astype(str).str.lower()
        mask_values |= values.isin(OSM_WATER_VALUES).to_numpy()
    if "other_tags" in columns:
        tags = gdf["other_tags"].fillna("").astype(str).str.lower()
        mask_values |= tags.str.contains('"natural"=>"water"', regex=False).to_numpy()
        mask_values |= tags.str.contains('"water"=>"', regex=False).to_numpy()
        mask_values |= tags.str.contains('"landuse"=>"reservoir"', regex=False).to_numpy()
    if not mask_values.any():
        return gdf.copy()
    return gdf[mask_values].copy()


def normalize_osm_water_schema(gdf):
    import numpy as np

    gdf = gdf.copy()
    if "osm_way_id" not in gdf.columns:
        gdf["osm_way_id"] = None
    if "type" not in gdf.columns:
        gdf["type"] = gdf["fclass"] if "fclass" in gdf.columns else None
    if "natural" not in gdf.columns:
        gdf["natural"] = "water"
    if "landuse" not in gdf.columns:
        if "fclass" in gdf.columns:
            fclass = gdf["fclass"].astype(str).str.lower()
            gdf["landuse"] = np.where(fclass.eq("reservoir"), "reservoir", None)
        else:
            gdf["landuse"] = None
    if "man_made" not in gdf.columns:
        gdf["man_made"] = None
    if "other_tags" not in gdf.columns:
        if "fclass" in gdf.columns:
            gdf["other_tags"] = ['"water"=>"{}"'.format(value) for value in gdf["fclass"].fillna("").astype(str)]
        else:
            gdf["other_tags"] = ""
    for column in ["osm_id", "osm_way_id", "name", "type", "natural", "landuse", "man_made", "other_tags"]:
        if column not in gdf.columns:
            gdf[column] = None
    return gdf[["osm_id", "osm_way_id", "name", "type", "natural", "landuse", "man_made", "other_tags", "geometry"]]


def build_esa_water_mask(source_path: Path, out_path: Path) -> None:
    import numpy as np
    import rasterio

    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        data = src.read(1)
    water = (data == 80).astype("uint8")
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(water, 1)


def clip_rasters(paths: list[Path], out_path: Path, geom, force: bool) -> None:
    import rasterio
    from rasterio.merge import merge

    if out_path.exists() and not force:
        return
    datasets = [rasterio.open(path) for path in paths]
    try:
        mosaic, transform = merge(datasets, bounds=geom.bounds)
        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="deflate",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()


def download_file(url: str, path: Path, force: bool = False, timeout: int = 120, proxy: str = "") -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"exists {display(path)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    part_path.unlink(missing_ok=True)
    print(f"download {url}")
    if download_with_curl(url, path, proxy):
        print(f"wrote {display(path)}")
        return
    request = Request(url, headers={"User-Agent": "lakes-prepare-data/0.1"})
    try:
        total = 0
        with urlopen(request, timeout=timeout) as response, part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                if total % (25 * 1024 * 1024) < len(chunk):
                    print(f"downloaded {total / 1024 / 1024:.1f} MiB", flush=True)
    except (HTTPError, URLError) as exc:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    part_path.replace(path)
    print(f"wrote {display(path)}")


def download_with_curl(url: str, path: Path, proxy: str = "") -> bool:
    if shutil.which("curl") is None:
        return False
    part_path = path.with_suffix(path.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "30",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-o",
        str(part_path),
        url,
    ]
    if proxy:
        cmd[1:1] = ["--proxy", normalize_proxy(proxy)]
    else:
        cmd[1:1] = ["--noproxy", "*"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        part_path.unlink(missing_ok=True)
        return False
    part_path.replace(path)
    return True


def normalize_proxy(proxy: str) -> str:
    proxy = proxy.strip()
    if not proxy:
        return proxy
    if "://" not in proxy:
        return f"http://{proxy}"
    return proxy


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)


def write_layer(path: Path, layer: str, gdf, force: bool) -> None:
    import pyogrio

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and force:
        path.unlink()
    pyogrio.write_dataframe(gdf, path, layer=layer, driver="GPKG")


def require_paths(paths: list[Path]) -> None:
    missing = [display(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(f"- {path}" for path in missing))


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def region_geom(region: RegionConfig):
    from shapely.geometry import box

    if not region.bounds:
        raise ValueError(f"Region {region.key} has no configured bounds")
    return box(*region.bounds)


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
