#!/usr/bin/env python3
"""Download the Sentinel-2 MGRS tile grid for the Lakes browser."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lakes_browser.region_config import load_region_configs  # noqa: E402


REGIONS, DEFAULT_REGION_KEY = load_region_configs()
DEFAULT_URLS = [
    "https://sentinels.copernicus.eu/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml",
    "https://sentinel.esa.int/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml",
]
DEFAULT_GEOJSON_URLS = [
    "https://zenodo.org/records/10998972/files/sentinel2_tiling_grid_wgs84.geojson?download=1",
]
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
TILE_RE = re.compile(r"^\d{2}[A-Z]{3}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(REGIONS), default=DEFAULT_REGION_KEY)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--url", action="append", help="KML URL to try before the built-in URLs")
    parser.add_argument("--geojson-url", action="append", help="GeoJSON URL to try if KML download fails")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve() if args.out_dir else (REGIONS[args.region].data_dir / "sentinel_2_tiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    kml_path = out_dir / "sentinel_2_tiling_grid.kml"
    geojson_path = out_dir / "sentinel_2_index.geojson"
    shp_path = out_dir / "sentinel_2_index_shapefile.shp"

    source_url = ""
    features = []
    urls = list(args.url or []) + DEFAULT_URLS
    try:
        source_url = download_first(urls, kml_path, timeout=args.timeout)
        features = parse_kml_features(kml_path)
    except Exception as exc:
        print(f"KML download failed, trying GeoJSON fallback: {exc}", file=sys.stderr)
        geojson_urls = list(args.geojson_url or []) + DEFAULT_GEOJSON_URLS
        source_url, features = download_geojson_first(geojson_urls, geojson_path, timeout=args.timeout)
    if not features:
        raise RuntimeError("No Sentinel-2 tile features parsed")

    payload = {
        "type": "FeatureCollection",
        "name": "sentinel_2_index",
        "source_url": source_url,
        "features": features,
    }
    geojson_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {display_path(geojson_path)} ({len(features)} tiles)")

    if write_shapefile_if_possible(features, shp_path):
        print(f"wrote {display_path(shp_path)}")
    else:
        print("shapefile skipped: install project GIS dependencies to generate it")


def download_first(urls: list[str], path: Path, timeout: int) -> str:
    errors = []
    for url in urls:
        try:
            clear_proxy_env()
            part_path = path.with_suffix(path.suffix + ".part")
            if path.exists() and not part_path.exists():
                path.replace(part_path)
            downloaded = part_path.stat().st_size if part_path.exists() else 0
            headers = {"User-Agent": "lakes-browser/0.1"}
            if downloaded:
                headers["Range"] = f"bytes={downloaded}-"
            request = Request(url, headers=headers)
            opener = build_opener(ProxyHandler({}))
            total = downloaded
            with opener.open(request, timeout=timeout) as response:
                if downloaded and getattr(response, "status", None) == 200:
                    downloaded = 0
                    total = 0
                    part_path.unlink(missing_ok=True)
                mode = "ab" if downloaded else "wb"
                with part_path.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        total += len(chunk)
                        if total % (25 * 1024 * 1024) < len(chunk):
                            print(f"downloaded {total / 1024 / 1024:.1f} MiB", flush=True)
            if not part_path.read_bytes()[:4096].strip().startswith(b"<"):
                part_path.unlink(missing_ok=True)
                raise RuntimeError("response is not KML/XML")
            ET.parse(part_path)
            part_path.replace(path)
            print(f"downloaded {url}")
            print(f"wrote {display_path(path)}")
            return url
        except Exception as exc:  # noqa: BLE001 - try all mirrors and report aggregate.
            path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("failed to download Sentinel-2 tiling grid:\n" + "\n".join(errors))


def download_geojson_first(urls: list[str], path: Path, timeout: int) -> tuple[str, list[dict]]:
    errors = []
    for url in urls:
        try:
            download_plain(url, path, timeout=timeout)
            features = parse_geojson_features(path)
            if not features:
                raise RuntimeError("No Sentinel-2 tile features parsed from GeoJSON")
            print(f"downloaded {url}")
            print(f"wrote {display_path(path)}")
            return url, features
        except Exception as exc:  # noqa: BLE001 - try all mirrors and report aggregate.
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("failed to download Sentinel-2 tiling grid GeoJSON:\n" + "\n".join(errors))


def download_plain(url: str, path: Path, timeout: int) -> None:
    clear_proxy_env()
    part_path = path.with_suffix(path.suffix + ".part")
    part_path.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "lakes-browser/0.1"})
    opener = build_opener(ProxyHandler({}))
    total = 0
    with opener.open(request, timeout=timeout) as response:
        with part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                if total % (25 * 1024 * 1024) < len(chunk):
                    print(f"downloaded {total / 1024 / 1024:.1f} MiB", flush=True)
    part_path.replace(path)


def clear_proxy_env() -> None:
    for key in [
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "ftp_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "FTP_PROXY",
    ]:
        os.environ.pop(key, None)


def parse_kml_features(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    features = []
    for placemark in root.findall(".//kml:Placemark", KML_NS):
        name = text_or_empty(placemark.find("kml:name", KML_NS)).strip().upper().removeprefix("T")
        if not TILE_RE.fullmatch(name):
            continue
        geometry = placemark_geometry(placemark)
        if geometry is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"Name": name},
                "geometry": geometry,
            }
        )
    features.sort(key=lambda item: item["properties"]["Name"])
    return features


def parse_geojson_features(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        name = str(properties.get("Name") or properties.get("name") or "").strip().upper().removeprefix("T")
        if not TILE_RE.fullmatch(name):
            continue
        geometry = feature.get("geometry")
        if not geometry:
            continue
        features.append({"type": "Feature", "properties": {"Name": name}, "geometry": strip_z_geometry(geometry)})
    features.sort(key=lambda item: item["properties"]["Name"])
    return features


def strip_z_geometry(geometry: dict) -> dict:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geom_type == "Polygon":
        return {"type": geom_type, "coordinates": strip_z_polygon(coordinates)}
    if geom_type == "MultiPolygon":
        return {"type": geom_type, "coordinates": [strip_z_polygon(poly) for poly in coordinates]}
    return geometry


def strip_z_polygon(polygon):
    return [[[float(point[0]), float(point[1])] for point in ring] for ring in polygon]


def placemark_geometry(placemark: ET.Element) -> dict | None:
    polygons = []
    for polygon in placemark.findall(".//kml:Polygon", KML_NS):
        rings = polygon_rings(polygon)
        if rings:
            polygons.append(rings)
    if not polygons:
        return None
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def polygon_rings(polygon: ET.Element) -> list[list[list[float]]]:
    rings = []
    outer = polygon.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
    if outer is None:
        return rings
    rings.append(parse_coordinates(text_or_empty(outer)))
    for inner in polygon.findall(".//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS):
        rings.append(parse_coordinates(text_or_empty(inner)))
    return [ring for ring in rings if len(ring) >= 4]


def parse_coordinates(value: str) -> list[list[float]]:
    coords = []
    for item in value.split():
        parts = item.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        coords.append([lon, lat])
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def write_shapefile_if_possible(features: list[dict], shp_path: Path) -> bool:
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except Exception:
        return False

    records = [
        {
            "Name": feature["properties"]["Name"],
            "geometry": shape(feature["geometry"]),
        }
        for feature in features
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    for path in shp_path.parent.glob(f"{shp_path.stem}.*"):
        path.unlink()
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    return True


def text_or_empty(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else element.text


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI should print a concise failure.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
