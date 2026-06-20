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
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "sentinel_2_tiles"
DEFAULT_URLS = [
    "https://sentinels.copernicus.eu/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml",
    "https://sentinel.esa.int/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml",
]
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
TILE_RE = re.compile(r"^\d{2}[A-Z]{3}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--url", action="append", help="KML URL to try before the built-in URLs")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    kml_path = out_dir / "sentinel_2_tiling_grid.kml"
    geojson_path = out_dir / "sentinel_2_index.geojson"
    shp_path = out_dir / "sentinel_2_index_shapefile.shp"

    urls = list(args.url or []) + DEFAULT_URLS
    source_url = download_first(urls, kml_path, timeout=args.timeout)
    features = parse_kml_features(kml_path)
    if not features:
        raise RuntimeError(f"No Sentinel-2 tile features parsed from {kml_path}")

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
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("failed to download Sentinel-2 tiling grid:\n" + "\n".join(errors))


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
