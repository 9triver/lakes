#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-2 query and download helpers for the Lakes project."""

from __future__ import annotations

import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COPERNICUS_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
COPERNICUS_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
COPERNICUS_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def query_copernicus_tile_products(
    tile: str,
    start: str,
    end: str,
    cloud: float,
    product_type: str = "MSIL1C",
    limit: int = 50,
) -> list[dict]:
    """Query Copernicus Data Space for Sentinel products by MGRS tile.

    Network requests explicitly ignore proxy environment variables because this
    project usually wants direct Copernicus access.
    """
    tile = str(tile).strip().upper().removeprefix("T")
    product_type = str(product_type or "MSIL1C").strip()
    cloud = max(0.0, min(100.0, float(cloud)))
    limit = max(1, min(200, int(limit)))
    cloud_filter = (
        "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {cloud})"
    )
    query_filter = (
        f"contains(Name,'{product_type}')"
        f" and contains(Name,'_T{tile}_')"
        f" and {cloud_filter}"
        f" and ContentDate/Start ge {start}T00:00:00.000Z"
        f" and ContentDate/Start le {end}T23:59:59.000Z"
    )
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            COPERNICUS_CATALOGUE_URL,
            params={
                "$filter": query_filter,
                "$expand": "Attributes",
                "$orderby": "ContentDate/Start desc",
                "$top": str(limit),
            },
            timeout=60,
        )
    response.raise_for_status()
    products = response.json().get("value", [])
    return [summarize_copernicus_product(product) for product in products]


def summarize_copernicus_product(product: dict) -> dict:
    name = str(product.get("Name", ""))
    return {
        "product_id": product.get("Id"),
        "name": name,
        "tile": product_tile_name(name),
        "date": product_date(name),
        "cloud_cover": product_cloud_cover(product),
        "online": product.get("Online"),
        "content_length": product.get("ContentLength"),
        "origin_date": product.get("OriginDate"),
    }


def load_env_file(env_path: Path | None = None) -> str:
    candidates = [env_path] if env_path else [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
        return str(candidate)
    return ""


def get_copernicus_credentials(env_path: Path | None = None) -> tuple[str, str]:
    load_env_file(env_path)
    username = os.environ.get("COPERNICUS_USERNAME")
    password = os.environ.get("COPERNICUS_PASSWORD")
    if not username or not password:
        raise RuntimeError("请在 .env 或环境变量中设置 COPERNICUS_USERNAME 和 COPERNICUS_PASSWORD")
    return username, password


def get_copernicus_token(username: str, password: str) -> str:
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            COPERNICUS_TOKEN_URL,
            data={
                "grant_type": "password",
                "username": username,
                "password": password,
                "client_id": "cdse-public",
            },
            timeout=120,
        )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Copernicus token response does not contain access_token")
    return token


def download_copernicus_product(
    product: dict,
    out_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    env_path: Path | None = None,
) -> tuple[Path, Path]:
    product_id = _clean_text(product.get("product_id"))
    product_name = _clean_text(product.get("name"))
    if not product_id or not product_name:
        raise ValueError("product_id and name are required")
    product_name = Path(product_name).name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_dir = out_dir / product_name
    tci = find_tci_path(safe_dir)
    if tci:
        return safe_dir, tci

    username, password = get_copernicus_credentials(env_path)
    token = get_copernicus_token(username, password)
    zip_path = out_dir / f"{product_name}.zip"
    part_path = out_dir / f"{product_name}.zip.part"
    url = f"{COPERNICUS_DOWNLOAD_URL}({product_id})/$value"
    downloaded = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Authorization": f"Bearer {token}"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
    with requests.Session() as session:
        session.trust_env = False
        with session.get(url, headers=headers, stream=True, timeout=(60, 300)) as response:
            if response.status_code not in (200, 206):
                raise RuntimeError(f"Copernicus download failed: HTTP {response.status_code}")
            if response.status_code == 200 and downloaded > 0:
                part_path.unlink(missing_ok=True)
                downloaded = 0
            total = int(response.headers.get("Content-Length") or 0) + downloaded
            mode = "ab" if response.status_code == 206 and downloaded > 0 else "wb"
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)
    shutil.move(str(part_path), str(zip_path))
    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a zip: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    tci = find_tci_path(safe_dir)
    if not tci:
        raise FileNotFoundError(f"TCI jp2 not found in {safe_dir}")
    return safe_dir, tci


def find_tci_path(safe_dir: Path) -> Path | None:
    if not safe_dir.exists():
        return None
    candidates = sorted(safe_dir.glob("GRANULE/*/IMG_DATA/*_TCI.jp2"))
    if not candidates:
        candidates = sorted(safe_dir.glob("GRANULE/*/IMG_DATA/R10m/*_TCI_10m.jp2"))
    if not candidates:
        candidates = sorted(safe_dir.rglob("*_TCI*.jp2"))
    return candidates[0] if candidates else None


def product_tile_name(product_name: str) -> str:
    match = re.search(r"_T([0-9A-Z]{5})_", str(product_name))
    return match.group(1) if match else ""


def product_date(product_name: str) -> str:
    match = re.search(r"_MSIL\d[AC]?_(\d{8})T\d{6}", str(product_name))
    if not match:
        match = re.search(r"MSIL1C_(\d{8})T\d{6}", str(product_name))
    if not match:
        return ""
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def product_cloud_cover(product: dict):
    for attribute in product.get("Attributes", []):
        if attribute.get("Name") == "cloudCover":
            return attribute.get("Value")
    return None


def product_type_from_name(product_name: str) -> str:
    match = re.search(r"_MSI(L[12][AC])_", str(product_name))
    return match.group(1) if match else ""


def upsert_csv_row(path: Path, row: dict, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        table = pd.read_csv(path)
    else:
        table = pd.DataFrame(columns=list(row.keys()))
    for column in row:
        if column not in table.columns:
            table[column] = ""
    if key in table.columns and not table.empty:
        table = table[table[key].astype(str) != str(row[key])]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)


def _clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text
