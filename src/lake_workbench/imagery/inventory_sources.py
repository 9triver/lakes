"""Readers and writers for region imagery inventory files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lake_workbench.sentinel.download import product_date
from lake_workbench.utils import (
    clean_optional,
    parse_float_or_default,
    resolve_data_path,
)


def load_tci_index(region) -> dict[str, dict]:
    """Load precomputed tile imagery rows for a region."""
    if not region.tci_index.exists():
        return {}
    rows = {}
    for row in pd.read_csv(region.tci_index).to_dict("records"):
        path_text = clean_optional(row.get("tci_path"))
        path = resolve_data_path(path_text, region) if path_text else None
        tile = clean_optional(row.get("tile"))
        if not tile or not path:
            continue
        normalized_tile = tile.upper().removeprefix("T")
        rows[normalized_tile] = {
            "tile": normalized_tile,
            "date": str(row["date"]),
            "source": row.get("source", ""),
            "valid_ratio": parse_float_or_default(row.get("valid_ratio"), 0.0),
            "product": row.get("product", ""),
            "cloud_cover": row.get("cloud_cover", ""),
            "tci_path": path,
            "missing": not path.exists(),
        }
    return rows


def load_user_tci_rows(region) -> dict[str, list[dict]]:
    """Load downloaded product rows grouped by normalized tile name."""
    rows: dict[str, list[dict]] = {}
    if not region.user_sentinel_index.exists():
        return rows
    for row in pd.read_csv(region.user_sentinel_index).to_dict("records"):
        path_text = clean_optional(row.get("tci_path"))
        path = resolve_data_path(path_text, region) if path_text else None
        safe_path_text = clean_optional(row.get("safe_path")) or ""
        tile = str(row.get("tile", "")).upper().removeprefix("T")
        if not tile:
            continue
        item = {
            "tile": tile,
            "site_id": clean_optional(row.get("site_id")) or "",
            "date": clean_optional(row.get("date"))
            or product_date(row.get("product_name", "")),
            "source": clean_optional(row.get("source")) or "user_download",
            "valid_ratio": parse_float_or_default(row.get("valid_ratio"), 1.0),
            "product": clean_optional(row.get("product_name")) or "",
            "product_id": clean_optional(row.get("product_id")) or "",
            "cloud_cover": clean_optional(row.get("cloud_cover")) or "",
            "downloaded_at": clean_optional(row.get("downloaded_at")) or "",
            "safe_path": resolve_data_path(safe_path_text, region)
            if safe_path_text
            else None,
            "label_path": (
                resolve_data_path(clean_optional(row.get("label_path")), region)
                if clean_optional(row.get("label_path"))
                else None
            ),
            "tci_path": path,
            "missing": not path or not path.exists(),
        }
        rows.setdefault(tile, []).append(item)
    for tile in rows:
        rows[tile].sort(
            key=lambda item: (str(item.get("date", "")), str(item.get("product", ""))),
            reverse=True,
        )
    return rows


def load_active_imagery(path: Path) -> dict[str, str]:
    """Read active product selections and normalize legacy tile keys."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    active = {}
    for key, value in payload.items():
        text = str(key)
        if text.startswith("site:"):
            active[text] = str(value)
        elif ":" in text:
            site_id, tile = text.split(":", 1)
            active[f"{site_id}:{tile.upper().removeprefix('T')}"] = str(value)
        else:
            active[text.upper().removeprefix("T")] = str(value)
    return active


def save_active_imagery(path: Path, active: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")
