"""Shared parsing, path, CSV, and serialization helpers."""

from __future__ import annotations

import calendar
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from lake_workbench.regions.config import RegionConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def write_training_label(
    region: RegionConfig,
    sample_id: str,
    layer: dict,
    extra_properties: dict | None = None,
) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    path = region.training_label_dir / f"{safe_id}.geojson"
    properties = {**(layer.get("properties") or {}), **(extra_properties or {})}
    if layer.get("type") == "FeatureCollection":
        payload = {
            "type": "FeatureCollection",
            "properties": properties,
            "features": layer.get("features") or [],
        }
    else:
        payload = {
            "type": "Feature",
            "properties": properties,
            "geometry": layer.get("geometry"),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def first_present(*values: Any) -> str:
    for value in values:
        text = clean_optional(value)
        if text:
            return text
    return ""


def parse_float(value: Any) -> float | None:
    text = clean_optional(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_float_or_default(value: Any, default: float) -> float:
    parsed = parse_float(value)
    return default if parsed is None else parsed


def parse_int_or_default(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def truthy_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none", "null"}:
            return default
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def split_commas(value: Any) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def split_semicolon(value: Any) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        table = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return []
    return table.to_dict("records")


def write_csv_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def default_sentinel_date_range() -> tuple[str, str]:
    today = date.today()
    start = add_months(today, -2)
    return start.isoformat(), today.isoformat()


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def metadata_tiles(value: Any) -> list[str]:
    text = clean_optional(value)
    if not text:
        return []
    return [tile.strip() for tile in text.split(",") if tile.strip()]


def count_values(values: Any) -> list[dict]:
    counts = {}
    for value in values:
        text = clean_optional(value)
        if text:
            counts[text] = counts.get(text, 0) + 1
    return [{"value": key, "count": counts[key]} for key in sorted(counts)]


def area_in_bucket(area_km2: float, bucket: str) -> bool:
    if bucket == "gte100":
        return area_km2 >= 100
    if bucket == "10_100":
        return 10 <= area_km2 < 100
    if bucket == "1_10":
        return 1 <= area_km2 < 10
    if bucket == "0_1_1":
        return 0.1 <= area_km2 < 1
    if bucket == "lt0_1":
        return area_km2 < 0.1
    return True


def resolve_data_path(value: Any, region: RegionConfig) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "data_download" and parts[1] == "downloads":
        return region.data_dir.joinpath(*parts[2:])
    return PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def is_frontend_route(path: str) -> bool:
    if path == "/":
        return True
    first = path.strip("/").split("/", 1)[0]
    return first in {"profiles", "regions", "sites", "training", "model"}
