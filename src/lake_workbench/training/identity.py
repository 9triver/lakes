"""Stable identity and spatial similarity for captured training views."""

import hashlib
import json

from lake_workbench.utils import clean_optional, parse_float, parse_int_or_default


def rounded_float(value, digits: int = 5):
    parsed = parse_float(value)
    return None if parsed is None else round(parsed, digits)


def normalized_view_extent(view_state: dict, digits: int = 5) -> list[float]:
    map_state = view_state.get("map") if isinstance(view_state.get("map"), dict) else {}
    extent = map_state.get("extent") if isinstance(map_state.get("extent"), list) else []
    if len(extent) != 4:
        return []
    values = [rounded_float(item, digits) for item in extent]
    return [] if any(item is None for item in values) else values


def bbox_iou(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    bottom = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    top = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def bbox_from_row(row: dict) -> list[float]:
    values = [
        rounded_float(row.get("view_west")),
        rounded_float(row.get("view_south")),
        rounded_float(row.get("view_east")),
        rounded_float(row.get("view_north")),
    ]
    return [] if any(item is None for item in values) else values


def training_view_signature(
    site_id: str,
    product_key: str,
    label_source: str,
    label_threshold: str,
    view_state: dict,
) -> tuple[str, str, list[float]]:
    visible = view_state.get("visible_layers") if isinstance(view_state.get("visible_layers"), dict) else {}
    generated_source_keys = ("spectral_water", "spectral_osm_consensus")
    label_layers = {
        key: bool(visible.get(key))
        for key in (
            "osm",
            "hydrolakes",
            "context_osm",
            "context_hydrolakes",
            "esa",
            "jrc",
            "local_label",
            *generated_source_keys,
        )
    }
    local_label = view_state.get("selected_local_label") if isinstance(view_state.get("selected_local_label"), dict) else {}
    generated_labels = (
        view_state.get("selected_generated_labels")
        if isinstance(view_state.get("selected_generated_labels"), dict)
        else {}
    )
    extent = normalized_view_extent(view_state)
    base_payload = {
        "site_id": site_id,
        "product_key": product_key,
        "label_source": label_source,
        "label_threshold": str(label_threshold or ""),
        "label_layers": label_layers,
        "jrc_threshold": parse_int_or_default(view_state.get("jrc_threshold"), 75),
        "local_label_id": clean_optional(local_label.get("id")) or "",
        "local_label_path": clean_optional(local_label.get("path")) or "",
        "generated_label_ids": {
            source: clean_optional((generated_labels.get(source) or {}).get("id"))
            or ""
            for source in generated_source_keys
            if bool(visible.get(source))
            and isinstance(generated_labels.get(source), dict)
        },
    }
    exact_payload = {**base_payload, "extent": extent}
    base_json = json.dumps(base_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    exact_json = json.dumps(exact_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        hashlib.sha1(exact_json.encode("utf-8")).hexdigest(),
        hashlib.sha1(base_json.encode("utf-8")).hexdigest(),
        extent,
    )
