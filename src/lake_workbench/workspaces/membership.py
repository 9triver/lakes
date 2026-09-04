"""Domain operations for Workspace logical-patch membership.

This module deliberately does not know about files or workspaces. It keeps
membership rules independent from the persistence facade and easy to test.
"""

from __future__ import annotations

from typing import Iterable

from lake_workbench.utils import clean_optional, truthy_flag


Member = tuple[str, str]


def patch_is_included(row: dict) -> bool:
    """Return the effective review state of a logical patch row."""
    status = clean_optional(row.get("review_status"))
    if status in {"included", "excluded"}:
        return status == "included"
    return truthy_flag(row.get("include"), default=True)


def included_members(rows: Iterable[dict], region: str) -> set[Member]:
    """Convert included rows from one region manifest into member keys."""
    return {
        (region, str(row["logical_patch_id"]))
        for row in rows
        if row.get("logical_patch_id") and patch_is_included(row)
    }


def replace_region_members(members: set[Member], region: str, rows: Iterable[dict]) -> set[Member]:
    """Replace one region's membership while preserving other regions."""
    result = {(key, patch_id) for key, patch_id in members if key != region}
    result.update(included_members(rows, region))
    return result


def patch_bbox(row: dict) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(row[key]) for key in ("bounds_left", "bounds_bottom", "bounds_right", "bounds_top"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def patch_bbox_iou(first: dict, second: dict) -> float:
    """Return the intersection-over-union of two geographic patch boxes."""
    left_box = patch_bbox(first)
    right_box = patch_bbox(second)
    if not left_box or not right_box:
        return 0.0
    left = max(left_box[0], right_box[0])
    bottom = max(left_box[1], right_box[1])
    right = min(left_box[2], right_box[2])
    top = min(left_box[3], right_box[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    first_area = max(0.0, left_box[2] - left_box[0]) * max(0.0, left_box[3] - left_box[1])
    second_area = max(0.0, right_box[2] - right_box[0]) * max(0.0, right_box[3] - right_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def find_patch_conflicts(
    canonical: dict[str, dict],
    current_ids: Iterable[str],
    wanted_ids: Iterable[str],
    *,
    minimum_overlap: float = 0.9,
) -> list[dict]:
    """Find wanted patches overlapping an already selected patch.

    Overlap is a conflict only when both rows refer to the same source image.
    A changed label snapshot can therefore still be detected as a duplicate.
    """
    conflicts: list[dict] = []
    wanted = set(wanted_ids)
    for patch_id in wanted:
        incoming = canonical[patch_id]
        incoming_image = incoming.get("image_fingerprint") or incoming.get("image_path")
        for existing_id in set(current_ids) - wanted:
            existing = canonical.get(existing_id)
            if not existing:
                continue
            existing_image = existing.get("image_fingerprint") or existing.get("image_path")
            overlap = patch_bbox_iou(incoming, existing)
            if incoming_image and incoming_image == existing_image and overlap >= minimum_overlap:
                conflicts.append(
                    {
                        "type": "spatial_overlap",
                        "patch_id": patch_id,
                        "existing_patch_id": existing_id,
                        "overlap": round(overlap, 6),
                    }
                )
                break
    return conflicts


def apply_membership_operation(
    members: set[Member],
    region: str,
    patch_ids: Iterable[str],
    operation: str,
    conflicts: Iterable[dict] = (),
) -> set[Member]:
    """Apply include, restore, or exclude to a member set."""
    wanted = {str(value) for value in patch_ids if str(value)}
    result = set(members)
    if operation in {"include", "restore"}:
        result.difference_update((region, row["existing_patch_id"]) for row in conflicts)
        result.update((region, patch_id) for patch_id in wanted)
    elif operation == "exclude":
        result.difference_update((region, patch_id) for patch_id in wanted)
    else:
        raise ValueError(f"Unsupported membership operation: {operation}")
    return result
