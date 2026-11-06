"""Build an explicitly approved batch with optional per-site period limits."""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from decimal import Decimal

from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.runner import run_dataset_build, run_patch_export
from lake_workbench.utils import read_csv_records, resolve_data_path, write_csv_records
from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.workspaces.membership import patch_is_included


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/chongqing_five_site_batches"
OLD_REPORT = ROOT / "reports/chongqing_seed_all_20261013"
REGIONS, _ = load_region_configs()
REGION = REGIONS["chongqing"]
STORE = WorkspaceStore(REGIONS, model_root=ROOT / "data/models")
MANIFEST = ROOT / "data/workspaces/default/logical_patches/chongqing/manifest.csv"


def save(name, value):
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_ids():
    return set(load(OLD_REPORT / "water_filter_policy.json")["previous_patch_ids"])


def review_state(row):
    return {key: row.get(key, "") for key in ("include", "review_status", "exclude_reason", "patch_notes")}


def _threshold_fraction(value):
    numerator, denominator = Decimal(str(value)).as_integer_ratio()
    return int(numerator), int(denominator)


def exclusion_reasons(row, minimum_water_ratio="0.05", minimum_valid_ratio="0.80"):
    water_n, water_d = _threshold_fraction(minimum_water_ratio)
    valid_n, valid_d = _threshold_fraction(minimum_valid_ratio)
    valid = int(row.get("valid_pixels") or 0)
    water = int(row.get("water_pixels") or 0)
    width = int(row.get("window_width") or row.get("logical_size") or 0)
    height = int(row.get("window_height") or row.get("logical_size") or 0)
    reasons = []
    if valid <= 0 or water * water_d < valid * water_n:
        reasons.append("water_ratio_below_" + str(minimum_water_ratio))
    if valid <= 0 or (width > 0 and height > 0 and valid * valid_d < width * height * valid_n):
        reasons.append("valid_ratio_below_" + str(minimum_valid_ratio))
    return reasons


def excluded_copy(row, reasons):
    return {**row, "include": "false", "included": "false", "review_status": "excluded",
            "exclude_reason": "; ".join(reasons)}


def filter_new_rows(rows, sample_ids, previous_ids, minimum_water_ratio="0.05", minimum_valid_ratio="0.80"):
    new = [row for row in rows if row.get("sample_id") in sample_ids
           and row["logical_patch_id"] not in previous_ids]
    reasons = {row["logical_patch_id"]: exclusion_reasons(row, minimum_water_ratio, minimum_valid_ratio)
               for row in new}
    low = [row for row in new if reasons[row["logical_patch_id"]]]
    # Preserve all rows and manual exclusions, including their notes/reasons.
    marked = [excluded_copy(row, reasons[row["logical_patch_id"]])
              if reasons.get(row["logical_patch_id"]) and patch_is_included(row) else row
              for row in rows]
    return new, low, marked


def merge_filtered_rows(current, archived, site_ids, sample_ids, protected_ids):
    """Append only missing scoped rows as excluded; never alter an existing row."""
    index = {row["logical_patch_id"]: row for row in current}
    if len(index) != len(current):
        raise ValueError("Duplicate active Patch IDs")
    incoming = {}
    for row in archived:
        key = row["logical_patch_id"]
        if row.get("site_id") not in site_ids or row.get("sample_id") not in sample_ids or key in protected_ids:
            raise ValueError("Recovery row outside the approved scope")
        if key in incoming and incoming[key] != row:
            raise ValueError("Conflicting recovery snapshots")
        incoming[key] = row
    restored = []
    for key, row in incoming.items():
        if key in index:
            continue
        reasons = exclusion_reasons(row)
        if not reasons:
            raise ValueError("Missing Patch does not match a previous threshold exclusion: " + key)
        restored.append(excluded_copy(row, reasons))
    return list(current) + restored, restored


def initialize():
    if (REPORT / "initialization.json").exists():
        result = load(REPORT / "initialization.json")
        assert Path(result["archive"]).is_dir(), "Missing reset archive"
        return result
    previous = legacy_ids()
    rows = read_csv_records(MANIFEST)
    old = [row for row in rows if row["logical_patch_id"] in previous]
    assert len(previous) == len(old) == 690
    assert len(rows) == 381357, "Unexpected active inventory; do not reset blindly"
    policy = load(OLD_REPORT / "water_filter_policy.json")
    sample_ids = set(policy["sample_ids"])
    assert all(row.get("sample_id") in sample_ids for row in rows if row["logical_patch_id"] not in previous)
    active = MANIFEST.parent
    workspace = active.parent.parent.resolve()
    archive = workspace / "patch_archives" / f"chongqing_superseded_{time.strftime('%Y%m%d_%H%M%S')}"
    staging = active.with_name(active.name + ".legacy_reset")
    assert archive.resolve().is_relative_to(workspace) and staging.resolve().is_relative_to(workspace)
    assert not archive.exists() and not staging.exists()
    staging.mkdir(parents=True)
    for row in old:
        for field in ("preview_path", "preview_base_path"):
            if not row.get(field):
                continue
            source = resolve_data_path(row[field], REGION)
            if source.is_file() and source.resolve().is_relative_to(active.resolve()):
                target = staging / source.resolve().relative_to(active.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    write_csv_records(staging / "manifest.csv", old)
    sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    save("legacy_review_before.json", {row["logical_patch_id"]: review_state(row) for row in old})
    archive.parent.mkdir(parents=True, exist_ok=True)
    active.rename(archive)
    try:
        staging.rename(active)
    except BaseException:
        archive.rename(active)
        raise
    result = {"status": "initialized", "archive": str(archive), "archived_manifest_sha256": sha,
              "discarded_new_patches": len(rows) - len(old), "preserved_legacy_patches": len(old),
              "training_started": False, "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save("initialization.json", result)
    verify_legacy()
    print(json.dumps(result), flush=True)
    return result


def verify_legacy():
    wanted = load(REPORT / "legacy_review_before.json")
    current = {row["logical_patch_id"]: row for row in read_csv_records(MANIFEST)}
    assert len(wanted) == 690 and set(wanted).issubset(current)
    assert all(review_state(current[key]) == state for key, state in wanted.items()), "Legacy review state changed"


def select_plan_samples(samples, plan):
    site_ids = set(plan["site_ids"])
    period_limit = plan.get("periods_per_site")
    if period_limit is None:
        assert 1 <= len(site_ids) <= 5
        expected = {row["sample_id"] for row in samples if row.get("site_id") in site_ids}
    else:
        assert period_limit == 3 and plan["period_selection"] == "website_first_three_newest"
        assert 1 <= len(site_ids) <= 10
        expected = set()
        for site_id in site_ids:
            ordered = sorted((row for row in samples if row.get("site_id") == site_id),
                             key=lambda row: (row.get("product_date", ""), row.get("product_name", "")),
                             reverse=True)
            assert len(ordered) >= period_limit, "Site has fewer than three available samples"
            expected.update(row["sample_id"] for row in ordered[:period_limit])
    requested = set(plan["sample_ids"])
    assert len(requested) == len(plan["sample_ids"]) and requested == expected, "Requested periods differ from the approved selection"
    selected = [row for row in samples if row.get("sample_id") in requested]
    assert len(selected) == len(requested)
    assert plan["period_status_counts"].get("sample_saved", 0) == len(selected)
    return selected


def generate(plan_path):
    assert (REPORT / "initialization.json").exists(), "Reset must finish before a new batch"
    plan = load(plan_path)
    assert plan["source"] == "spectral_osm_intersection" and plan["max_dimension"] == 2048
    minimum_water_ratio = plan["minimum_water_ratio"]
    minimum_valid_ratio = plan.get("minimum_valid_ratio", "0.80")
    assert minimum_water_ratio == "0.05" and minimum_valid_ratio == "0.80" and plan["wait_for_human_review"] is True
    name = plan["batch_id"]
    assert name.startswith("batch_") and name.replace("_", "").isalnum()
    report_name = name + "_report.json"
    existing = load(REPORT / report_name) if (REPORT / report_name).exists() else {}
    if existing.get("status") == "awaiting_human_review":
        print(json.dumps(existing), flush=True)
        return
    # A later batch always requires a separately prepared plan after human review.
    for other in REPORT.glob("batch_*_report.json"):
        state = load(other)
        if other.name != report_name and state.get("status") != "human_review_completed":
            raise RuntimeError("Previous batch still awaits human review")
    samples = read_csv_records(STORE.ensure_workspace_training_samples("default", "chongqing"))
    selected = select_plan_samples(samples, plan)
    requested = set(plan["sample_ids"])
    baseline_path = REPORT / (name + "_before_generation.csv")
    if not baseline_path.exists():
        write_csv_records(baseline_path, read_csv_records(MANIFEST))
    baseline = {row["logical_patch_id"]: row for row in read_csv_records(baseline_path)}
    state = {"status": "generating_patches", "pid": os.getpid(), **plan,
             "training_started": False, "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save(report_name, state)
    try:
        export = run_patch_export("chongqing", {"workspace_id": "default", "sample_ids": sorted(requested),
                                              "patch_size": plan["patch_size"], "stride": plan["stride"]}, STORE)
        assert export["samples"] == len(selected)
        rows = read_csv_records(MANIFEST)
        previous = legacy_ids()
        new, low, kept = filter_new_rows(rows, requested, previous, minimum_water_ratio, minimum_valid_ratio)
        write_csv_records(REPORT / (name + "_before_water_filter.csv"), rows)
        write_csv_records(REPORT / (name + "_below_one_percent.csv"), low)
        with STORE.transaction():
            temporary = MANIFEST.with_suffix(".filtered.csv.tmp")
            write_csv_records(temporary, kept)
            temporary.replace(MANIFEST)
            STORE._logical_rows_cache.pop(("default", "chongqing"), None)
            STORE._selection_summary_cache.pop("default", None)
        verify_legacy()
        state.update(status="building_training_files", generated_new_patches=len(new),
                     excluded_below_threshold=len(low), retained_new_patches=len(new) - len(low),
                     low_quality_action="retain_as_excluded",
                     minimum_valid_ratio=minimum_valid_ratio, export=export)
        save(report_name, state)
        print(json.dumps({key: value for key, value in state.items() if key != "sample_ids"}), flush=True)
        dataset = run_dataset_build("chongqing", {"workspace_id": "default", "config_id": "resize256_v1"}, STORE)
        from lake_workbench.training.dataset_status import workspace_training_dataset_status
        status = workspace_training_dataset_status(REGION, "resize256_v1", STORE, "default")
        assert status["ready"], "Training dataset not ready"
        actual = read_csv_records(resolve_data_path(dataset["manifest"], REGION))
        assert len(actual) == dataset["patches"] and all(resolve_data_path(row["npz_path"], REGION).is_file() for row in actual)
        if name == "batch_001":
            assert all(row.get("site_id") in plan["site_ids"] for row in actual)
        verify_legacy()
        active_new = [row for row in read_csv_records(MANIFEST) if row.get("sample_id") in requested and row["logical_patch_id"] not in previous]
        assert len(active_new) == len(new), "Filtered Patch rows must remain imported"
        assert all(not patch_is_included(row) or not exclusion_reasons(
            row, minimum_water_ratio, minimum_valid_ratio) for row in active_new)
        current_index = {row["logical_patch_id"]: row for row in read_csv_records(MANIFEST)}
        for key, old in baseline.items():
            assert key in current_index and all(current_index[key].get(field, "") == value
                                                for field, value in old.items()), "An existing Patch record changed"
        selected_site_ids = set(plan["site_ids"])
        assert all(row.get("sample_id") in requested for key, row in current_index.items()
                   if key not in baseline and row.get("site_id") in selected_site_ids)
        state.update(preserved_existing_patches=len(baseline), existing_rows_unchanged=True)
        state.update(status="awaiting_human_review", verified=True, dataset=dataset,
                     training_files=len(actual), dataset_status=status,
                     included_new_patches=sum(patch_is_included(row) for row in active_new),
                     preserved_legacy_patches=690, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        save(report_name, state)
        print(json.dumps({key: value for key, value in state.items() if key != "sample_ids"}), flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        save(report_name, state)
        raise


def rethreshold(minimum_water_ratio="0.05"):
    """Apply a stricter water threshold to already exported new patches."""
    previous = legacy_ids()
    rows = read_csv_records(MANIFEST)
    _, low, kept = filter_new_rows(rows, {row.get("sample_id") for row in rows},
                                   previous, minimum_water_ratio, "0.80")
    verify_legacy()
    if low:
        write_csv_records(REPORT / "batch_001_below_threshold_005.csv", low)
        with STORE.transaction():
            temporary = MANIFEST.with_suffix(".threshold005.csv.tmp")
            write_csv_records(temporary, kept)
            temporary.replace(MANIFEST)
            STORE._logical_rows_cache.pop(("default", "chongqing"), None)
            STORE._selection_summary_cache.pop("default", None)
    verify_legacy()
    dataset = run_dataset_build("chongqing", {"workspace_id": "default", "config_id": "resize256_v1"}, STORE)
    from lake_workbench.training.dataset_status import workspace_training_dataset_status
    status = workspace_training_dataset_status(REGION, "resize256_v1", STORE, "default")
    assert status["ready"], "Training dataset not ready"
    result = {"status": "completed", "minimum_water_ratio": minimum_water_ratio,
              "excluded_below_threshold": len(low), "active_patches": len(kept),
              "preserved_legacy_patches": len(previous), "dataset": dataset,
              "dataset_status": status, "training_started": False,
              "updated_at": time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save("threshold_005_report.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["initialize", "generate", "rethreshold"])
    parser.add_argument("--plan", type=Path, default=REPORT / "batch_001_plan.json")
    args = parser.parse_args()
    if fcntl is None:
        raise SystemExit("Run this worker on the Linux Lakes server")
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize() if args.mode == "initialize" else rethreshold() if args.mode == "rethreshold" else generate(args.plan)
