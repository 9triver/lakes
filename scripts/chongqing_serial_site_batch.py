"""Export an explicitly authorized site group serially, retaining excluded patches."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from collections import Counter, defaultdict

from chongqing_five_site_batches import (
    ROOT, REPORT, REGION, STORE, MANIFEST, load, save, legacy_ids,
    filter_new_rows, exclusion_reasons, verify_legacy,
)
from lake_workbench.training.runner import run_patch_export, run_dataset_build
from lake_workbench.training.patch_processing import training_label_fingerprint
from lake_workbench.training.dataset_status import workspace_training_dataset_status
from lake_workbench.utils import read_csv_records, write_csv_records, resolve_data_path
from lake_workbench.workspaces.membership import patch_is_included

BATCH = os.environ.get("CQ_BATCH_ID", "batch_006")
PLAN = REPORT / (BATCH + "_plan.json")
REPORT_NAME = BATCH + "_report.json"
SAMPLES = ROOT / "data/workspaces/default/samples/chongqing/manifest.csv"
# Evidence from the original high-confidence seed run: no usable polygon/sample.
KNOWN_UNAVAILABLE = {
    "chongqing_22319_S2B_MSIL1C_20250410": {
        "status": "no_seed_water", "label_id": "derived_ea9ca781b8b38e051dca",
        "reason": "Original spectral_osm_intersection result had zero polygons; no saved sample",
        "water_pixels": 8, "polygon_count": 0,
    }
}

def get(suffix):
    with urllib.request.urlopen("http://127.0.0.1:18765/api/" + suffix, timeout=60) as response:
        return json.load(response)

def select_periods(site_ids, samples, imagery, unavailable):
    groups = defaultdict(list)
    for row in samples:
        groups[row["site_id"]].append(row)
    periods, chosen = {}, []
    for site in site_ids:
        # Use the first two available website periods; some sites have only one.
        assets = imagery[site]["assets"][:2]
        periods[site] = []
        for asset in assets:
            matches = [row for row in groups[site] if row["product_name"] == asset["product"]]
            item = {key: asset[key] for key in ("product", "date", "asset_id")}
            if len(matches) == 1:
                row = matches[0]
                assert row.get("context_sources") == "spectral_osm_intersection"
                item.update(status="sample_saved", sample_id=row["sample_id"])
                chosen.append(row)
            else:
                assert not matches, "Duplicate or ambiguous saved sample: " + asset["asset_id"]
                item.update(status="no_seed_water" if not asset.get("missing") else "missing_imagery",
                            reason=("No saved high-confidence seed sample for this period" if not asset.get("missing")
                                    else (asset.get("missing_reason") or "Imagery unavailable")))
            periods[site].append(item)
    assert len({row["sample_id"] for row in chosen}) == len(chosen)
    return periods, chosen

def signatures(rows):
    paths = {resolve_data_path(row["label_path"], REGION) for row in rows if row.get("label_path")}
    # A few legacy imported rows point to an old dataset-build placeholder;
    # preserve those rows while fingerprinting only labels still on disk.
    return {str(path): training_label_fingerprint(path) for path in sorted(paths) if path.is_file()}

def prepare():
    assert not PLAN.exists(), "Existing plan must be resumed, not replaced"
    seen = set()
    for path in REPORT.glob("batch_*_plan.json"):
        seen.update(load(path)["site_ids"])
    samples = read_csv_records(SAMPLES)
    sites = [site for site in sorted({row["site_id"] for row in samples}) if site not in seen][:50]
    assert sites, "No remaining Chongqing sites"
    imagery = {}
    for index, site in enumerate(sites, 1):
        imagery[site] = get("regions/chongqing/sites/" + site + "/imagery")
        if index % 10 == 0:
            print("Selected website periods for", index, "sites", flush=True)
    periods, chosen = select_periods(sites, samples, imagery, KNOWN_UNAVAILABLE)
    runs = get("workspaces/default/regions/chongqing/training-runs?dataset_config_id=resize256_v1&dataset_source=workspace")
    assert not any(row.get("status") in {"queued", "running", "starting"} for row in runs.get("items", []))
    current = read_csv_records(MANIFEST)
    assert not any(row.get("site_id") in sites and row["logical_patch_id"] not in legacy_ids() for row in current)
    write_csv_records(REPORT / (BATCH + "_before_generation.csv"), current)
    save(BATCH + "_annotation_signatures.json", signatures(current + chosen))
    save(BATCH + "_samples_signature.json", {"sha256": hashlib.sha256(SAMPLES.read_bytes()).hexdigest()})
    plan = {
        "batch_id": BATCH, "site_ids": sites, "site_count": len(sites), "periods_per_site": 2,
        "period_selection": "website_first_two_newest", "registered_periods": sum(len(v) for v in periods.values()),
        "selected_periods": periods, "sample_ids": [row["sample_id"] for row in chosen],
        "period_status_counts": dict(Counter(p["status"] for ps in periods.values() for p in ps)),
        "source": "spectral_osm_intersection", "max_dimension": 2048,
        "minimum_water_ratio": "0.05", "minimum_valid_ratio": "0.80", "patch_size": 256, "stride": 128,
        "low_quality_action": "retain_as_excluded", "training_started": False,
        "groups": [sites[offset:offset+10] for offset in range(0, len(sites), 10)],
        "authorization": "User requested the next 50 new sites; prior first-three-period rule and review preservation continue",
        "wait_for_human_review": True, "internal_groups_require_review": False,
        "existing_patches": len(current), "existing_included": sum(map(patch_is_included, current)),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save(PLAN.name, plan)
    print(json.dumps({"prepared": PLAN.name, "sites": len(sites), "periods": plan["period_status_counts"],
                      "baseline": len(current), "included": plan["existing_included"]}), flush=True)

def check_baseline(current, baseline):
    index = {row["logical_patch_id"]: row for row in current}
    assert len(index) == len(current), "Duplicate Patch IDs"
    for key, old in baseline.items():
        assert key in index and all(index[key].get(field, "") == value for field, value in old.items()), "Existing Patch changed: " + key

def publish_filtered(rows):
    temporary = MANIFEST.with_suffix(".batch005.csv.tmp")
    write_csv_records(temporary, rows)
    temporary.replace(MANIFEST)
    STORE._logical_rows_cache.pop(("default", "chongqing"), None)
    STORE._selection_summary_cache.pop("default", None)

def scoped_summary(rows, sample_ids, baseline_ids):
    new = [row for row in rows if row.get("sample_id") in sample_ids and row["logical_patch_id"] not in baseline_ids]
    return {
        "patches": len(new), "included": sum(map(patch_is_included, new)),
        "excluded": sum(not patch_is_included(row) for row in new),
        "below_threshold": sum(bool(exclusion_reasons(row)) for row in new),
        "site_ids": sorted({row["site_id"] for row in new}),
    }

def run():
    plan = load(PLAN)
    assert plan["site_count"] == len(plan["site_ids"]) and len(set(plan["site_ids"])) == plan["site_count"]
    assert plan["source"] == "spectral_osm_intersection" and plan["max_dimension"] == 2048
    assert plan["minimum_water_ratio"] == "0.05" and plan["minimum_valid_ratio"] == "0.80"
    assert plan["period_selection"] == "website_first_two_newest" and plan["periods_per_site"] == 2
    assert plan["low_quality_action"] == "retain_as_excluded"
    state = load(REPORT / REPORT_NAME) if (REPORT / REPORT_NAME).exists() else {"groups": []}
    if state.get("status") == "awaiting_human_review" and state.get("verified"):
        print("Already complete; no export submitted", flush=True)
        return
    sample_ids = set(plan["sample_ids"])
    selected = [row for row in read_csv_records(SAMPLES) if row["sample_id"] in sample_ids]
    assert len(selected) == len(sample_ids) == plan["period_status_counts"]["sample_saved"]
    assert hashlib.sha256(SAMPLES.read_bytes()).hexdigest() == load(REPORT / (BATCH + "_samples_signature.json"))["sha256"]
    baseline = {row["logical_patch_id"]: row for row in read_csv_records(REPORT / (BATCH + "_before_generation.csv"))}
    state.update(batch_id=BATCH, pid=os.getpid(), training_started=False)
    def checkpoint(status, **details):
        state.update(status=status, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details)
        save(REPORT_NAME, state)
        print(json.dumps({key: value for key, value in state.items() if key not in {"dataset_status", "dataset", "per_site"}}), flush=True)
    try:
        verify_legacy()
        for index, sites in enumerate(plan["groups"]):
            requested = {row["sample_id"] for row in selected if row["site_id"] in sites}
            rows = read_csv_records(MANIFEST)
            check_baseline(rows, baseline)
            done = next((group for group in state["groups"] if group["index"] == index and group["status"] == "completed"), None)
            if done:
                assert scoped_summary(rows, requested, baseline)["patches"] == done["patches"]
                continue
            if not requested:
                state["groups"].append({"index": index, "status": "completed", "samples": 0,
                                        "sites": sites, "patches": 0, "included": 0,
                                        "excluded": 0, "below_threshold": 0, "site_ids": sites})
                checkpoint("group_completed", completed_sites=sum(len(group["sites"]) for group in state["groups"]))
                continue
            checkpoint("generating_patches", current_group=index+1, completed_sites=index*10)
            receipt_name = BATCH + "_group_" + str(index+1) + "_export.json"
            receipt_path = REPORT / receipt_name
            if receipt_path.exists():
                export = load(receipt_path)
                assert export["samples"] == len(requested)
            else:
                # A partial group after a crash must be inspected rather than overwritten.
                assert not any(row.get("sample_id") in requested for row in rows), "Unreceipted group found; inspect committed rows before recovery"
                export = run_patch_export("chongqing", {"workspace_id": "default",
                    "sample_ids": sorted(requested), "patch_size": 256, "stride": 128}, STORE)
                assert export["samples"] == len(requested)
                save(receipt_name, export)
            rows = read_csv_records(MANIFEST)
            check_baseline(rows, baseline)
            _, low, marked = filter_new_rows(rows, requested, set(baseline), "0.05", "0.80")
            write_csv_records(REPORT / (BATCH + "_group_" + str(index+1) + "_before_filter.csv"), rows)
            with STORE.transaction():
                # Detect concurrent web review updates before publishing.
                assert read_csv_records(MANIFEST) == rows, "Concurrent review change; resume filtering with fresh metadata"
                publish_filtered(marked)
            verify_legacy()
            summary = scoped_summary(marked, requested, baseline)
            assert summary["patches"] == export["generated_patches"]
            assert all(not patch_is_included(row) or not exclusion_reasons(row) for row in marked if row.get("sample_id") in requested)
            state["groups"].append({"index": index, "status": "completed", "samples": len(requested),
                                    "sites": sites, **summary})
            checkpoint("group_completed", completed_sites=sum(len(group["sites"]) for group in state["groups"]))
        checkpoint("building_training_files", completed_sites=50)
        dataset = run_dataset_build("chongqing", {"workspace_id": "default", "config_id": "resize256_v1"}, STORE)
        status = workspace_training_dataset_status(REGION, "resize256_v1", STORE, "default")
        assert status["ready"], "Dataset not ready"
        actual = read_csv_records(resolve_data_path(dataset["manifest"], REGION))
        assert len(actual) == dataset["patches"] and all(resolve_data_path(row["npz_path"], REGION).is_file() for row in actual)
        rows = read_csv_records(MANIFEST)
        check_baseline(rows, baseline)
        verify_legacy()
        new = [row for row in rows if row["logical_patch_id"] not in baseline]
        assert all(row.get("sample_id") in sample_ids and row.get("site_id") in plan["site_ids"] for row in new)
        assert all(row.get("label_sources") == "spectral_osm_intersection" for row in new)
        assert {row.get("sample_id") for row in new} == sample_ids
        assert all(not patch_is_included(row) or not exclusion_reasons(row) for row in new)
        assert all(resolve_data_path(row["preview_path"], REGION).is_file() for row in new)
        assert len(actual) == sum(map(patch_is_included, rows))
        assert signatures(rows + selected) == load(REPORT / (BATCH + "_annotation_signatures.json"))
        assert hashlib.sha256(SAMPLES.read_bytes()).hexdigest() == load(REPORT / (BATCH + "_samples_signature.json"))["sha256"]
        per_site = []
        for site in plan["site_ids"]:
            ids = {row["sample_id"] for row in selected if row["site_id"] == site}
            summary = scoped_summary(rows, ids, baseline)
            per_site.append({"site_id": site, "registered_periods": 2, "exported_periods": len(ids), **summary})
        write_csv_records(REPORT / (BATCH + "_site_counts.csv"), [{key: value for key, value in item.items() if key != "site_ids"} for item in per_site])
        audit = {"existing_rows_unchanged": True, "preserved_existing_patches": len(baseline),
                 "annotation_signatures_unchanged": True, "sample_manifest_unchanged": True,
                 "all_threshold_failures_excluded": True, "all_new_previews_exist": True,
                 "all_new_rows_retained": True, "preserved_legacy_patches": 690, "training_started": False}
        save(BATCH + "_audit.json", audit)
        checkpoint("awaiting_human_review", verified=True, total_patches=len(rows),
                   included_total=sum(map(patch_is_included, rows)), excluded_total=sum(not patch_is_included(row) for row in rows),
                   new_patches=len(new), new_included=sum(map(patch_is_included, new)),
                   new_excluded=sum(not patch_is_included(row) for row in new),
                   below_threshold=sum(bool(exclusion_reasons(row)) for row in new),
                   exported_periods=len(sample_ids), registered_periods=plan["registered_periods"], site_count=len(plan["site_ids"]),
                   per_site=per_site, training_files=len(actual), dataset=dataset, dataset_status=status, audit=audit)
    except BaseException as error:
        checkpoint("failed", error=str(error))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if args.mode == "prepare" else run()
