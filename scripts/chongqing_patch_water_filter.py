"""Exclude low-water Chongqing patches after the existing export, before build.

Run on the server. Read the CSV for compact statistics; apply changes through
the running backend so its Workspace transaction serializes review updates.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def included(row):
    status = row.get("review_status")
    if status in {"included", "excluded"}:
        return status == "included"
    return str(row.get("include", "true")).lower() in {"true", "1", "yes"}


def below_threshold(row, threshold):
    # Compare integer pixel counts, avoiding rounded six-decimal CSV ratios.
    water, valid = int(row["water_pixels"]), int(row["valid_pixels"])
    if valid <= 0 or water < 0 or water > valid:
        raise ValueError("Invalid pixel counts: " + row.get("logical_patch_id", ""))
    return Decimal(water) < Decimal(str(threshold)) * valid


def review_state(row):
    return (included(row), row.get("exclude_reason", ""), row.get("patch_notes", ""))


def select_patches(rows, policy):
    previous = set(policy["previous_patch_ids"])
    samples = set(policy["sample_ids"])
    targets, preserved, sample_ids = [], {}, set()
    stats = {"new_patches": 0, "below_threshold": 0, "already_excluded_below_threshold": 0,
             "at_or_above_threshold": 0, "included_at_or_above_threshold": 0}
    for row in rows:
        patch_id = row["logical_patch_id"]
        if patch_id in previous:
            preserved[patch_id] = review_state(row)
            continue
        if row.get("sample_id") not in samples:
            continue
        stats["new_patches"] += 1
        sample_ids.add(row["sample_id"])
        if below_threshold(row, policy["minimum_water_ratio"]):
            stats["below_threshold"] += 1
            if included(row):
                targets.append(patch_id)
            else:
                stats["already_excluded_below_threshold"] += 1
        else:
            stats["at_or_above_threshold"] += 1
            stats["included_at_or_above_threshold"] += int(included(row))
    if set(preserved) != previous:
        raise RuntimeError("Existing patch inventory changed; stop before filtering")
    if not stats["new_patches"]:
        raise RuntimeError("Export published no patches for the requested samples")
    stats["sample_periods_with_patches"] = len(sample_ids)
    return targets, preserved, stats


def verify_patches(rows, policy, preserved):
    previous = set(policy["previous_patch_ids"])
    samples = set(policy["sample_ids"])
    remaining, seen_previous, new_count, included_count = 0, set(), 0, 0
    for row in rows:
        patch_id = row["logical_patch_id"]
        if patch_id in previous:
            seen_previous.add(patch_id)
            if review_state(row) != preserved[patch_id]:
                raise RuntimeError("Existing patch review changed: " + patch_id)
        elif row.get("sample_id") in samples:
            new_count += 1
            included_count += int(included(row))
            remaining += int(included(row) and below_threshold(row, policy["minimum_water_ratio"]))
    if seen_previous != previous or remaining:
        raise RuntimeError(f"Filter verification failed: {remaining} low-water patches remain included")
    return {"remaining_below_threshold_included": remaining,
            "preserved_existing_patches": len(seen_previous),
            "verified_new_patches": new_count, "included_new_patches_after_filter": included_count}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def completed_export_receipt(policy):
    receipt = policy.get("completed_export_receipt")
    if receipt is None:
        return None
    result = receipt.get("result", {})
    if (receipt.get("status") != "completed" or receipt.get("job_id") != policy["job_id"]
            or result.get("region") != "chongqing" or result.get("workspace_id") != "default"
            or result.get("samples") != len(set(policy["sample_ids"]))
            or not isinstance(result.get("generated_patches"), int) or result["generated_patches"] <= 0
            or result.get("patches") != result["generated_patches"] + len(set(policy["previous_patch_ids"]))):
        raise RuntimeError("Completed export receipt does not match the requested batch")
    return receipt


def api(policy, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(policy["api_base"] + path, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "PATCH")
    while True:
        try:
            with urllib.request.urlopen(request, timeout=1800 if body else 180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if body is not None or error.code not in {429, 500, 502, 503, 504, 520, 522, 524}:
                raise
        except (TimeoutError, urllib.error.URLError):
            if body is not None:
                # Do not replay an ambiguous membership change automatically.
                raise
        print(json.dumps({"stage": "api_retry", "path": path, "retry_seconds": 30}), flush=True)
        time.sleep(30)


def save(path, report):
    report = {**report, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def watch(policy_path):
    import fcntl
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    report_path = Path(policy["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {"status": "waiting_for_export", "pid": os.getpid(),
                  "job_id": policy["job_id"], "minimum_water_ratio": policy["minimum_water_ratio"]}
        try:
            save(report_path, report)
            receipt = completed_export_receipt(policy)
            session = api(policy, "/api/auth/session")
            if session["user"]["id"] != "default" or session["user"]["default_workspace_id"] != "default":
                raise RuntimeError("Unexpected user/workspace")
            prefix = "/api/workspaces/default/regions/chongqing"
            while receipt is None:
                job = api(policy, prefix + "/training-patches/export-jobs/" + policy["job_id"])
                if job["status"] == "failed":
                    raise RuntimeError("Export failed: " + str(job.get("error") or job.get("message")))
                if job["status"] == "completed":
                    break
                save(report_path, report)
                time.sleep(30)
            report["status"] = "selecting_low_water_patches"
            save(report_path, report)
            manifest = Path(policy["manifest_path"])
            targets, preserved, statistics = select_patches(read_rows(manifest), policy)
            if receipt is not None and statistics["new_patches"] != receipt["result"]["generated_patches"]:
                raise RuntimeError("Export receipt patch count does not match the published manifest")
            backup = report_path.parent / "manifest_before_water_filter.csv"
            if not backup.exists():
                shutil.copy2(manifest, backup)
            report.update(statistics)
            report.update(status="applying_filter", patches_to_exclude=len(targets), backup=str(backup))
            save(report_path, report)
            if targets:
                result = api(policy, prefix + "/logical-patches",
                    {"operation": "exclude", "logical_patch_ids": targets})
                report["newly_excluded"] = result["updated"]
            else:
                report["newly_excluded"] = 0
            report.update(verify_patches(read_rows(manifest), policy, preserved))
            if report["verified_new_patches"] != report["new_patches"]:
                raise RuntimeError("Patch inventory changed during filtering")
            report.update(status="completed", verified=True, training_started=False)
            save(report_path, report)
        except BaseException as error:
            save(report_path, {**report, "status": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    watch(parser.parse_args().policy)
