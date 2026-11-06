"""Inspect and generate Chongqing high-confidence seed samples through Lakes API."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import quote

import requests

BASE = "https://lakes.njuics.cn"
PREFIX = "/api/workspaces/default/regions/chongqing"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "chongqing_seed_all_20261013"
SOURCE = "spectral_osm_intersection"
_LOCAL = threading.local()
_WRITE_LOCK = threading.Lock()
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504, 520, 522, 524}


class APIError(RuntimeError):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


def request(path: str, payload=None, timeout=180, attempts=4):
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
        _LOCAL.session.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0 LakesDatasetClient/1.0"})
    for attempt in range(attempts):
        try:
            response = _LOCAL.session.request("GET" if payload is None else "POST", BASE + path, json=payload, timeout=(20, timeout))
            if response.status_code in TRANSIENT_HTTP_STATUSES and attempt < attempts - 1:
                time.sleep(min(15, 2 ** attempt * 2))
                continue
            if not response.ok:
                try:
                    message = response.json().get("error") or response.json().get("message")
                except ValueError:
                    message = response.text[:200]
                raise APIError(response.status_code, message)
            return response.json()
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(min(15, 2 ** attempt * 2))
    raise RuntimeError("request attempts exhausted")


def save(name, value):
    path = REPORT_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(value):
    print(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **value}, ensure_ascii=False), flush=True)


def append_result(value):
    with _WRITE_LOCK:
        with (REPORT_DIR / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def plan():
    plan_path = REPORT_DIR / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    sites = request(PREFIX + "/sites?limit=500")["items"]
    assert len(sites) == 290, f"unexpected region inventory: {len(sites)}"
    tasks = []
    errors = []

    def imagery(site):
        site_id = site["site_id"]
        path = REPORT_DIR / "imagery" / (site_id + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = request(PREFIX + "/sites/" + quote(site_id) + "/imagery")
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        unique = {}
        for asset in value.get("assets", []):
            key = asset.get("asset_id") or asset.get("product")
            if key:
                unique[key] = {"site_id": site_id, "extent": site["bbox"], "asset": asset}
        return list(unique.values())

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(imagery, site): site["site_id"] for site in sites}
        for future in as_completed(futures):
            try:
                tasks.extend(future.result())
            except Exception as exc:
                errors.append({"site_id": futures[future], "error": str(exc)})
            if (len(futures) - sum(not item.done() for item in futures)) % 25 == 0:
                log({"stage": "inventory", "assets": len(tasks), "errors": len(errors)})
    tasks.sort(key=lambda task: (task["site_id"], task["asset"].get("date", ""), task["asset"].get("asset_id", "")))
    result = {"sites": len(sites), "registered_images": sum(site["image_count"] for site in sites), "tasks": tasks, "inventory_errors": errors}
    save("plan.json", result)
    return result


def generate_one(task):
    asset = task["asset"]
    asset_id = asset.get("asset_id") or asset["product"]
    row = {"site_id": task["site_id"], "asset_id": asset_id, "date": asset.get("date", "")}
    started = time.monotonic()
    try:
        if asset.get("missing") or not asset.get("downloaded"):
            return {**row, "status": "missing_imagery", "reason": asset.get("missing_reason") or "not downloaded", "seconds": round(time.monotonic() - started, 2)}
        path = PREFIX + "/sites/" + quote(task["site_id"])
        generated = request(path + "/generated-labels/" + SOURCE, {
            "asset_id": asset_id, "product": asset["product"], "extent": task["extent"], "max_dimension": 2048,
        })
        properties = generated["label"]["properties"]
        actual_imagery = properties.get("imagery", {})
        assert (actual_imagery.get("product_id") or actual_imagery.get("product_name")) == asset_id, "selected imagery mismatch"
        assert generated["source"] == SOURCE, "source mismatch"
        row.update({"label_id": generated["label_id"], "stats": generated["stats"], "grid": properties.get("grid"), "details": generated["details"]})
        if not generated["label"].get("features") or not generated["stats"].get("water_pixels"):
            return {**row, "status": "no_seed_water", "seconds": round(time.monotonic() - started, 2)}
        captured = {
            "region": "chongqing", "site_id": task["site_id"],
            "map": {"extent": properties["extent"]},
            "visible_layers": {SOURCE: True},
            "selected_generated_labels": {SOURCE: {"id": generated["label_id"]}},
            "selected_tile": asset.get("tile", ""), "selected_product": asset["product"],
            "selected_imagery_asset_id": asset_id, "model_prediction_excluded": True,
            "jrc_threshold": 75,
        }
        sample = request(path + "/training-samples", {
            "label_source": "current_view", "label_threshold": "75", "buffer_ratio": 0.8,
            "auto_patch": False, "view_state": captured,
        })
        row.update({"sample_id": sample["sample"]["sample_id"], "duplicate": sample["sample"].get("duplicate", False), "status": "sample_saved", "seconds": round(time.monotonic() - started, 2)})
    except Exception as exc:
        row.update({"status": "failed", "error": str(exc), "seconds": round(time.monotonic() - started, 2)})
    return row


def latest_results():
    path = REPORT_DIR / "results.jsonl"
    if not path.exists():
        return {}
    return {row["asset_id"]: row for line in path.read_text(encoding="utf-8").splitlines() if line.strip() for row in [json.loads(line)]}


def stage_progress(stage, **details):
    results = latest_results()
    save("progress.json", {
        "stage": stage,
        "processed": len(results),
        "planned": len(plan()["tasks"]),
        "counts": dict(Counter(row["status"] for row in results.values())),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **details,
    })


def wait_for_response(path, stage, timeout=130):
    # Export/build hold the server's Workspace transaction lock. Authentication
    # and job-status GETs can time out until that transaction finishes. Keep
    # waiting for the existing job; never submit another POST on a poll failure.
    while True:
        try:
            return request(path, timeout=timeout, attempts=1)
        except APIError as exc:
            if exc.status_code not in TRANSIENT_HTTP_STATUSES:
                raise
            error = str(exc)
        except requests.RequestException as exc:
            error = str(exc)
        stage_progress(stage, polling_error=error)
        log({"stage": stage, "polling_error": error, "retry_seconds": 60})
        time.sleep(60)


def export_patches():
    job_file = REPORT_DIR / "export_job.json"
    if job_file.exists():
        job = json.loads(job_file.read_text(encoding="utf-8"))
    else:
        # Match the website's TrainingCapture defaults, but export only once.
        job = request(PREFIX + "/training-patches/export-jobs", {"patch_size": 256, "stride": 128, "preview_scale": 2, "overwrite": False})
        save("export_job.json", job)
    while job["status"] not in {"completed", "failed"}:
        stage_progress("export", job=job)
        log({"stage": "export", "job_id": job["job_id"], "status": job["status"], "message": job.get("message")})
        time.sleep(30)
        job = wait_for_response(PREFIX + "/training-patches/export-jobs/" + job["job_id"], "export")
        save("export_job.json", job)
    if job["status"] != "completed":
        raise RuntimeError("Patch export failed: " + json.dumps(job, ensure_ascii=False))
    return job


def build_dataset():
    job_file = REPORT_DIR / "dataset_job.json"
    if job_file.exists():
        job = json.loads(job_file.read_text(encoding="utf-8"))
    else:
        job = request(PREFIX + "/training-datasets/resize256_v1/build-jobs", {})
        save("dataset_job.json", job)
    while job["status"] not in {"completed", "failed"}:
        stage_progress("dataset", job=job)
        log({"stage": "dataset", "job_id": job["job_id"], "status": job["status"], "message": job.get("message")})
        time.sleep(30)
        job = wait_for_response(PREFIX + "/training-datasets/resize256_v1/build-jobs/" + job["job_id"], "dataset")
        save("dataset_job.json", job)
    if job["status"] != "completed":
        raise RuntimeError("Dataset build failed: " + json.dumps(job, ensure_ascii=False))
    return job


def wait_for_water_filter(export):
    policy_path = REPORT_DIR / "water_filter_policy.json"
    if not policy_path.exists():
        return None
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    result_path = REPORT_DIR / "water_filter_result.json"
    while True:
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "failed":
                raise RuntimeError("Water filter failed: " + str(result.get("error")))
            if result.get("status") == "completed":
                assert result.get("job_id") == export["job_id"], "filter export job mismatch"
                assert result.get("minimum_water_ratio") == policy["minimum_water_ratio"], "filter threshold mismatch"
                assert result.get("verified") is True, "filter not verified"
                assert result.get("remaining_below_threshold_included") == 0, "low-water patches remain included"
                assert result.get("preserved_existing_patches") == len(policy["previous_patch_ids"]), "existing patch preservation not verified"
                stage_progress("water_filter_completed", water_filter=result)
                return result
        stage_progress("water_filter", job_id=export["job_id"], minimum_water_ratio=policy["minimum_water_ratio"])
        log({"stage": "water_filter", "message": "waiting for verified server filtering result"})
        time.sleep(30)


def verified_patch_summary(export, water_filter, previous):
    assert previous["total"] == water_filter["preserved_existing_patches"], "existing patch inventory mismatch"
    assert export["result"]["generated_patches"] == water_filter["verified_new_patches"], "new patch inventory mismatch"
    total = previous["total"] + water_filter["verified_new_patches"]
    assert total == export["result"]["patches"], "export patch total mismatch"
    included = previous["included_count"] + water_filter["included_new_patches_after_filter"]
    return {"total": total, "included_count": included, "excluded_count": total - included,
            "verification_source": "verified_server_manifest", "items_omitted": True}


def generate():
    if (REPORT_DIR / "superseded_by_five_site_batches.json").exists():
        raise RuntimeError("Full-region generation was cancelled; use the five-site batch workflow and wait for human review between batches")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    session = wait_for_response("/api/auth/session", "authentication")
    assert session["user"]["default_workspace_id"] == "default", "workspace mismatch"
    if not (REPORT_DIR / "training_runs_before.json").exists():
        save("training_runs_before.json", request(PREFIX + "/training-runs?dataset_config_id=resize256_v1&dataset_source=workspace"))
    inventory = plan()
    if inventory["inventory_errors"]:
        raise RuntimeError("Inventory incomplete; inspect plan.json before continuing")
    log({"stage": "plan_ready", "sites": inventory["sites"], "assets": len(inventory["tasks"]), "registered_images": inventory["registered_images"]})
    done = latest_results()
    pending = [task for task in inventory["tasks"] if (done.get(task["asset"].get("asset_id") or task["asset"]["product"]) or {}).get("status") not in {"sample_saved", "no_seed_water", "missing_imagery"}]
    if not done and pending:
        result = generate_one(pending.pop(0))
        append_result(result)
        log({"stage": "pilot", **result})
        if result["status"] == "failed":
            raise RuntimeError("Pilot failed; inspect results.jsonl")
    # The server controls CPU concurrency with its derived-label semaphore.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(generate_one, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_result(result)
            if index % 10 == 0 or result["status"] == "failed":
                current = latest_results()
                counts = Counter(row["status"] for row in current.values())
                save("progress.json", {"stage": "labels_and_samples", "processed": len(current), "planned": len(inventory["tasks"]), "counts": dict(counts), "last": result})
                log({"stage": "labels_and_samples", "processed": len(current), "planned": len(inventory["tasks"]), "counts": dict(counts), "last_asset": result["asset_id"], "last_seconds": result["seconds"]})
    results = latest_results()
    failed = [row for row in results.values() if row["status"] == "failed"]
    # Retry each unsuccessful asset once after the main pass; cache identities
    # and training fingerprints make this safe after a lost HTTP response.
    if failed:
        failed_ids = {row["asset_id"] for row in failed}
        for task in inventory["tasks"]:
            asset_id = task["asset"].get("asset_id") or task["asset"]["product"]
            if asset_id not in failed_ids:
                continue
            result = generate_one(task)
            append_result(result)
            log({"stage": "retry", "asset_id": asset_id, "status": result["status"]})
        results = latest_results()
        failed = [row for row in results.values() if row["status"] == "failed"]
    if failed:
        save("failed.json", failed)
        raise RuntimeError(f"{len(failed)} assets failed; rerun resumes these assets")
    stage_progress("labels_and_samples_completed")
    export = export_patches()
    water_filter = wait_for_water_filter(export)
    dataset = build_dataset()
    if water_filter is not None:
        previous = json.loads((REPORT_DIR / "patches_before.json").read_text(encoding="utf-8"))
        patches = verified_patch_summary(export, water_filter, previous)
    else:
        patches = request(PREFIX + "/logical-patches", timeout=300)
    samples = request(PREFIX + "/training-samples", timeout=300)
    statuses = request(PREFIX + "/training-datasets", timeout=300)
    runs_after = request(PREFIX + "/training-runs?dataset_config_id=resize256_v1&dataset_source=workspace", timeout=300)
    for name, value in [("patches_after.json", patches), ("samples_after.json", samples), ("datasets_after.json", statuses), ("training_runs_after.json", runs_after)]:
        save(name, value)
    runs_before = json.loads((REPORT_DIR / "training_runs_before.json").read_text(encoding="utf-8"))
    assert {row["job_id"] for row in runs_after.get("items", [])} == {row["job_id"] for row in runs_before.get("items", [])}, "training runs changed externally"
    summary = {"status": "completed", "sites": inventory["sites"], "planned_assets": len(inventory["tasks"]), "counts": dict(Counter(row["status"] for row in results.values())), "patches": patches["total"], "included_patches": patches["included_count"], "samples": samples["total"], "export": export, "dataset": dataset, "training_started": False, "manual_review": "pending"}
    if water_filter is not None:
        summary["water_filter"] = water_filter
    save("summary.json", summary)
    log(summary)


def launch_background():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # Inherit the approved network access and keep the worker independent of
    # the interactive tool session. Both output streams stay in the workspace.
    with (REPORT_DIR / "worker.log").open("a", encoding="utf-8") as output:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "generate", "--worker"],
            cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
    save("worker_state.json", {"status": "starting", "pid": worker.pid, "started_at": datetime.now().isoformat(timespec="seconds")})
    log({"stage": "background_started", "pid": worker.pid, "report_dir": str(REPORT_DIR)})


def worker_main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")}
    save("worker_state.json", state)
    try:
        generate()
    except BaseException as exc:
        save("worker_state.json", {**state, "status": "failed", "error": str(exc), "ended_at": datetime.now().isoformat(timespec="seconds")})
        raise
    save("worker_state.json", {**state, "status": "completed", "ended_at": datetime.now().isoformat(timespec="seconds")})


def inspect():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    session = request("/api/auth/session")
    user = session["user"]
    print(json.dumps({"user": user["id"], "workspace": user["default_workspace_id"]}, ensure_ascii=False), flush=True)
    for name, path in [
        ("sites", PREFIX + "/sites?limit=500"),
        ("samples_before", PREFIX + "/training-samples"),
        ("patches_before", PREFIX + "/logical-patches"),
        ("datasets_before", PREFIX + "/training-datasets"),
    ]:
        value = request(path)
        (REPORT_DIR / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        items = value.get("items", [])
        print(json.dumps({"resource": name, "total": value.get("total"), "included_count": value.get("included_count"), "first": items[:1]}, ensure_ascii=False), flush=True)
    sites = json.loads((REPORT_DIR / "sites.json").read_text(encoding="utf-8"))
    first = sites["items"][0]
    site_id = str(first.get("site_id") or first.get("id"))
    for suffix in ["", "/imagery"]:
        value = request(PREFIX + "/sites/" + site_id + suffix)
        filename = "example_site" if not suffix else "example_imagery"
        (REPORT_DIR / f"{filename}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"resource": filename, "value": value}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "generate"])
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.mode == "inspect":
        inspect()
    elif args.worker:
        worker_main()
    else:
        launch_background()
