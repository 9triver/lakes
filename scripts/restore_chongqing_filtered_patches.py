"""Recover only previously filtered rows from the first ten Chongqing sites."""
from __future__ import annotations
import fcntl
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

from chongqing_five_site_batches import (
    MANIFEST, REGION, REPORT, ROOT, STORE, exclusion_reasons, legacy_ids,
    load, merge_filtered_rows, patch_is_included, read_csv_records,
    resolve_data_path, save, verify_legacy, write_csv_records,
)
from lake_workbench.training.patch_processing import training_label_fingerprint

EXPECTED_SITES = {
    "chongqing_21969", "chongqing_21973", "chongqing_22017", "chongqing_22028", "chongqing_22030",
    "chongqing_22052", "chongqing_22053", "chongqing_22061", "chongqing_22085", "chongqing_22086",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scope():
    sites, samples, snapshots = set(), set(), []
    previous = legacy_ids()
    for batch in ("batch_001", "batch_002"):
        plan = load(REPORT / (batch + "_plan.json"))
        sites.update(plan["site_ids"])
        selected = set(plan["sample_ids"])
        samples.update(selected)
        rows = read_csv_records(REPORT / (batch + "_before_water_filter.csv"))
        scoped = [row for row in rows if row.get("sample_id") in selected
                  and row["logical_patch_id"] not in previous]
        assert len(scoped) == (6980 if batch == "batch_001" else 7468)
        snapshots.extend(scoped)
    assert sites == EXPECTED_SITES and len(samples) == 254
    assert len(snapshots) == len({row["logical_patch_id"] for row in snapshots}) == 14448
    return sites, samples, snapshots, previous


def annotation_signatures(rows):
    paths = {resolve_data_path(row["label_path"], REGION) for row in rows if row.get("label_path")}
    result = {}
    for path in paths:
        if path.is_file():
            result[str(path)] = training_label_fingerprint(path)
    sample_manifest = STORE.ensure_workspace_training_samples("default", "chongqing")
    result[str(sample_manifest)] = sha(sample_manifest)
    return result


def stage_previews(restored, archive):
    copied, checked = 0, 0
    active_dir = MANIFEST.parent.resolve()
    for row in restored:
        for field in ("preview_path", "preview_base_path"):
            if not row.get(field):
                continue
            target = resolve_data_path(row[field], REGION)
            assert target.resolve().is_relative_to(active_dir)
            if not target.is_file():
                source = archive / "preview" / target.name
                assert source.is_file(), "Missing archived preview: " + str(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                assert sha(source) == sha(target)
                copied += 1
            checked += 1
    return {"copied": copied, "checked": checked}


def assert_no_training():
    url = ("http://127.0.0.1:18765/api/workspaces/default/regions/chongqing/training-runs"
           "?dataset_config_id=resize256_v1&dataset_source=workspace")
    with urllib.request.urlopen(url, timeout=30) as response:
        result = json.load(response)
    active = [item for item in result.get("items", [])
              if item.get("status") in {"queued", "running", "starting", "cancelling"}]
    assert not active, "Active training must not be interrupted"
    return len(result.get("items", []))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sites, samples, snapshots, previous = load_scope()
        rows = read_csv_records(MANIFEST)
        _, restored = merge_filtered_rows(rows, snapshots, sites, samples, previous)
        archive = Path(load(REPORT / "initialization.json")["archive"])
        previews = stage_previews(restored, archive)
        print(json.dumps({"phase": "previews_ready", "restore_candidates": len(restored), **previews}), flush=True)
        assert_no_training()
        assert subprocess.run(["systemctl", "--user", "is-active", "--quiet", "lakes.service"]).returncode == 0
        backup_dir = REPORT / ("restore_first_ten_" + time.strftime("%Y%m%d_%H%M%S"))
        backup_dir.mkdir()
        stopped, committed = False, False
        result = {"status": "failed", "training_started": False}
        try:
            subprocess.run(["systemctl", "--user", "stop", "lakes.service"], check=True, timeout=60)
            stopped = True
            assert subprocess.run(["systemctl", "--user", "is-active", "--quiet", "lakes.service"]).returncode != 0
            # Take the authoritative state only after web requests can no longer write.
            current = read_csv_records(MANIFEST)
            before = {row["logical_patch_id"]: row for row in current}
            assert previous.issubset(before) and len(previous) == 690
            verify_legacy()
            labels_before = annotation_signatures(snapshots)
            included_before = {key for key, row in before.items() if patch_is_included(row)}
            merged, restored = merge_filtered_rows(current, snapshots, sites, samples, previous)
            assert all(not patch_is_included(row) for row in restored)
            backup = backup_dir / "manifest.before.csv"
            shutil.copy2(MANIFEST, backup)
            backup_sha = sha(backup)
            (backup_dir / "annotation_signatures.before.json").write_text(
                json.dumps(labels_before, indent=2), encoding="utf-8")
            temp = MANIFEST.with_suffix(".restore.tmp")
            write_csv_records(temp, merged)
            staged = read_csv_records(temp)
            staged_index = {row["logical_patch_id"]: row for row in staged}
            assert len(staged_index) == len(staged)
            assert all(staged_index[key] == row for key, row in before.items()), "An existing record would change"
            assert sha(MANIFEST) == backup_sha
            temp.replace(MANIFEST)
            committed = True
            after = read_csv_records(MANIFEST)
            index = {row["logical_patch_id"]: row for row in after}
            assert all(index[key] == row for key, row in before.items())
            assert {key for key, row in index.items() if patch_is_included(row)} == included_before
            assert all(not patch_is_included(index[row["logical_patch_id"]]) for row in restored)
            assert all(row["logical_patch_id"] in index for row in snapshots)
            assert len(after) == 15138
            labels_after = annotation_signatures(snapshots)
            assert labels_before == labels_after, "Annotation content changed"
            verify_legacy()
            result = {
                "status": "completed", "verified": True,
                "site_ids": sorted(sites), "sample_count": len(samples),
                "restored_as_excluded": len(restored),
                "restored_by_site": dict(Counter(row["site_id"] for row in restored)),
                "total": len(after), "included": len(included_before),
                "excluded": len(after) - len(included_before),
                "preserved_existing_rows": len(before), "preserved_legacy_patches": len(previous),
                "annotations_unchanged": True, "annotation_signatures_checked": len(labels_after),
                "existing_rows_unchanged": True, "included_ids_unchanged": True,
                "minimum_water_ratio": "0.05", "minimum_valid_ratio": "0.80",
                "future_low_quality_action": "retain_as_excluded",
                "preview_files": previews, "backup": str(backup),
                "before_sha256": backup_sha, "after_sha256": sha(MANIFEST),
                "training_started": False,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            save("restore_first_ten_report.json", result)
            (backup_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except BaseException as exc:
            if committed:
                rollback = MANIFEST.with_suffix(".rollback.tmp")
                shutil.copy2(backup_dir / "manifest.before.csv", rollback)
                rollback.replace(MANIFEST)
            result.update(error=str(exc), rolled_back=committed)
            save("restore_first_ten_failed.json", result)
            raise
        finally:
            if stopped:
                subprocess.run(["systemctl", "--user", "start", "lakes.service"], check=True, timeout=60)


if __name__ == "__main__":
    main()

