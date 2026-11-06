"""Prepare the authorized ten-site batch using the website's first three images."""
from pathlib import Path
import csv, fcntl, hashlib, json, time, urllib.request
from collections import defaultdict
from lake_workbench.regions.config import load_region_configs
from lake_workbench.training.patch_processing import training_label_fingerprint
from lake_workbench.utils import resolve_data_path
from lake_workbench.workspaces.membership import patch_is_included

ROOT = Path("/home/lake/lakes")
REPORT = ROOT / "reports/chongqing_five_site_batches"
EXPECTED = ["chongqing_22123", "chongqing_22147", "chongqing_22151", "chongqing_22158",
            "chongqing_22160", "chongqing_22162", "chongqing_22172", "chongqing_22180",
            "chongqing_22185", "chongqing_22197"]
REGION = load_region_configs()[0]["chongqing"]


def get(suffix):
    with urllib.request.urlopen("http://127.0.0.1:18765/api/" + suffix, timeout=60) as response:
        return json.load(response)


def main():
    with (REPORT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = REPORT / "batch_004_plan.json"
        assert not destination.exists(), "Existing batch plan must be inspected before resubmitting"
        seen = set()
        for file in REPORT.glob("batch_*_plan.json"):
            seen.update(json.loads(file.read_text())["site_ids"])
        samples_path = ROOT / "data/workspaces/default/samples/chongqing/manifest.csv"
        samples = list(csv.DictReader(samples_path.open()))
        groups = defaultdict(list)
        for row in samples:
            groups[row["site_id"]].append(row)
        sites = [site for site in sorted(groups) if site not in seen][:10]
        assert sites == EXPECTED
        chosen, periods = [], {}
        for site in sites:
            assets = get("regions/chongqing/sites/" + site + "/imagery")["assets"][:3]
            assert len(assets) == 3 and not any(asset.get("missing") for asset in assets)
            periods[site] = []
            for asset in assets:
                matched = [row for row in groups[site] if row["product_name"] == asset["product"]]
                assert len(matched) == 1
                row = matched[0]
                chosen.append(row)
                periods[site].append({"sample_id": row["sample_id"], "product": asset["product"],
                                      "date": asset["date"], "asset_id": asset["asset_id"]})
            sorted_samples = sorted(groups[site], key=lambda row: (row["product_date"], row["product_name"]), reverse=True)[:3]
            assert [row["sample_id"] for row in sorted_samples] == [period["sample_id"] for period in periods[site]]
        runs = get("workspaces/default/regions/chongqing/training-runs?dataset_config_id=resize256_v1&dataset_source=workspace")
        assert not any(row.get("status") in {"queued", "running", "starting"} for row in runs.get("items", []))
        manifest = ROOT / "data/workspaces/default/logical_patches/chongqing/manifest.csv"
        current = list(csv.DictReader(manifest.open()))
        sample_labels = {resolve_data_path(row["label_path"], REGION) for row in current + chosen if row.get("label_path")}
        signatures = {str(path): training_label_fingerprint(path) for path in sample_labels if path.is_file()}
        signatures[str(samples_path)] = hashlib.sha256(samples_path.read_bytes()).hexdigest()
        (REPORT / "batch_004_annotation_signatures.json").write_text(json.dumps(signatures, indent=2))
        prior = REPORT / "batch_003_report.json"
        state = json.loads(prior.read_text())
        assert state["status"] == "awaiting_human_review"
        (REPORT / "batch_003_report.before_batch_004.json").write_bytes(prior.read_bytes())
        reviewed_samples = set(state["sample_ids"])
        state.update(status="human_review_completed", human_review_completed=True,
                     human_review_completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     reviewed_included_patches=sum(patch_is_included(row) for row in current if row.get("sample_id") in reviewed_samples))
        prior.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        plan = {"batch_id": "batch_004", "site_ids": sites, "periods_per_site": 3,
                "period_selection": "website_first_three_newest",
                "registered_periods": len(chosen), "period_counts": {site: 3 for site in sites},
                "available_period_counts": {site: len(groups[site]) for site in sites},
                "selected_periods": periods, "sample_ids": [row["sample_id"] for row in chosen],
                "period_status_counts": {"sample_saved": len(chosen)}, "source": "spectral_osm_intersection",
                "max_dimension": 2048, "minimum_water_ratio": "0.05", "minimum_valid_ratio": "0.80",
                "patch_size": 256, "stride": 128, "wait_for_human_review": True,
                "low_quality_action": "retain_as_excluded", "training_started": False}
        destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
        print(json.dumps({"plan": str(destination), "sites": sites, "selected_periods": periods,
                          "period_count": len(chosen), "existing_patches": len(current),
                          "existing_included": sum(map(patch_is_included, current)),
                          "annotation_signatures": len(signatures)}), flush=True)


if __name__ == "__main__":
    main()

