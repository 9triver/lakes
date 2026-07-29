"""Training metadata and safe model weight discovery."""

from pathlib import Path

from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.training.datasets import dataset_summary_from_config, read_json_file, timestamp_for_path
from lake_workbench.utils import clean_optional, display_path, parse_float, parse_int_or_default


GLOBAL_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "all"


def model_training_metadata(model_path: Path, scope: str) -> dict:
    run_dir = model_path.parent
    config = read_json_file(run_dir / "config.json", {})
    if not isinstance(config, dict):
        config = {}
    history = read_json_file(run_dir / "history.json", [])
    if not isinstance(history, list):
        history = []
    best_iou = None
    best_epoch = None
    for record in history:
        val = record.get("val") or {}
        train = record.get("train") or {}
        score = parse_float(val.get("iou"))
        if score is None:
            score = parse_float(train.get("iou"))
        if score is not None and (best_iou is None or score > best_iou):
            best_iou = score
            best_epoch = parse_int_or_default(record.get("epoch"), 0)
    return {
        "run_name": run_dir.name,
        "scope": clean_optional(config.get("scope")) or scope,
        "config": {
            key: config.get(key)
            for key in (
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "base_channels",
                "model_type",
                "model_options",
                "architecture_label",
                "val_ratio",
                "seed",
                "threshold",
                "no_augment",
                "augmentation_enabled",
                "device",
                "device_requested",
                "train_count",
                "val_count",
                "train_site_count",
                "val_site_count",
                "split_group",
                "manifest",
                "profile_id",
            )
            if key in config
        },
        "dataset": dataset_summary_from_config(config) if config else {},
        "history_count": len(history),
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "latest": history[-1] if history else {},
        "updated_at": timestamp_for_path(run_dir / "history.json") or timestamp_for_path(model_path),
    }


def model_sort_key(item: dict) -> tuple:
    score = parse_float(item.get("best_iou"))
    return (
        1 if item.get("error") else 0,
        1 if score is None else 0,
        -(score or 0),
        0 if item.get("weight") == "best.pt" else 1,
        -(parse_int_or_default(item.get("epoch"), 0)),
        item.get("label", ""),
    )


def iter_global_model_paths() -> list[Path]:
    if not GLOBAL_MODEL_DIR.exists():
        return []
    return sorted(GLOBAL_MODEL_DIR.glob("*/*.pt"))


def global_model_key(path: Path) -> str:
    try:
        return f"all/{path.resolve().relative_to(GLOBAL_MODEL_DIR.resolve())}"
    except ValueError:
        return f"all/{path.name}"


def global_model_path_from_key(model_key: str) -> Path:
    key = clean_optional(model_key) or ""
    path = Path(key)
    if path.is_absolute():
        raise ValueError("absolute model paths are not allowed")
    parts = path.parts
    if len(parts) == 3 and parts[0] == "all":
        run_name, weight = parts[1], parts[2]
    elif len(parts) == 2:
        run_name, weight = parts
    else:
        raise ValueError(f"invalid model key: {key}")
    if run_name in {"", ".", ".."} or weight in {"", ".", ".."}:
        raise ValueError(f"invalid model key: {key}")
    return GLOBAL_MODEL_DIR / run_name / weight


def persisted_training_job(scope: str, run_dir: Path) -> dict | None:
    config = read_json_file(run_dir / "config.json", {})
    if not isinstance(config, dict) or not config:
        return None
    history = read_json_file(run_dir / "history.json", [])
    if not isinstance(history, list):
        history = []
    completed = (run_dir / "best.pt").exists() or (run_dir / "last.pt").exists() or bool(history)
    status = "completed" if completed else "failed"
    result = {
        "status": status,
        "output_dir": display_path(run_dir),
        "manifest": config.get("manifest") or display_path(run_dir / "manifest.csv"),
        "best_model": display_path(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
        "last_model": display_path(run_dir / "last.pt") if (run_dir / "last.pt").exists() else "",
        "history": history,
        "config": config,
        "profile_id": clean_optional(config.get("profile_id")) or "",
    }
    if history:
        result["best_iou"] = max((parse_float((record.get("val") or {}).get("iou")) or 0 for record in history), default=0)
    epoch = parse_int_or_default((history[-1] if history else {}).get("epoch"), 0)
    epochs = parse_int_or_default(config.get("epochs"), epoch)
    config_path = run_dir / "config.json"
    return {
        "job_id": run_dir.name,
        "scope": scope,
        "run_name": run_dir.name,
        "status": status,
        "message": "历史训练任务" if completed else "训练未完成（服务重启或启动失败）",
        "progress": 100 if status == "completed" else 5,
        "epoch": epoch,
        "epochs": epochs,
        "history": history,
        "config": config,
        "dataset": dataset_summary_from_config(config),
        "result": result,
        "output_dir": display_path(run_dir),
        "created_at": timestamp_for_path(config_path),
        "updated_at": timestamp_for_path(run_dir / "history.json") or timestamp_for_path(config_path),
        "persisted": True,
    }
