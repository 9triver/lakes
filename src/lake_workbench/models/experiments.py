"""Lightweight discovery of local experimental checkpoints, without loading networks on GPU."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from functools import lru_cache
from pathlib import Path

from lake_workbench.models.runtime import architecture_label, model_options
from lake_workbench.paths import PROJECT_ROOT


@lru_cache(maxsize=256)
def _checkpoint_description(path_string: str, mtime: int, size: int) -> dict:
    import torch

    path = Path(path_string)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    normalization = checkpoint.get("normalization") or config.get("normalization") or {}
    channels = int(config.get("in_channels") or len(normalization.get("mean") or []))
    kind = config.get("model_type", "unet")
    options = model_options(kind, config)
    epoch = int(checkpoint.get("epoch") or 0)
    history = checkpoint.get("history") or []
    record = next((r for r in history if r.get("epoch") == epoch), {})
    validation = record.get("validation") or record.get("val") or {}
    return dict(epoch=epoch, in_channels=channels, base_channels=options.get("base_channels"),
                model_type=kind, model_options=options, architecture_label=architecture_label(kind, channels, options),
                selected_validation_iou=validation.get("iou"), selected_validation_loss=validation.get("loss"),
                config=config)


def checkpoint_description(path: Path) -> dict:
    stat = path.stat()
    return _checkpoint_description(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def input_variant(config: dict) -> str:
    bands=config.get('input_bands') or config.get('input_features') or []
    if bands:
        if 'slope_degrees' in bands:
            return '五光谱+坡度' if 'swir1' in bands else 'RGB-NIR-SLOPE'
        return '五光谱'
    if config.get('in_channels')==6:
        return '五光谱+坡度'
    if config.get('input_indices')==[2,1,0,3,5]:
        return 'RGB-NIR-SLOPE'
    return '五光谱'


_DISCOVERY_LOCK=threading.Lock()
_DISCOVERY_CACHE={}


def experimental_models(workspace_id: str) -> list[dict]:
    # Concurrent page queries share a short-lived snapshot. New weights appear
    # on the next request after expiry; checkpoint loading still validates files.
    key=(str(PROJECT_ROOT.resolve()),workspace_id)
    with _DISCOVERY_LOCK:
        cached=_DISCOVERY_CACHE.get(key)
        if cached and time.monotonic()<cached[0]:return cached[1]
        items=_discover_experimental_models(workspace_id)
        if len(_DISCOVERY_CACHE)>=32:_DISCOVERY_CACHE.pop(next(iter(_DISCOVERY_CACHE)))
        _DISCOVERY_CACHE[key]=(time.monotonic()+15,items)
        return items


def _discover_experimental_models(workspace_id: str) -> list[dict]:
    items = []
    for path in sorted((PROJECT_ROOT / "outputs").rglob("best*.pt")):
        if path.name not in {"best.pt", "best_loss.pt"} or any("interrupted" in p for p in path.parts):
            continue
        config_path = path.parent / "config.json"
        history_path = path.parent / "history.json"
        if not config_path.exists() or not history_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            if config.get("workspace_id", "default") != workspace_id:
                continue
            history = json.loads(history_path.read_text())
            validation_scores=[(r.get('validation') or r.get('val') or {}).get('iou') for r in history]
            max_validation_iou=max((v for v in validation_scores if isinstance(v,(int,float)) and math.isfinite(v)),default=None)
            relative = path.relative_to(PROJECT_ROOT / "outputs")
            identity = hashlib.sha256(str(relative).encode()).hexdigest()[:24]
            completed = len(history) >= int(config.get("epochs", 1))
            selection = "最低验证损失" if path.name == "best_loss.pt" else "最高验证IoU"
            scope = config.get("region") or ("shaanxi" if "shaanxi" in relative.parts[0] else "guizhou")
            arm = {"A_site_random": "A", "B_overlap_grouped": "B"}.get(config.get("arm"), "")
            variant = input_variant(config)
            fold = config.get("fold")
            fold_name = f"fold{fold.get('fold')}" if isinstance(fold, dict) else ""
            paper_name={'aer_unet_adapted':'AER U-Net（论文结构适配版）','resddsc_unetpp_adapted':'ResDDSCUNet++（论文结构适配版）'}.get(config.get('model_type'))
            loss_name={'weighted_bce':'加权 BCE','weighted_bce_hard_negative':'加权 BCE+OHEM'}.get(config.get('pilot_arm'))
            label = " · ".join(str(v) for v in [loss_name, relative.parts[0], config.get('encoder') or paper_name, variant, arm, fold_name,
                    f"seed{config.get('seed',42)}", f"lr{config.get('lr')}", selection] if v)
            item = dict(key=f"experiments/{workspace_id}/{identity}/{path.name}",label=label,
                        name=label,weight=path.name,path=str(path.relative_to(PROJECT_ROOT)),
                        workspace_id=workspace_id,scope=scope,region=scope,experimental=True,
                        completed=completed,history_count=len(history),selection=selection,
                        max_validation_iou=max_validation_iou,
                        updated_at=path.stat().st_mtime,config=config)
            try:
                description = checkpoint_description(path)
                item.update(description)
                item.update(best_iou=description["selected_validation_iou"], best_epoch=description["epoch"])
                score=description['selected_validation_iou']
                if score is not None:
                    label+=f' {score:.2%}'
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            if not completed:
                label += f" · 未完成({len(history)}/{config.get('epochs')})"
            item.update(label=label,name=label)
            items.append(item)
        except (OSError, ValueError, TypeError):
            continue
    return items
