"""Manifest loading and numerical preparation for model training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lake_workbench.paths import PROJECT_ROOT


IGNORE_INDEX = 255


@dataclass
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def make_patch_dataset(torch_module, dataset_base, rows: list[dict], normalization: Normalization, augment: bool = False):
    """Create a torch Dataset without importing torch during module loading."""

    class PatchDataset(dataset_base):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            with np.load(resolve_project_path(row["npz_path"])) as data:
                image = data["image"].astype("float32")
                mask = data["mask"].astype("uint8")
                valid = data["valid"].astype("uint8")
            image = (image - normalization.mean[:, None, None]) / normalization.std[:, None, None]
            if augment:
                image, mask, valid = augment_patch(image, mask, valid)
            return {
                "image": torch_module.from_numpy(np.ascontiguousarray(image)),
                "mask": torch_module.from_numpy(np.ascontiguousarray(mask)).long(),
                "valid": torch_module.from_numpy(np.ascontiguousarray(valid)).bool(),
                "patch_id": row.get("patch_id", ""),
            }

    return PatchDataset()


def load_manifest_rows(path: Path) -> list[dict]:
    table = pd.read_csv(path, dtype=str).fillna("")
    rows = []
    for row in table.to_dict("records"):
        include = row.get("include") or row.get("included") or "true"
        if include.strip().lower() in {"0", "false", "no", "n"}:
            continue
        npz_path = resolve_project_path(row.get("npz_path", ""))
        if npz_path.exists():
            rows.append(row)
    return rows


def compute_normalization(rows: list[dict], max_patches: int = 0) -> Normalization:
    selected = rows if max_patches <= 0 else rows[:max_patches]
    sums = None
    squares = None
    count = 0
    for row in selected:
        with np.load(resolve_project_path(row["npz_path"])) as data:
            image = data["image"].astype("float64")
            valid = data["valid"].astype(bool)
        values = image[:, valid]
        if values.size == 0:
            continue
        if sums is None:
            sums = values.sum(axis=1)
            squares = (values * values).sum(axis=1)
        else:
            sums += values.sum(axis=1)
            squares += (values * values).sum(axis=1)
        count += values.shape[1]
    if sums is None or count == 0:
        raise SystemExit("cannot compute normalization: no valid pixels")
    mean = sums / count
    variance = np.maximum(squares / count - mean * mean, 1e-6)
    return Normalization(mean.astype("float32"), np.sqrt(variance).astype("float32"))


def compute_pos_weight(rows: list[dict]) -> float:
    positive = 0
    negative = 0
    for row in rows:
        with np.load(resolve_project_path(row["npz_path"])) as data:
            mask = data["mask"].astype("uint8")
            valid = data["valid"].astype(bool) & (mask != IGNORE_INDEX)
        positive += int(((mask == 1) & valid).sum())
        negative += int(((mask == 0) & valid).sum())
    if positive == 0:
        return 1.0
    return float(min(max(negative / positive, 1.0), 50.0))


def patch_channels(row: dict) -> int:
    with np.load(resolve_project_path(row["npz_path"])) as data:
        return int(data["image"].shape[0])


def augment_patch(image: np.ndarray, mask: np.ndarray, valid: np.ndarray):
    if random.random() < 0.5:
        image = image[:, :, ::-1]
        mask = mask[:, ::-1]
        valid = valid[:, ::-1]
    if random.random() < 0.5:
        image = image[:, ::-1, :]
        mask = mask[::-1, :]
        valid = valid[::-1, :]
    turns = random.randint(0, 3)
    if turns:
        image = np.rot90(image, turns, axes=(1, 2))
        mask = np.rot90(mask, turns, axes=(0, 1))
        valid = np.rot90(valid, turns, axes=(0, 1))
    return image, mask, valid


def resolve_project_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path
