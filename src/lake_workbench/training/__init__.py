"""Public training APIs."""

from lake_workbench.training.datasets import (
    dataset_summary_from_config,
    merge_training_dataset_summaries,
    split_rows_by_site,
    summarize_training_manifest,
)
from lake_workbench.training.identity import bbox_from_row, bbox_iou, training_view_signature
from lake_workbench.training.runner import run_patch_export, run_training_job


_MODEL_METADATA_EXPORTS = {
    "model_sort_key",
    "model_training_metadata",
    "persisted_training_job",
}


def __getattr__(name: str):
    if name not in _MODEL_METADATA_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from lake_workbench.models import metadata

    return getattr(metadata, name)


__all__ = [
    "bbox_from_row",
    "bbox_iou",
    "dataset_summary_from_config",
    "merge_training_dataset_summaries",
    "model_sort_key",
    "model_training_metadata",
    "persisted_training_job",
    "run_patch_export",
    "run_training_job",
    "split_rows_by_site",
    "summarize_training_manifest",
    "training_view_signature",
]
