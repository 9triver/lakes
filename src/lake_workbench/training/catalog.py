"""Training catalog facade composed from sample and patch responsibilities."""

from lake_workbench.training.patch_catalog import TrainingPatchCatalogMixin
from lake_workbench.training.samples import TrainingSampleCatalogMixin


class TrainingCatalogMixin(TrainingSampleCatalogMixin, TrainingPatchCatalogMixin):
    """Stable SiteCatalog mixin API for training-related operations."""
