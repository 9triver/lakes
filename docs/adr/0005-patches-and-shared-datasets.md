# ADR 0005: Patch Registry and Shared Training Datasets

## Status

Accepted

## Decision

Training samples belong to a workspace. A sample records a user's selected
imagery, label sources, and map view for a region site. Logical patches are
independent derived objects: they retain `source_sample_id` for provenance,
but Dataset membership does not depend on traversing the sample relationship.

Each patch identity includes the source image fingerprint, native raster
window (`left`, `top`, `width`, `height`), label fingerprint, channels, and
preprocessing configuration. Its geographic bounds are used for map display
and duplicate/overlap checks. Patch size is therefore part of the identity;
changing the size creates a distinct patch variant.

Workspace Dataset membership is represented by the patch review state in the
workspace manifest. Global Dataset membership is stored separately as a
versioned snapshot containing the source workspace, source patch, label
snapshot, and content provenance. A global Dataset never mutates or deletes
the source workspace patch.

Label selection is Patch-local. Patches from one Site may use different label
sources without creating a Site-level conflict or blocking the Workspace.

Adding a patch checks for exact duplicates and high-overlap patches from the
same image. A conflict returns `409` so the user can choose whether to replace
the target Dataset membership. Replacement never deletes the old Patch object.

Region remains the boundary for shared catalog data; `all` is a Dataset scope
that aggregates contributions from multiple regions.
