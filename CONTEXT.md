# Lakes Domain Context

Lakes separates shared observation facts from user-owned training workspaces.

## Language

**User**

An authenticated operator identity resolved from Cloudflare Access in production or a fixed local identity in development. A User stores a one-way `default_workspace_id` mapping; a Workspace does not store a reverse `user_id`.

**Workspace**

The owner of one training selection and all derived training resources: Logical Patches, Source Variant selections, Workspace Datasets, Training Runs, and Models.

**Training Sample**

An immutable-enough snapshot of imagery, map extent, and visible annotation geometry recorded from an observation Site. Training Samples are shared observation facts and do not belong to a User or Workspace.

**Logical Patch**

A fixed `512 x 512` geographic grid cell generated from a Training Sample inside one Workspace. Its preview and include/exclude state belong to that Workspace.

**Source Variant**

An immutable annotation snapshot identified by provider, parameters or local-label asset, and geometry content.

**Site Source Selection**

The Source Variant set selected by a Workspace for one Site. It generates training targets for all selected Logical Patches of that Site.

**Dataset Config**

A versioned rule for materializing selected Logical Patches, including output size and valid-pixel eligibility.

**Workspace Dataset**

Rebuildable model input determined by Workspace, region scope, Dataset Config, Logical Patch selection, and Site Source Selection.

**Training Run**

One model-training execution over a frozen Workspace Dataset manifest and parameter set.

**Model**

Weights and metadata produced by a Training Run. A Model belongs to the same Workspace as its Run.

## Invariants

- `User -> default Workspace` is one-way and one-to-one for now.
- Every API request resolves an authenticated external identity to one active User.
- A regular User can access only its default Workspace; an admin may access other Workspaces.
- User archival does not archive or delete its Workspace.
- Region metadata, imagery, annotations, and Training Samples are shared.
- Logical Patch manifests, previews, review state, source selections, datasets, runs, and models are Workspace-owned.
- Every training-resource URL and internal operation requires an explicit `workspace_id`.
- A new Workspace starts empty. Existing region-level Patch manifests are never imported.
- Saving a Training Sample generates its Logical Patches only in the active Workspace.
- Rebuilding Logical Patches processes only Sample IDs already present in that Workspace, plus an explicitly requested Sample.
- Workspace source conflicts must be resolved before dataset materialization or training.
- Old Profile APIs, paths, model keys, and implicit default-Workspace behavior are unsupported.
