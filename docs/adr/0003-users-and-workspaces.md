# Separate User identity from Workspace ownership

Status: accepted.

## Decision

A User is a named identity with a one-way `default_workspace_id` mapping. Each User has exactly one default Workspace for now. Workspace remains an independent domain concept and does not store `user_id`.

Shared observation data consists of Region, Site, Imagery, Annotation, and Training Sample. Workspace owns Logical Patch catalogs and review state, Source Variant selections, materialized datasets, Training Runs, and Models.

Training APIs use `/api/workspaces/<workspace_id>/regions/...`. User management uses `/api/users`. Frontend deep links contain both IDs so the application can validate their mapping.

## Consequences

- User archival does not delete or archive training resources.
- Authentication maps an external identity to User without changing Workspace ownership.
- New Workspaces start empty and never absorb region-level Logical Patch manifests.
- Old Profile APIs, storage paths, model lookup fallbacks, and short model keys are unsupported.
- Model keys are fully qualified as `workspaces/<workspace>/<scope>/<run>/<weight>`.
