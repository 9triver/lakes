# Use global Training Profiles as overlays on shared training facts

> Superseded by [ADR 0002](0002-profile-owned-logical-patch-catalogs.md). Training Samples remain shared, but Logical Patch catalogs are now user-owned.

Training Samples, label snapshots, imagery, and Logical Patch grids remain global shared facts; a Training Profile stores only Patch membership, one resolved Source Variant set per site, derived datasets, runs, and models. Profiles may span regions, union creation takes a point-in-time Patch snapshot, and differing site source selections create a manually resolved draft instead of silently duplicating examples or querying live sources. This keeps samples immutable and reusable while making each experiment reproducible; the cost is an explicit conflict-resolution step and Profile-aware dataset materialization.

Legacy `include` state and existing models are attributed to the default Profile without moving model files. New runs freeze their manifest and use Profile-specific model paths.
