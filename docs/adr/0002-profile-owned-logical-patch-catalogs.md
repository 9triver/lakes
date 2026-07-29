# Use Profile-owned Logical Patch catalogs

Only base observation data is shared: region/site metadata, imagery, annotations, and Training Samples. Every Training Profile owns its Logical Patch manifest, previews, include/exclude state, materialized datasets, runs, and models.

An empty Profile starts without Logical Patches. Saving a shared or new Training Sample generates that sample's grid only for the current Profile. Rebuilds process only sample IDs already present in that Profile, so they cannot absorb another Profile's catalog. Union creation copies the selected rows of its source Profiles as a point-in-time snapshot; later changes remain isolated.

Legacy region-level manifests are migration input only. The default Profile receives the complete legacy catalog with its historical selection state. Existing non-default Profiles receive only rows named by their own membership file. Runtime Profile APIs and training jobs read only Profile-owned manifests.
