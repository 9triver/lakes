"""Bounded review responses built from the Workspace manifest cache."""

from lake_workbench.utils import read_csv_records
from lake_workbench.workspaces.membership import patch_is_included


def workspace_review_payload(handler, params, *, all_regions=False):
    store = handler.__class__.workspace_store
    catalogs = (handler.__class__.region_service.catalogs if all_regions
                else {handler.catalog.region.key: handler.catalog})
    value = lambda key, default="": params.get(key, [default])[0]
    limit = max(1, min(1000, int(value("limit", "18"))))
    offset = max(0, int(value("offset", "0")))
    rows = []
    for region, catalog in catalogs.items():
        manifest = store.ensure_workspace_logical_patch_manifest(handler.workspace_id, region)
        for row in store._workspace_logical_index(handler.workspace_id, region).values():
            if value("site_id") and row.get("site_id") != value("site_id"):
                continue
            if value("sample_id") and row.get("sample_id") != value("sample_id"):
                continue
            if value("image_index") and str(row.get("image_index")) != value("image_index"):
                continue
            rows.append((catalog, manifest, row))
    # Contributions are sparse; avoid allocating one empty list per Patch.
    shared = {}
    registry = handler.__class__.dataset_registry
    for manifest in sorted(registry.root.glob("*/manifest.csv")):
        for row in read_csv_records(manifest):
            if row.get("source_workspace_id") == handler.workspace_id:
                shared.setdefault(row.get("source_patch_id", ""), []).append(manifest.parent.name)
    counts = dict(all=len(rows), included=0, excluded=0, contributed=0, uncontributed=0)
    sites = {}
    selected = []
    for catalog, manifest, row in rows:
        included = patch_is_included(row)
        contributed = bool(shared.get(row.get("logical_patch_id", "")))
        counts["included" if included else "excluded"] += 1
        counts["contributed" if contributed else "uncontributed"] += 1
        key = f"{catalog.region.key}:{row.get('site_id', '')}"
        if key not in sites:
            site = catalog.get_site(row.get("site_id", ""))
            sites[key] = {"key": key, "region": catalog.region.key,
                          "regionName": catalog.region.name, "siteId": row.get("site_id", ""),
                          "displayName": site.display_name if site else row.get("site_name") or row.get("site_id", ""),
                          "patchCount": 0, "includedCount": 0}
        sites[key]["patchCount"] += 1
        sites[key]["includedCount"] += int(included)
        status = value("include")
        if ((status == "included" and not included) or (status == "excluded" and included)
                or (status == "contributed" and not contributed) or (status == "uncontributed" and contributed)):
            continue
        water = int(row.get("water_pixels") or 0) > 0
        if (value("water") == "water" and not water) or (value("water") == "negative" and water):
            continue
        selected.append((catalog, manifest, row))
    if value("view") == "sites":
        items = sorted(sites.values(), key=lambda item: (item["region"], item["displayName"]))
    else:
        items = []
        for catalog, manifest, row in selected[offset:offset + limit]:
            item = catalog._logical_patch_summary(row, manifest)
            scopes = shared.get(item["logical_patch_id"], [])
            items.append({**item, "workspace_id": handler.workspace_id,
                          "region": catalog.region.key, "region_name": catalog.region.name,
                          "contribution_scopes": scopes, "contributed": bool(scopes),
                          "preview_url": f"/api/workspaces/{handler.workspace_id}/regions/{catalog.region.key}/logical-patches/{item['logical_patch_id']}/preview.png"})
    return {"workspace_id": handler.workspace_id, "total": len(selected),
            "included_count": counts["included"], "excluded_count": counts["excluded"],
            "counts": counts, "limit": limit, "offset": offset, "items": items}
