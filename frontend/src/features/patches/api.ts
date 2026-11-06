import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, patchJson, postJson } from "../../api/client";
import type { TrainingPatch } from "../../api/types";
import type { TileMeta } from "../../api/types";
import { workspaceRegionApi } from "../workspaces/api";

export interface GlobalDatasetContribution {
  dataset_id: string;
  added: number;
  replaced: number;
  total: number;
  conflicts?: Array<Record<string, unknown>>;
}

export interface TrainingPatchSiteSummary {
  key: string;
  region: string;
  regionName: string;
  siteId: string;
  displayName: string;
  patchCount: number;
  includedCount: number;
}

export interface PatchPage {
  items: TrainingPatch[];
  total: number;
  included_count: number;
  excluded_count: number;
  counts: Record<string, number>;
}

export function useTrainingPatches(workspaceId: string, scope: string, include = "", sampleId = "", siteId = "", page = 1, water = "") {
  const params = new URLSearchParams({ limit: "18", offset: String((page - 1) * 18) });
  if (include && include !== "all") params.set("include", include);
  if (sampleId) params.set("sample_id", sampleId);
  if (siteId) params.set("site_id", siteId);
  if (water) params.set("water", water);
  return useQuery({ queryKey: ["logical-patches", workspaceId, scope, include, sampleId, siteId, page, water], queryFn: () => getJson<PatchPage>(workspaceRegionApi(workspaceId, scope, `/logical-patches?${params}`)), enabled: Boolean(workspaceId && scope) });
}

export function useTrainingPatchSites(workspaceId: string, scope: string, enabled: boolean) {
  return useQuery({
    queryKey: ["logical-patches", workspaceId, scope, "sites"],
    queryFn: () => getJson<{ items: TrainingPatchSiteSummary[] }>(workspaceRegionApi(workspaceId, scope, "/logical-patches?view=sites")),
    enabled: Boolean(enabled && workspaceId && scope),
    select: (payload) => [...payload.items].sort((left, right) =>
        left.region.localeCompare(right.region) || left.displayName.localeCompare(right.displayName, "zh-CN", { numeric: true }),
      ),
  });
}

export function useUpdateTrainingPatch(workspaceId: string, scope: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ patch, replace = false }: { patch: TrainingPatch; replace?: boolean }) => patchJson(workspaceRegionApi(workspaceId, patch.region || scope, `/logical-patches/${encodeURIComponent(patch.patch_id)}`), { include: !patch.included, replace }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["logical-patches"] }),
        client.invalidateQueries({ queryKey: ["sites"] }),
        client.invalidateQueries({ queryKey: ["training-datasets"] }),
      ]);
    },
  });
}

export function useContributePatches(workspaceId: string, targetScope: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ patches, replace = false }: { patches: TrainingPatch[]; replace?: boolean }) => {
      const groups = new Map<string, TrainingPatch[]>();
      for (const patch of patches) {
        const region = patch.region || targetScope;
        groups.set(region, [...(groups.get(region) || []), patch]);
      }
      return postJson<GlobalDatasetContribution>(workspaceRegionApi(workspaceId, targetScope, "/global-dataset/patches"), {
        source_groups: Object.fromEntries([...groups.entries()].map(([sourceRegion, items]) => [sourceRegion, items.map((patch) => patch.logical_patch_id || patch.patch_id)])),
        replace,
      });
    },
    onSuccess: async () => Promise.all([
      client.invalidateQueries({ queryKey: ["logical-patches"] }),
      client.invalidateQueries({ queryKey: ["global-dataset"] }),
      client.invalidateQueries({ queryKey: ["training-datasets"] }),
    ]),
  });
}

export function useSiteLogicalPatches(workspaceId: string, region: string, siteId: string) {
  const params = new URLSearchParams({ site_id: siteId, limit: "1000" });
  return useQuery({
    queryKey: ["logical-patches", workspaceId, region, siteId],
    queryFn: () => getJson<{ items: TrainingPatch[]; total: number; included_count: number; excluded_count: number }>(workspaceRegionApi(workspaceId, region, `/logical-patches?${params}`)),
    enabled: Boolean(workspaceId && region && siteId),
  });
}

export function useBatchUpdateLogicalPatches(workspaceId: string, region: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ operation, ids }: { operation: "exclude" | "restore"; ids: string[] }) => patchJson(workspaceRegionApi(workspaceId, region, "/logical-patches"), { operation, logical_patch_ids: ids }),
    onSuccess: async () => Promise.all([
      client.invalidateQueries({ queryKey: ["logical-patches"] }),
      client.invalidateQueries({ queryKey: ["sites"] }),
      client.invalidateQueries({ queryKey: ["training-datasets"] }),
    ]),
  });
}

export function useLogicalPatchSourceMeta(workspaceId: string, region: string, siteId: string, patchId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["logical-patch-source", workspaceId, region, siteId, patchId],
    queryFn: () => getJson<TileMeta>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(siteId)}/logical-patch-source/meta?patch_id=${encodeURIComponent(patchId)}`)),
    enabled: Boolean(enabled && workspaceId && region && siteId && patchId),
  });
}
