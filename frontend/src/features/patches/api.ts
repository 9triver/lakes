import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, patchJson, postJson } from "../../api/client";
import type { TrainingPatch } from "../../api/types";
import type { TileMeta } from "../../api/types";

export interface PatchExportJob {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  message?: string;
  progress?: number;
}

export interface PatchExportOptions {
  patch_size: number;
  stride: number;
  preview_scale: number;
  overwrite: boolean;
}

export function useTrainingPatches(scope: string, include: string) {
  const suffix = include ? `?include=${encodeURIComponent(include)}` : "";
  return useQuery({ queryKey: ["logical-patches", scope, include], queryFn: () => getJson<{ items: TrainingPatch[]; total: number; included_count: number; excluded_count: number }>(`/api/regions/${encodeURIComponent(scope)}/logical-patches${suffix}`), enabled: Boolean(scope) });
}

export function useUpdateTrainingPatch(scope: string, include: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (patch: TrainingPatch) => patchJson(`/api/regions/${encodeURIComponent(patch.region || scope)}/logical-patches/${encodeURIComponent(patch.patch_id)}`, { include: !patch.included }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["logical-patches"] }),
        client.invalidateQueries({ queryKey: ["sites"] }),
        client.invalidateQueries({ queryKey: ["training-datasets"] }),
      ]);
    },
  });
}

export function useStartPatchExport(scope: string) {
  return useMutation({ mutationFn: (options: PatchExportOptions) => postJson<PatchExportJob>(`/api/regions/${encodeURIComponent(scope)}/logical-patches/build-jobs`, options) });
}

export function usePatchExportJob(scope: string, jobId: string) {
  return useQuery({
    queryKey: ["patch-export-job", scope, jobId],
    queryFn: () => getJson<PatchExportJob>(`/api/regions/${encodeURIComponent(scope)}/training-patches/export-jobs/${encodeURIComponent(jobId)}`),
    enabled: Boolean(scope && jobId),
    refetchInterval: (query) => ["completed", "failed"].includes(query.state.data?.status || "") ? false : 1500,
  });
}

export function useSiteLogicalPatches(region: string, siteId: string) {
  const params = new URLSearchParams({ site_id: siteId });
  return useQuery({
    queryKey: ["logical-patches", region, siteId],
    queryFn: () => getJson<{ items: TrainingPatch[]; total: number; included_count: number; excluded_count: number }>(`/api/regions/${encodeURIComponent(region)}/logical-patches?${params}`),
    enabled: Boolean(region && siteId),
  });
}

export function useBatchUpdateLogicalPatches(region: string, siteId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ operation, ids }: { operation: "exclude" | "restore"; ids: string[] }) => patchJson(`/api/regions/${encodeURIComponent(region)}/logical-patches`, { operation, logical_patch_ids: ids }),
    onSuccess: async () => Promise.all([
      client.invalidateQueries({ queryKey: ["logical-patches"] }),
      client.invalidateQueries({ queryKey: ["sites"] }),
      client.invalidateQueries({ queryKey: ["training-datasets"] }),
    ]),
  });
}

export function useLogicalPatchSourceMeta(region: string, siteId: string, patchId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["logical-patch-source", region, siteId, patchId],
    queryFn: () => getJson<TileMeta>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/logical-patch-source/meta?patch_id=${encodeURIComponent(patchId)}`),
    enabled: Boolean(enabled && region && siteId && patchId),
  });
}
