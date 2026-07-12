import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, patchJson, postJson } from "../../api/client";
import type { TrainingPatch } from "../../api/types";

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
  return useQuery({ queryKey: ["training-patches", scope, include], queryFn: () => getJson<{ items: TrainingPatch[]; total: number; included_count: number; excluded_count: number }>(`/api/regions/${encodeURIComponent(scope)}/training-patches${suffix}`), enabled: Boolean(scope) });
}

export function useUpdateTrainingPatch(scope: string, include: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (patch: TrainingPatch) => patchJson(`/api/regions/${encodeURIComponent(patch.region || scope)}/training-patches/${encodeURIComponent(patch.patch_id)}`, { include: !patch.included }), onSuccess: () => client.invalidateQueries({ queryKey: ["training-patches", scope, include] }) });
}

export function useStartPatchExport(scope: string) {
  return useMutation({ mutationFn: (options: PatchExportOptions) => postJson<PatchExportJob>(`/api/regions/${encodeURIComponent(scope)}/training-patches/export-jobs`, options) });
}

export function usePatchExportJob(scope: string, jobId: string) {
  return useQuery({
    queryKey: ["patch-export-job", scope, jobId],
    queryFn: () => getJson<PatchExportJob>(`/api/regions/${encodeURIComponent(scope)}/training-patches/export-jobs/${encodeURIComponent(jobId)}`),
    enabled: Boolean(scope && jobId),
    refetchInterval: (query) => ["completed", "failed"].includes(query.state.data?.status || "") ? false : 1500,
  });
}
