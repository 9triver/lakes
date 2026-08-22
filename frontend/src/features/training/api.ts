import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, postJson } from "../../api/client";
import { workspaceRegionApi } from "../workspaces/api";
import type { GlobalDataset } from "../../api/types";

export interface TrainingDataset {
  scope: string;
  regions?: string[];
  usable_patches?: number;
  included_patches?: number;
  excluded_patches?: number;
  sample_count?: number;
  site_count?: number;
  train_count?: number;
  val_count?: number;
  in_channels?: number;
  patch_size?: number[];
  water_ratio?: number;
  error?: string;
}

interface EpochMetrics { loss?: number; iou?: number; dice?: number }
export interface TrainingEpoch { epoch: number; train?: EpochMetrics; val?: EpochMetrics }
export interface TrainingRun {
  job_id: string;
  run_name?: string;
  status: string;
  message?: string;
  progress?: number;
  epoch?: number;
  epochs?: number;
  created_at?: string;
  updated_at?: string;
  history?: TrainingEpoch[];
  dataset?: TrainingDataset;
  config?: Record<string, unknown>;
  result?: { best_iou?: number; best_model?: string; output_dir?: string; config?: Record<string, unknown>; dataset?: TrainingDataset };
}

export interface TrainingOptions {
  run_name: string;
  model_type: "unet" | "pixel_mlp";
  epochs: number;
  batch_size: number;
  lr: number;
  base_channels: number;
  hidden_channels: [number, number];
  device: string;
  no_augment: boolean;
  dataset_config_id: string;
  dataset_source: "workspace" | "global";
}

export interface TrainingDatasetConfigStatus {
  config_id: string;
  config: { id: string; label: string; output_size: number; min_valid_ratio: number; min_water_pixels: number; negative_ratio: number };
  status: "ready" | "stale" | "missing" | "missing_logical";
  ready: boolean;
  patches: number;
}

export interface DatasetBuildJob { job_id: string; status: string; message?: string; progress?: number }

const runningStatuses = new Set(["queued", "configured", "running", "cancel_requested"]);

export function isTrainingRunActive(run?: TrainingRun) {
  return Boolean(run && runningStatuses.has(run.status));
}

export function useTrainingRuns(workspaceId: string, scope: string, datasetConfigId = "resize256_v1", datasetSource: "workspace" | "global" = "workspace") {
  return useQuery({
    queryKey: ["training-runs", workspaceId, scope, datasetConfigId, datasetSource],
    queryFn: () => getJson<{ scope: string; dataset: TrainingDataset; items: TrainingRun[] }>(workspaceRegionApi(workspaceId, scope, `/training-runs?dataset_config_id=${encodeURIComponent(datasetConfigId)}&dataset_source=${datasetSource}`)),
    enabled: Boolean(workspaceId && scope),
    refetchInterval: (query) => query.state.data?.items.some(isTrainingRunActive) ? 2000 : false,
  });
}

export function useStartTrainingRun(workspaceId: string, scope: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (options: TrainingOptions) => postJson<TrainingRun>(workspaceRegionApi(workspaceId, scope, "/training-runs"), options), onSuccess: () => client.invalidateQueries({ queryKey: ["training-runs", workspaceId, scope] }) });
}

export function useCancelTrainingRun(workspaceId: string, scope: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (jobId: string) => postJson<TrainingRun>(workspaceRegionApi(workspaceId, scope, `/training-runs/${encodeURIComponent(jobId)}/cancel`), {}), onSuccess: () => client.invalidateQueries({ queryKey: ["training-runs", workspaceId, scope] }) });
}

export function useTrainingDatasetConfigs(workspaceId: string, scope: string) {
  return useQuery({ queryKey: ["training-datasets", workspaceId, scope], queryFn: () => getJson<{ items: TrainingDatasetConfigStatus[] }>(workspaceRegionApi(workspaceId, scope, "/training-datasets")), enabled: Boolean(workspaceId && scope) });
}

export function useGlobalDataset(workspaceId: string, scope: string) {
  return useQuery({
    queryKey: ["global-dataset", workspaceId, scope],
    queryFn: () => getJson<GlobalDataset>(workspaceRegionApi(workspaceId, scope, "/global-dataset")),
    enabled: Boolean(workspaceId && scope),
  });
}

export function useBuildGlobalDataset(workspaceId: string, scope: string) {
  return useMutation({
    mutationFn: (configId: string) => postJson<DatasetBuildJob>(workspaceRegionApi(workspaceId, scope, "/global-dataset/build-jobs"), { config_id: configId }),
  });
}

export function useGlobalDatasetBuildJob(workspaceId: string, scope: string, jobId: string) {
  const client = useQueryClient();
  return useQuery({
    queryKey: ["global-dataset-build", workspaceId, scope, jobId],
    queryFn: async () => {
      const job = await getJson<DatasetBuildJob>(workspaceRegionApi(workspaceId, scope, `/global-dataset/build-jobs/${encodeURIComponent(jobId)}`));
      if (job.status === "completed") await client.invalidateQueries({ queryKey: ["global-dataset", workspaceId, scope] });
      return job;
    },
    enabled: Boolean(workspaceId && scope && jobId),
    refetchInterval: (query) => ["completed", "failed"].includes(query.state.data?.status || "") ? false : 1500,
  });
}

export function useBuildTrainingDataset(workspaceId: string, scope: string) {
  return useMutation({ mutationFn: (configId: string) => postJson<DatasetBuildJob>(workspaceRegionApi(workspaceId, scope, `/training-datasets/${encodeURIComponent(configId)}/build-jobs`), {}) });
}

export function useTrainingDatasetBuildJob(workspaceId: string, scope: string, configId: string, jobId: string) {
  const client = useQueryClient();
  return useQuery({
    queryKey: ["training-dataset-build", workspaceId, scope, configId, jobId],
    queryFn: async () => {
      const job = await getJson<DatasetBuildJob>(workspaceRegionApi(workspaceId, scope, `/training-datasets/${encodeURIComponent(configId)}/build-jobs/${encodeURIComponent(jobId)}`));
      if (job.status === "completed") {
        await Promise.all([client.invalidateQueries({ queryKey: ["training-datasets", workspaceId, scope] }), client.invalidateQueries({ queryKey: ["training-runs", workspaceId, scope] })]);
      }
      return job;
    },
    enabled: Boolean(workspaceId && scope && configId && jobId),
    refetchInterval: (query) => ["completed", "failed"].includes(query.state.data?.status || "") ? false : 1500,
  });
}
