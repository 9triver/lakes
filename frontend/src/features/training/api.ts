import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, postJson } from "../../api/client";

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
  epochs: number;
  batch_size: number;
  lr: number;
  base_channels: number;
  device: string;
  no_augment: boolean;
}

const runningStatuses = new Set(["queued", "configured", "running", "cancel_requested"]);

export function isTrainingRunActive(run?: TrainingRun) {
  return Boolean(run && runningStatuses.has(run.status));
}

export function useTrainingRuns(scope: string) {
  return useQuery({
    queryKey: ["training-runs", scope],
    queryFn: () => getJson<{ scope: string; dataset: TrainingDataset; items: TrainingRun[] }>(`/api/regions/${encodeURIComponent(scope)}/training-runs`),
    enabled: Boolean(scope),
    refetchInterval: (query) => query.state.data?.items.some(isTrainingRunActive) ? 2000 : false,
  });
}

export function useStartTrainingRun(scope: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (options: TrainingOptions) => postJson<TrainingRun>(`/api/regions/${encodeURIComponent(scope)}/training-runs`, options), onSuccess: () => client.invalidateQueries({ queryKey: ["training-runs", scope] }) });
}

export function useCancelTrainingRun(scope: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (jobId: string) => postJson<TrainingRun>(`/api/regions/${encodeURIComponent(scope)}/training-runs/${encodeURIComponent(jobId)}/cancel`, {}), onSuccess: () => client.invalidateQueries({ queryKey: ["training-runs", scope] }) });
}
