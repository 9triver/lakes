import { useMutation, useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { FeatureCollection, SiteSummary, SiteDetail } from "../../api/types";
import type { TrainingDataset, TrainingEpoch } from "../training/api";
import { workspaceRegionApi } from "../workspaces/api";

export type ThresholdStrategy = "manual" | "calibrated_adaptive";

export interface RoutingInfo {
  k?: number; rule?: string; rejected_windows?: number; low_valid_windows?: number; profile_id: string; windows: number; fallback_fraction: number; overlay: string;
  components: { cluster: number; label: string; windows: number; coverage_fraction: number; color: string }[];
}

export interface ModelOption {
  routing_model?: boolean;
  threshold_calibration?: { id: string; label: string };
  key: string;
  label: string;
  name?: string;
  weight?: string;
  path?: string;
  scope?: string;
  region?: string;
  epoch?: number;
  in_channels?: number;
  base_channels?: number;
  model_type?: "unet" | "pixel_mlp" | "aer_unet_adapted" | "resddsc_unetpp_adapted";
  model_options?: Record<string, unknown>;
  architecture_label?: string;
  best_iou?: number;
  best_epoch?: number;
  updated_at?: string | number;
  error?: string;
  latest?: TrainingEpoch;
  dataset?: TrainingDataset;
  config?: Record<string, unknown>;
  workspace_id?: string;
  workspace_name?: string;
  workspace_status?: string;
  completed?: boolean;
  experimental?: boolean;
  selection?: string;
  selected_validation_iou?: number;
  selected_validation_loss?: number;
  max_validation_iou?: number | null;
}

export interface ModelValidationResult {
  region: string;
  site_id: string;
  site?: SiteSummary;
  model: { key: string; name: string; path?: string; device?: string; epoch?: number; in_channels?: number; base_channels?: number; model_type?: "unet" | "pixel_mlp" | "aer_unet_adapted" | "resddsc_unetpp_adapted"; model_options?: Record<string, unknown>; architecture_label?: string; threshold?: number; threshold_strategy?: ThresholdStrategy; threshold_selection?: { profile_id?: string; calibrated_threshold?: number } };
  prediction: FeatureCollection;
  stats: { routing?: RoutingInfo | null; area_km2?: number; predicted_ratio?: number; threshold?: number };
  imagery?: { tile?: string; product?: string; product_name?: string; date?: string; products?: string[]; tiles?: string[] };
  cached?: boolean;
}

function sortedModels(items: ModelOption[]) {
  return [...items].sort((a, b) => {
    if (Boolean(a.error) !== Boolean(b.error)) return a.error ? 1 : -1;
    const aScore = Number(a.best_iou);
    const bScore = Number(b.best_iou);
    if (Number.isFinite(aScore) !== Number.isFinite(bScore)) return Number.isFinite(aScore) ? -1 : 1;
    if (Number.isFinite(aScore) && bScore !== aScore) return bScore - aScore;
    if ((a.weight === "best.pt") !== (b.weight === "best.pt")) return a.weight === "best.pt" ? -1 : 1;
    return a.label.localeCompare(b.label, "zh-CN");
  });
}

export function useValidationModels(workspaceId: string, scope: string, visibility: "current" | "all") {
  return useQuery({
    queryKey: ["validation-models", workspaceId, scope, visibility],
    queryFn: async () => {
      const result = await getJson<{ default: string; items: ModelOption[] }>(workspaceRegionApi(workspaceId, scope, `/model-validation/models?visibility=${visibility}`));
      return { ...result, items: sortedModels(result.items) };
    },
    enabled: Boolean(workspaceId && scope),
  });
}

export interface RandomModelValidationTarget {
  region: string;
  site_id: string;
  site: SiteDetail;
}

export function useRandomModelValidationTarget(workspaceId: string) {
  return useMutation({ mutationFn: () => getJson<RandomModelValidationTarget>(
    workspaceRegionApi(workspaceId, "all", "/model-validation/random-target"),
  ) });
}

export function useRandomModelValidation(workspaceId: string, scope: string) {
  return useMutation({ mutationFn: ({ model, threshold }: { model: string; threshold: number }) => {
    const params = new URLSearchParams({ model, threshold: String(threshold) });
    return getJson<ModelValidationResult>(workspaceRegionApi(workspaceId, scope, `/model-validation/random?${params}`));
  } });
}

export function useSiteModelPrediction(workspaceId: string, region: string, siteId: string, model: string, threshold: number, enabled: boolean, thresholdStrategy: ThresholdStrategy = "manual") {
  return useQuery({
    queryKey: ["model-prediction", workspaceId, region, siteId, model, threshold, thresholdStrategy],
    queryFn: () => {
      const params = new URLSearchParams({ model, threshold: String(threshold), threshold_strategy: thresholdStrategy });
      return getJson<ModelValidationResult>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(siteId)}/model-prediction?${params}`));
    },
    enabled: enabled && Boolean(workspaceId && region && siteId && model),
    retry: false, staleTime: Infinity, refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
}
