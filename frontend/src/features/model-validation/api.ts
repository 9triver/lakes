import { useMutation, useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { FeatureCollection, SiteSummary } from "../../api/types";
import type { TrainingDataset, TrainingEpoch } from "../training/api";
import { workspaceRegionApi } from "../workspaces/api";

export interface ModelOption {
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
  model_type?: "unet" | "pixel_mlp";
  model_options?: Record<string, unknown>;
  architecture_label?: string;
  best_iou?: number;
  best_epoch?: number;
  updated_at?: string;
  error?: string;
  latest?: TrainingEpoch;
  dataset?: TrainingDataset;
  config?: Record<string, unknown>;
  workspace_id?: string;
  workspace_name?: string;
  workspace_status?: string;
}

export interface ModelValidationResult {
  region: string;
  site_id: string;
  site?: SiteSummary;
  model: { key: string; name: string; path?: string; device?: string; epoch?: number; in_channels?: number; base_channels?: number; model_type?: "unet" | "pixel_mlp"; model_options?: Record<string, unknown>; architecture_label?: string; threshold?: number };
  prediction: FeatureCollection;
  stats: { area_km2?: number; predicted_ratio?: number; threshold?: number };
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

export function useRandomModelValidation(workspaceId: string, scope: string) {
  return useMutation({ mutationFn: ({ model, threshold }: { model: string; threshold: number }) => {
    const params = new URLSearchParams({ model, threshold: String(threshold) });
    return getJson<ModelValidationResult>(workspaceRegionApi(workspaceId, scope, `/model-validation/random?${params}`));
  } });
}

export function useSiteModelPrediction(workspaceId: string, region: string, siteId: string, model: string, threshold: number, enabled: boolean) {
  return useQuery({
    queryKey: ["model-prediction", workspaceId, region, siteId, model, threshold],
    queryFn: () => {
      const params = new URLSearchParams({ model, threshold: String(threshold) });
      return getJson<ModelValidationResult>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(siteId)}/model-prediction?${params}`));
    },
    enabled: enabled && Boolean(workspaceId && region && siteId && model),
    retry: false,
  });
}
