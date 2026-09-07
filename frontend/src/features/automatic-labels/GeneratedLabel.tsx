import { useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { Typography } from "@mui/material";
import { postJson } from "../../api/client";
import type { GeneratedLabelResult, GeneratedLabelSource } from "../../api/types";
import type { ImagerySelection } from "../imagery/ImageryPanel";
import type { SiteMapHandle } from "../map/SiteMap";
import { workspaceRegionApi } from "../workspaces/api";

const SOURCE_LABELS: Record<GeneratedLabelSource, string> = {
  spectral_water: "光谱水体",
  spectral_osm_consensus: "光谱 + OSM 一致",
};

interface GeneratedLabelOptions {
  workspaceId: string;
  region: string;
  siteId: string;
  source: GeneratedLabelSource;
  imagery: ImagerySelection;
  mapHandle?: React.RefObject<SiteMapHandle | null>;
  onGenerated: (result: GeneratedLabelResult) => void;
}

export function useGeneratedLabel({ workspaceId, region, siteId, source, imagery, mapHandle, onGenerated }: GeneratedLabelOptions) {
  const contextKey = `${workspaceId}:${region}:${siteId}:${imagery.assetId || imagery.product}`;
  const currentContextKey = useRef(contextKey);
  const latestRequest = useRef(0);
  currentContextKey.current = contextKey;
  return useMutation({
    mutationFn: async () => {
      const requestId = ++latestRequest.current;
      if (!imagery.assetId && !imagery.product) throw new Error("请先选择本地影像期次");
      const extent = mapHandle?.current?.captureView()?.map.extent;
      const result = await postJson<GeneratedLabelResult>(
        workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(siteId)}/generated-labels/${source}`),
        {
          asset_id: imagery.assetId,
          product: imagery.product,
          extent,
          max_dimension: 2048,
        },
      );
      return { result, requestId, contextKey };
    },
    onSuccess: ({ result, requestId, contextKey: completedContext }) => {
      if (requestId === latestRequest.current && completedContext === currentContextKey.current) onGenerated(result);
    },
  });
}

export function GeneratedLabelStatus({ source, result, pending, error }: { source: GeneratedLabelSource; result?: GeneratedLabelResult; pending?: boolean; error?: Error | null }) {
  const label = SOURCE_LABELS[source];
  if (pending) return <Typography variant="caption" color="text.secondary">{label}生成中</Typography>;
  if (error) return <Typography variant="caption" color="error">{label}失败：{error.message}</Typography>;
  if (!result) return null;
  const method = source === "spectral_osm_consensus" ? " · 交集" : "";
  const confidence = source === "spectral_water" || source === "spectral_osm_consensus"
    ? ` · 高置信 ${(result.stats.confident_ratio * 100).toFixed(1)}% · 待确认 ${(result.stats.ignore_pixels / Math.max(result.stats.valid_pixels, 1) * 100).toFixed(1)}%`
    : "";
  return <Typography variant="caption" color="text.secondary">
    {label}{method} · 水体 {(result.stats.water_ratio * 100).toFixed(1)}%{confidence} · {result.stats.polygon_count} 个多边形
  </Typography>;
}
