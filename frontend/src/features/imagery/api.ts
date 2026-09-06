import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, postJson } from "../../api/client";
import type { ImageryResponse, SentinelTile } from "../../api/types";

export function useImagery(region: string, siteId: string) {
  return useQuery({ queryKey: ["imagery", region, siteId], queryFn: () => getJson<ImageryResponse>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/imagery`), enabled: Boolean(region && siteId) });
}

export function useSentinelTiles(region: string, siteId: string) {
  return useQuery({ queryKey: ["sentinel-tiles", region, siteId], queryFn: async () => (await getJson<{ tiles: SentinelTile[] }>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/sentinel/tiles`)).tiles, enabled: Boolean(region && siteId) });
}

export function useSetActiveImagery(region: string, siteId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ assetId, product }: { assetId: string; product?: string }) => postJson(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/imagery/active`, { asset_id: assetId, product }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["imagery", region, siteId] }),
        client.invalidateQueries({ queryKey: ["tile-meta", region, siteId] }),
      ]);
    },
  });
}
