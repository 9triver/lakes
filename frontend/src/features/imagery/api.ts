import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, postJson } from "../../api/client";
import type { ImageryResponse, SentinelProduct, SentinelTile } from "../../api/types";

export function useImagery(region: string, lakeId: string) {
  return useQuery({ queryKey: ["imagery", region, lakeId], queryFn: () => getJson<ImageryResponse>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(lakeId)}/imagery`), enabled: Boolean(region && lakeId) });
}

export function useSentinelTiles(region: string, lakeId: string) {
  return useQuery({ queryKey: ["sentinel-tiles", region, lakeId], queryFn: async () => (await getJson<{ tiles: SentinelTile[] }>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(lakeId)}/sentinel/tiles`)).tiles, enabled: Boolean(region && lakeId) });
}

export async function searchSentinelProducts(region: string, lakeId: string, params: { tile: string; start: string; end: string; cloud: number }) {
  const query = new URLSearchParams({ tile: params.tile, site_id: lakeId, start: params.start, end: params.end, cloud: String(params.cloud), product_type: "MSIL1C", limit: "50" });
  return (await getJson<{ products: SentinelProduct[] }>(`/api/regions/${encodeURIComponent(region)}/sentinel/products?${query}`)).products;
}

export function useDownloadSentinel(region: string, lakeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (product: SentinelProduct) => {
      let job = await postJson<{ job_id?: string; status?: string; message?: string }>(`/api/regions/${encodeURIComponent(region)}/sentinel/downloads`, { product });
      while (job.job_id && !["completed", "failed"].includes(job.status || "")) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        job = await getJson(`/api/regions/${encodeURIComponent(region)}/sentinel/downloads/${encodeURIComponent(job.job_id)}`);
      }
      if (job.status === "failed") throw new Error(job.message || "Sentinel 下载失败");
      return job;
    },
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["imagery", region, lakeId] }),
        client.invalidateQueries({ queryKey: ["tile-meta", region, lakeId] }),
        client.invalidateQueries({ queryKey: ["sentinel-tiles", region, lakeId] }),
      ]);
    },
  });
}

export function useSetActiveImagery(region: string, lakeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ tile, product }: { tile: string; product: string }) => postJson(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(lakeId)}/imagery/active`, { tile, product }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["imagery", region, lakeId] }),
        client.invalidateQueries({ queryKey: ["tile-meta", region, lakeId] }),
      ]);
    },
  });
}
