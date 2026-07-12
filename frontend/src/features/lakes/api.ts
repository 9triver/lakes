import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { ContextWaterResponse, FeatureCollection, GeoJsonLayer, LakeDetail, LakeFilters, LakesResponse, LocalLabelItem, TileMeta } from "../../api/types";

const lakePageSize = 200;

export function useLakes(region: string, query: string, filters: LakeFilters) {
  return useInfiniteQuery({
    queryKey: ["lakes", region, query, filters],
    queryFn: ({ pageParam }) => {
      const params = new URLSearchParams({ q: query, limit: String(lakePageSize), offset: String(pageParam) });
      Object.entries(filters).forEach(([key, value]) => { if (value) params.set(key, value); });
      return getJson<LakesResponse>(`/api/regions/${encodeURIComponent(region)}/lakes?${params}`);
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage, pages) => {
      const loaded = pages.reduce((total, page) => total + page.items.length, 0);
      return loaded < lastPage.total ? loaded : undefined;
    },
    enabled: Boolean(region),
  });
}

export function useLake(region: string, lakeId: string) {
  return useQuery({
    queryKey: ["lake", region, lakeId],
    queryFn: () => getJson<LakeDetail>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}`),
    enabled: Boolean(region && lakeId),
  });
}

export function useTileMeta(region: string, lakeId: string) {
  return useQuery({
    queryKey: ["tile-meta", region, lakeId],
    queryFn: () => getJson<TileMeta>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}/tile-meta?padding=0.8`),
    enabled: Boolean(region && lakeId),
    retry: false,
  });
}

export function useContextWater(region: string, lakeId: string) {
  return useQuery({ queryKey: ["context-water", region, lakeId], queryFn: () => getJson<ContextWaterResponse>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}/context-water?padding=0.8&min_area_km2=10&limit=500`), enabled: Boolean(region && lakeId) });
}

export function useEsaLayer(region: string, lakeId: string) {
  return useQuery({ queryKey: ["esa", region, lakeId], queryFn: async () => (await getJson<{ esa: GeoJsonLayer | null }>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}/esa`)).esa, enabled: Boolean(region && lakeId), retry: false });
}

export function useJrcLayer(region: string, lakeId: string, threshold: number) {
  return useQuery({ queryKey: ["jrc", region, lakeId, threshold], queryFn: async () => (await getJson<{ jrc: GeoJsonLayer | null }>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}/jrc?threshold=${threshold}`)).jrc, enabled: Boolean(region && lakeId), retry: false });
}

export function useLocalLabels(region: string, lakeId: string) {
  return useQuery({ queryKey: ["local-labels", region, lakeId], queryFn: async () => (await getJson<{ items: LocalLabelItem[] }>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}/local-labels`)).items, enabled: Boolean(region && lakeId) });
}

export function useLocalLabel(region: string, lakeId: string, labelId: string) {
  return useQuery({ queryKey: ["local-label", region, lakeId, labelId], queryFn: async () => (await getJson<{ geojson: FeatureCollection }>(`/api/regions/${encodeURIComponent(region)}/lakes/${encodeURIComponent(lakeId)}/local-labels/${encodeURIComponent(labelId)}`)).geojson, enabled: Boolean(region && lakeId && labelId) });
}
