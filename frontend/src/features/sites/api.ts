import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { ContextWaterResponse, FeatureCollection, GeoJsonLayer, SiteDetail, SiteFilters, SitesResponse, LocalLabelItem, TileMeta } from "../../api/types";

const sitePageSize = 200;

export function useSites(region: string, query: string, filters: SiteFilters) {
  return useInfiniteQuery({
    queryKey: ["sites", region, query, filters],
    queryFn: ({ pageParam }) => {
      const params = new URLSearchParams({ q: query, limit: String(sitePageSize), offset: String(pageParam) });
      Object.entries(filters).forEach(([key, value]) => { if (value) params.set(key, value); });
      return getJson<SitesResponse>(`/api/regions/${encodeURIComponent(region)}/sites?${params}`);
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage, pages) => {
      const loaded = pages.reduce((total, page) => total + page.items.length, 0);
      return loaded < lastPage.total ? loaded : undefined;
    },
    enabled: Boolean(region),
  });
}

export function useSite(region: string, siteId: string) {
  return useQuery({
    queryKey: ["site", region, siteId],
    queryFn: () => getJson<SiteDetail>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}`),
    enabled: Boolean(region && siteId),
  });
}

export function useTileMeta(region: string, siteId: string) {
  return useQuery({
    queryKey: ["tile-meta", region, siteId],
    queryFn: () => getJson<TileMeta>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/tile-meta?padding=0.8`),
    enabled: Boolean(region && siteId),
    retry: false,
  });
}

export function useContextWater(region: string, siteId: string) {
  return useQuery({ queryKey: ["context-water", region, siteId], queryFn: () => getJson<ContextWaterResponse>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/context-water?padding=0.8&min_area_km2=10&limit=500`), enabled: Boolean(region && siteId) });
}

function useAnnotation<T extends GeoJsonLayer | FeatureCollection>(region: string, siteId: string, source: string, query = "", enabled = true) {
  return useQuery({
    queryKey: ["annotation", region, siteId, source, query],
    queryFn: async () => (
      await getJson<{ annotation: T | null }>(
        `/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/annotations/${source}${query}`,
      )
    ).annotation,
    enabled: enabled && Boolean(region && siteId),
    retry: false,
  });
}

export function useOsmLayer(region: string, siteId: string) {
  return useAnnotation<GeoJsonLayer>(region, siteId, "osm");
}

export function useHydrolakesLayer(region: string, siteId: string) {
  return useAnnotation<GeoJsonLayer>(region, siteId, "hydrolakes");
}

export function useEsaLayer(region: string, siteId: string) {
  return useAnnotation<GeoJsonLayer>(region, siteId, "esa");
}

export function useJrcLayer(region: string, siteId: string, threshold: number) {
  return useAnnotation<GeoJsonLayer>(region, siteId, "jrc", `?threshold=${threshold}`);
}

export function useLocalLabels(region: string, siteId: string) {
  return useQuery({ queryKey: ["local-labels", region, siteId], queryFn: async () => (await getJson<{ items: LocalLabelItem[] }>(`/api/regions/${encodeURIComponent(region)}/sites/${encodeURIComponent(siteId)}/local-labels`)).items, enabled: Boolean(region && siteId) });
}

export function useLocalLabel(region: string, siteId: string, labelId: string) {
  return useAnnotation<FeatureCollection>(region, siteId, "local", `?label_id=${encodeURIComponent(labelId)}`, Boolean(labelId));
}
