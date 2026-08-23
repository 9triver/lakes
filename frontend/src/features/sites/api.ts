import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { ContextWaterResponse, FeatureCollection, GeoJsonLayer, SiteDetail, SitesResponse, TileMeta } from "../../api/types";
import { workspaceRegionApi } from "../workspaces/api";

const sitePageSize = 200;

export function useSites(workspaceId: string, region: string, query: string) {
  return useInfiniteQuery({
    queryKey: ["sites", workspaceId, region, query],
    queryFn: ({ pageParam }) => {
      const params = new URLSearchParams({ q: query, limit: String(sitePageSize), offset: String(pageParam) });
      return getJson<SitesResponse>(workspaceRegionApi(workspaceId, region, `/sites?${params}`));
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage, pages) => {
      const loaded = pages.reduce((total, page) => total + page.items.length, 0);
      return loaded < lastPage.total ? loaded : undefined;
    },
    enabled: Boolean(workspaceId && region),
  });
}

export function useSite(region: string, siteId: string, workspaceId: string) {
  return useQuery({
    queryKey: ["site", workspaceId, region, siteId],
    queryFn: () => getJson<SiteDetail>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(siteId)}`)),
    enabled: Boolean(workspaceId && region && siteId),
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

export function useLocalLabel(region: string, siteId: string, labelId: string) {
  return useAnnotation<FeatureCollection>(region, siteId, "local", `?label_id=${encodeURIComponent(labelId)}`, Boolean(labelId));
}
