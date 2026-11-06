import { useQuery, useQueryClient } from "@tanstack/react-query";
import { workspaceRegionApi } from "../workspaces/api";
import { streamJson } from "./streamJson";

export interface ComparisonSite {
  site_id: string; display_name: string; seed_cached: boolean; slope_cached: boolean;
  imagery_available: boolean; rank: number; cache_checked?: boolean;
}
interface SiteList {
  items: ComparisonSite[]; total: number | null; checked: number; complete: boolean; message: string;
}
interface SiteEvent {
  type: "progress" | "sites" | "batch" | "complete" | "error";
  items?: ComparisonSite[]; completed?: number; total?: number | null; message?: string;
}

export function useComparisonSites(workspaceId: string, region: string) {
  const client = useQueryClient();
  const key = ["comparison-sites-readiness", workspaceId, region];
  return useQuery({
    queryKey: key,
    enabled: Boolean(workspaceId && region), staleTime: 30000, retry: false,
    queryFn: async ({ signal }): Promise<SiteList> => {
      const entries = new Map<string, ComparisonSite>();
      let latest: SiteList = { items: [], total: null, checked: 0, complete: false, message: "正在读取观测区域目录" };
      await streamJson<SiteEvent>(workspaceRegionApi(workspaceId, region, "/model-comparison/sites?stream=1"), (event) => {
        if (signal.aborted) return;
        if (event.type === "error") throw new Error(event.message || "区域缓存检查失败");
        if (event.type === "sites") entries.clear();
        for (const item of event.items || []) entries.set(item.site_id, item);
        const items = [...entries.values()].sort((a, b) => a.rank - b.rank
          || (a.display_name || a.site_id).localeCompare(b.display_name || b.site_id, "zh-CN") || a.site_id.localeCompare(b.site_id));
        latest = { items, total: event.total ?? latest.total, checked: event.completed ?? latest.checked,
          complete: event.type === "complete", message: event.message || "正在检查坡度和高置信种子缓存" };
        client.setQueryData(key, latest);
      }, signal);
      if (!latest.complete && !signal.aborted) throw new Error("区域列表传输中断，已显示区域仍可选择，请重试");
      return latest;
    },
  });
}
