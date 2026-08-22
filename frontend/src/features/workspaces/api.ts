import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, patchJson } from "../../api/client";
import type { SiteSourceConflict, TrainingWorkspace } from "../../api/types";

export function workspaceRegionApi(workspaceId: string, scope: string, suffix: string) {
  return `/api/workspaces/${encodeURIComponent(workspaceId)}/regions/${encodeURIComponent(scope)}${suffix}`;
}

export function useWorkspace(workspaceId: string) {
  return useQuery({
    queryKey: ["workspace", workspaceId],
    queryFn: () => getJson<{ workspace: TrainingWorkspace }>(`/api/workspaces/${encodeURIComponent(workspaceId)}`),
    enabled: Boolean(workspaceId),
  });
}

export function useSourceConflicts(workspaceId: string) {
  return useQuery({
    queryKey: ["workspace-conflicts", workspaceId],
    queryFn: () => getJson<{ workspace: TrainingWorkspace; items: SiteSourceConflict[]; total: number }>(`/api/workspaces/${encodeURIComponent(workspaceId)}/source-conflicts`),
    enabled: Boolean(workspaceId),
  });
}

export function useResolveSourceConflict(workspaceId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ siteId, variantIds }: { siteId: string; variantIds: string[] }) => patchJson(`/api/workspaces/${encodeURIComponent(workspaceId)}/sites/${encodeURIComponent(siteId)}/source-variants`, { variant_ids: variantIds }),
    onSuccess: async () => Promise.all([
      client.invalidateQueries({ queryKey: ["workspace-conflicts", workspaceId] }),
      client.invalidateQueries({ queryKey: ["workspace", workspaceId] }),
      client.invalidateQueries({ queryKey: ["training-datasets", workspaceId] }),
      client.invalidateQueries({ queryKey: ["logical-patches", workspaceId] }),
    ]),
  });
}
