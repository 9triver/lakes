import { useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { TrainingWorkspace } from "../../api/types";

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
