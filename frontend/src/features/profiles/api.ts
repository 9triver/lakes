import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson, patchJson, postJson } from "../../api/client";
import type { SiteSourceConflict, TrainingProfile } from "../../api/types";

export function profileRegionApi(profileId: string, scope: string, suffix: string) {
  return `/api/profiles/${encodeURIComponent(profileId)}/regions/${encodeURIComponent(scope)}${suffix}`;
}

export function useProfiles(includeArchived = false) {
  return useQuery({
    queryKey: ["profiles", includeArchived],
    queryFn: () => getJson<{ default: string; items: TrainingProfile[] }>(`/api/profiles${includeArchived ? "?include_archived=true" : ""}`),
  });
}

export function useProfile(profileId: string) {
  return useQuery({
    queryKey: ["profile", profileId],
    queryFn: () => getJson<{ profile: TrainingProfile }>(`/api/profiles/${encodeURIComponent(profileId)}`),
    enabled: Boolean(profileId),
  });
}

export function useCreateProfile() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: { name: string; mode: "empty" | "union"; source_profile_ids: string[] }) => postJson<{ profile: TrainingProfile }>("/api/profiles", payload),
    onSuccess: () => client.invalidateQueries({ queryKey: ["profiles"] }),
  });
}

export function useRenameProfile() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => patchJson<{ profile: TrainingProfile }>(`/api/profiles/${encodeURIComponent(id)}`, { name }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["profiles"] }),
  });
}

export function useArchiveProfile() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ id, restore }: { id: string; restore: boolean }) => postJson<{ profile: TrainingProfile }>(`/api/profiles/${encodeURIComponent(id)}/${restore ? "restore" : "archive"}`, {}),
    onSuccess: () => client.invalidateQueries({ queryKey: ["profiles"] }),
  });
}

export function useSourceConflicts(profileId: string) {
  return useQuery({
    queryKey: ["profile-conflicts", profileId],
    queryFn: () => getJson<{ profile: TrainingProfile; items: SiteSourceConflict[]; total: number }>(`/api/profiles/${encodeURIComponent(profileId)}/source-conflicts`),
    enabled: Boolean(profileId),
  });
}

export function useResolveSourceConflict(profileId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ siteId, variantIds }: { siteId: string; variantIds: string[] }) => patchJson(`/api/profiles/${encodeURIComponent(profileId)}/sites/${encodeURIComponent(siteId)}/source-variants`, { variant_ids: variantIds }),
    onSuccess: async () => Promise.all([
      client.invalidateQueries({ queryKey: ["profile-conflicts", profileId] }),
      client.invalidateQueries({ queryKey: ["profiles"] }),
      client.invalidateQueries({ queryKey: ["training-datasets", profileId] }),
      client.invalidateQueries({ queryKey: ["logical-patches", profileId] }),
    ]),
  });
}
