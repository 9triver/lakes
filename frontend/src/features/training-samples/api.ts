import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteJson, getJson, patchJson } from "../../api/client";
import type { TrainingSample } from "../../api/types";

export function useTrainingSamples(scope: string) {
  return useQuery({ queryKey: ["training-samples", scope], queryFn: async () => (await getJson<{ items: TrainingSample[]; total: number }>(`/api/regions/${encodeURIComponent(scope)}/training-samples`)), enabled: Boolean(scope) });
}

export function useUpdateTrainingSample(scope: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: ({ sample, changes }: { sample: TrainingSample; changes: { split: string; notes: string } }) => patchJson(`/api/regions/${encodeURIComponent(sample.region || scope)}/training-samples/${encodeURIComponent(sample.sample_id)}`, changes), onSuccess: () => client.invalidateQueries({ queryKey: ["training-samples", scope] }) });
}

export function useDeleteTrainingSample(scope: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (sample: TrainingSample) => deleteJson(`/api/regions/${encodeURIComponent(sample.region || scope)}/training-samples/${encodeURIComponent(sample.sample_id)}`), onSuccess: () => client.invalidateQueries({ queryKey: ["training-samples", scope] }) });
}
