interface TerrainEvidence {
  context_id: string;
  cached: boolean;
  model: { input_bands: string[] };
}

export function comparisonTerrainReady(
  contextId: string | undefined,
  asset: string | undefined,
  site: { asset?: string; slope_cached: boolean } | null,
  results: TerrainEvidence[],
): boolean {
  if (!contextId || !asset) return false;
  if (site?.asset === asset && site.slope_cached) return true;
  return results.some((result) => result.context_id === contextId && result.cached === false
    && result.model.input_bands.includes("slope_degrees"));
}
