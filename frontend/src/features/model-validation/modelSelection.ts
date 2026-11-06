import type { ModelOption } from "./api";

export function isRgbNirSlope(model: ModelOption): boolean {
  const config = model.config || {};
  const bands = config.input_bands || config.input_features;
  if (Array.isArray(bands)) {
    const required = ["red", "green", "blue", "nir", "slope_degrees"];
    return bands.length === required.length && required.every((band) => bands.includes(band));
  }
  const indices = config.input_indices;
  return Array.isArray(indices) && indices.length === 5
    && [2, 1, 0, 3, 5].every((source, index) => indices[index] === source)
    && (model.in_channels ?? config.in_channels ?? 5) === 5;
}

function validationIou(model: ModelOption): number {
  // Score this checkpoint, including best_loss.pt, rather than another epoch's maximum.
  for (const score of [model.selected_validation_iou, model.best_iou]) {
    if (typeof score === "number" && Number.isFinite(score)) return score;
  }
  return Number.NEGATIVE_INFINITY;
}

export function preferredValidationModel(items: ModelOption[], fallback = ""): string {
  const available = items.filter((model) => !model.error);
  const slope = available.filter(isRgbNirSlope).sort((a, b) => {
    const aScore = validationIou(a); const bScore = validationIou(b);
    if (aScore !== bScore) return bScore > aScore ? 1 : -1;
    if ((a.weight === "best.pt") !== (b.weight === "best.pt")) return a.weight === "best.pt" ? -1 : 1;
    return a.key.localeCompare(b.key);
  });
  return slope[0]?.key || available.find((model) => model.key === fallback)?.key || available[0]?.key || "";
}
