import assert from "node:assert/strict";

import { formatPercentText, typeLabel } from "../src/lake_workbench/static/formatters.js";
import { sortModelOptions } from "../src/lake_workbench/static/model-ui.js";
import { filterPatches, normalizePage } from "../src/lake_workbench/static/patch-review-controller.js";
import { buildRouteUrl, parseRoute } from "../src/lake_workbench/static/routing.js";

const route = parseRoute({
  pathname: "/regions/yunnan/training/patches",
  search: "?q=reservoir&has_tci=true",
});
assert.equal(route.region, "yunnan");
assert.equal(route.mode, "training");
assert.equal(route.trainingView, "patches");
assert.equal(route.query, "reservoir");
assert.equal(route.filters.has_tci, "true");

assert.equal(buildRouteUrl({
  region: "all",
  activeRegion: "shaanxi",
  activeId: "shaanxi_1",
  sidebarMode: "model",
  selectedModel: "shaanxi/model/best.pt",
  filters: {},
  query: "",
}), "/regions/all/model/shaanxi_1?model=shaanxi%2Fmodel%2Fbest.pt&lake_region=shaanxi");

assert.equal(formatPercentText(0.125), "12.5%");
assert.equal(typeLabel("reservoir"), "水库");

const models = sortModelOptions([
  { label: "last", best_iou: 0.8, weight: "last.pt" },
  { label: "best", best_iou: 0.8, weight: "best.pt" },
  { label: "lower", best_iou: 0.7, weight: "best.pt" },
]);
assert.deepEqual(models.map((item) => item.label), ["best", "last", "lower"]);

const patches = [{ water_pixels: 0 }, { water_pixels: 12 }];
assert.equal(filterPatches(patches, "water").length, 1);
assert.equal(filterPatches(patches, "negative").length, 1);
assert.equal(filterPatches(patches, "").length, 2);
assert.equal(normalizePage(0, 40, 18), 1);
assert.equal(normalizePage(9, 40, 18), 3);

console.log("frontend module tests passed");
