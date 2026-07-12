const FILTER_KEYS = [
  "water_type",
  "area_bucket",
  "has_name",
  "has_tci",
  "polygon_quality",
  "metadata_quality",
];

export function parseRoute(location = window.location) {
  const parts = location.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(location.search);
  const route = {
    region: "",
    mode: "lakes",
    lakeId: "",
    trainingView: "samples",
    query: params.get("q") || "",
    model: params.get("model") || "",
    lakeRegion: params.get("lake_region") || "",
    filters: {},
  };
  if (parts[0] === "regions" && parts[1]) {
    route.region = parts[1];
    if (parts[2] === "lakes" && parts[3]) {
      route.lakeId = parts[3];
    } else if (parts[2] === "training") {
      route.mode = "training";
      route.trainingView = ["samples", "patches", "train"].includes(parts[3]) ? parts[3] : "samples";
    } else if (parts[2] === "model") {
      route.mode = "model";
      route.lakeId = parts[3] || "";
    }
  }
  for (const key of FILTER_KEYS) route.filters[key] = params.get(key) || "";
  return route;
}

export function buildRouteUrl(state) {
  let path = `/regions/${encodeURIComponent(state.region)}`;
  if (state.sidebarMode === "training") {
    const view = ["samples", "patches", "train"].includes(state.trainingView) ? state.trainingView : "samples";
    path += `/training/${view}`;
  } else if (state.sidebarMode === "model") {
    path += "/model";
    if (state.activeId) path += `/${encodeURIComponent(state.activeId)}`;
  } else if (state.activeId) {
    path += `/lakes/${encodeURIComponent(state.activeId)}`;
  } else {
    path += "/lakes";
  }
  const params = new URLSearchParams();
  if (state.query) params.set("q", state.query);
  if (state.sidebarMode === "model" && state.selectedModel) params.set("model", state.selectedModel);
  if (state.region === "all" && state.activeRegion && state.activeId) params.set("lake_region", state.activeRegion);
  for (const [key, value] of Object.entries(state.filters || {})) {
    if (value) params.set(key, value);
  }
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}
