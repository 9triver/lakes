const state = {
  region: "hunan",
  activeRegion: "",
  regions: [],
  lakes: [],
  activeId: null,
  tileMeta: null,
  lake: null,
  loadingId: null,
  total: 0,
  offset: 0,
  limit: 300,
  query: "",
  filters: {},
  metaParts: {},
  imagery: null,
  localLabels: [],
  trainingSamples: [],
  trainingPatches: [],
  trainingDataset: null,
  sidebarMode: "lakes",
  trainingView: "samples",
  trainingLoadId: 0,
  patchIncludeFilter: "",
  patchWaterFilter: "",
  patchPage: 1,
  patchPageSize: 18,
  activePatch: null,
  trainingRuns: [],
  activeTrainingJob: null,
  modelValidation: null,
  modelValidationBusy: false,
  modelValidationRunId: 0,
  modelOptions: [],
  selectedModel: "",
  downloadJobs: new Map(),
  restoringUrl: false,
};

const tabLakesEl = document.querySelector("#tab-lakes");
const tabTrainingEl = document.querySelector("#tab-training");
const tabModelEl = document.querySelector("#tab-model");
const regionSelectEl = document.querySelector("#region-select");
const lakeSidebarPanelEl = document.querySelector("#lake-sidebar-panel");
const trainingSidebarPanelEl = document.querySelector("#training-sidebar-panel");
const modelSidebarPanelEl = document.querySelector("#model-sidebar-panel");
const listEl = document.querySelector("#lake-list");
const trainingListEl = document.querySelector("#training-list");
const countEl = document.querySelector("#count");
const trainingSummaryEl = document.querySelector("#training-summary");
const trainingRefreshEl = document.querySelector("#training-refresh");
const modelSummaryEl = document.querySelector("#model-summary");
const modelSelectEl = document.querySelector("#model-select");
const modelDetailEl = document.querySelector("#model-detail");
const modelRandomEl = document.querySelector("#model-random");
const trainingViewSamplesEl = document.querySelector("#training-view-samples");
const trainingViewPatchesEl = document.querySelector("#training-view-patches");
const trainingViewTrainEl = document.querySelector("#training-view-train");
const patchControlsEl = document.querySelector("#patch-controls");
const patchExportControlsEl = document.querySelector("#patch-export-controls");
const patchSizeEl = document.querySelector("#patch-size");
const patchStrideEl = document.querySelector("#patch-stride");
const patchPreviewScaleEl = document.querySelector("#patch-preview-scale");
const patchOverwriteEl = document.querySelector("#patch-overwrite");
const patchExportEl = document.querySelector("#patch-export");
const patchExportStatusEl = document.querySelector("#patch-export-status");
const patchIncludeFilterEl = document.querySelector("#patch-include-filter");
const patchWaterFilterEl = document.querySelector("#patch-water-filter");
const trainingRunControlsEl = document.querySelector("#training-run-controls");
const trainRunNameEl = document.querySelector("#train-run-name");
const trainEpochsEl = document.querySelector("#train-epochs");
const trainBatchSizeEl = document.querySelector("#train-batch-size");
const trainLrEl = document.querySelector("#train-lr");
const trainBaseChannelsEl = document.querySelector("#train-base-channels");
const trainDeviceEl = document.querySelector("#train-device");
const trainNoAugmentEl = document.querySelector("#train-no-augment");
const trainStartEl = document.querySelector("#train-start");
const trainCancelEl = document.querySelector("#train-cancel");
const trainStatusEl = document.querySelector("#train-status");
const searchEl = document.querySelector("#search");
const filterTypeEl = document.querySelector("#filter-type");
const filterAreaEl = document.querySelector("#filter-area");
const filterNameEl = document.querySelector("#filter-name");
const filterTciEl = document.querySelector("#filter-tci");
const filterPolygonQualityEl = document.querySelector("#filter-polygon-quality");
const filterMetadataQualityEl = document.querySelector("#filter-metadata-quality");
const loadMoreEl = document.querySelector("#load-more");
const titleEl = document.querySelector("#lake-title");
const subtitleEl = document.querySelector("#lake-subtitle");
const metaEl = document.querySelector("#meta");
const emptyEl = document.querySelector("#empty");
const toolbarEl = document.querySelector(".toolbar");
const mapWrapEl = document.querySelector("#map-wrap");
const toolsEl = document.querySelector(".tools");
const modelPredictionToolEl = document.querySelector("#model-prediction-tool");
const mapEl = document.querySelector("#map");
const loadingEl = document.querySelector("#loading");
const loadingTextEl = document.querySelector("#loading-text");
const patchReviewEl = document.querySelector("#patch-review");
const patchReviewSummaryEl = document.querySelector("#patch-review-summary");
const patchGridEl = document.querySelector("#patch-grid");
const patchPrevEl = document.querySelector("#patch-prev");
const patchNextEl = document.querySelector("#patch-next");
const patchPageLabelEl = document.querySelector("#patch-page-label");
const patchModalEl = document.querySelector("#patch-modal");
const patchModalCloseEl = document.querySelector("#patch-modal-close");
const patchModalTitleEl = document.querySelector("#patch-modal-title");
const patchModalSubtitleEl = document.querySelector("#patch-modal-subtitle");
const patchModalImageEl = document.querySelector("#patch-modal-image");
const patchModalMetaEl = document.querySelector("#patch-modal-meta");
const patchModalToggleEl = document.querySelector("#patch-modal-toggle");
const trainingRunViewEl = document.querySelector("#training-run-view");
const trainingRunSummaryEl = document.querySelector("#training-run-summary");
const trainingRunBodyEl = document.querySelector("#training-run-body");
const toggleImageEl = document.querySelector("#toggle-image");
const toggleTileGridEl = document.querySelector("#toggle-tile-grid");
const toggleOsmEl = document.querySelector("#toggle-osm");
const toggleHydroEl = document.querySelector("#toggle-hydro");
const toggleContextWaterEl = document.querySelector("#toggle-context-water");
const toggleEsaEl = document.querySelector("#toggle-esa");
const toggleJrcEl = document.querySelector("#toggle-jrc");
const toggleLocalLabelEl = document.querySelector("#toggle-local-label");
const toggleModelPredictionEl = document.querySelector("#toggle-model-prediction");
const localLabelSelectEl = document.querySelector("#local-label-select");
const jrcThresholdEl = document.querySelector("#jrc-threshold");
const jrcThresholdValueEl = document.querySelector("#jrc-threshold-value");
const lakeControlsEl = document.querySelector("#lake-controls");
const sentinelPanelEl = document.querySelector("#sentinel-panel");
const sentinelTileEl = document.querySelector("#sentinel-tile");
const imageryProductEl = document.querySelector("#imagery-product");
const imageryApplyEl = document.querySelector("#imagery-apply");
const sentinelStartEl = document.querySelector("#sentinel-start");
const sentinelEndEl = document.querySelector("#sentinel-end");
const sentinelCloudEl = document.querySelector("#sentinel-cloud");
const sentinelQueryEl = document.querySelector("#sentinel-query");
const sentinelProductsEl = document.querySelector("#sentinel-products");
const trainingPanelEl = document.querySelector("#training-panel");
const trainingNotesEl = document.querySelector("#training-notes");
const trainingSaveEl = document.querySelector("#training-save");
const trainingStatusEl = document.querySelector("#training-status");
const zoomLakeEl = document.querySelector("#zoom-lake");
const zoomTileEl = document.querySelector("#zoom-tile");

let searchTimer = null;
let jrcTimer = null;

function syncLakeControlsVisibility() {
  lakeControlsEl.hidden = sentinelPanelEl.hidden && trainingPanelEl.hidden;
}

for (const panel of [sentinelPanelEl, trainingPanelEl]) {
  new MutationObserver(syncLakeControlsVisibility).observe(panel, { attributes: true, attributeFilter: ["hidden"] });
}
syncLakeControlsVisibility();

setDefaultSentinelFilters();

const rasterLayer = new ol.layer.Tile({ visible: true });
const vectorSources = {
  tileGrid: new ol.source.Vector(),
  osm: new ol.source.Vector(),
  hydrolakes: new ol.source.Vector(),
  contextOsm: new ol.source.Vector(),
  contextHydrolakes: new ol.source.Vector(),
  esa: new ol.source.Vector(),
  jrc: new ol.source.Vector(),
  localLabel: new ol.source.Vector(),
  modelPrediction: new ol.source.Vector(),
};
const vectorLayers = {
  tileGrid: new ol.layer.Vector({ source: vectorSources.tileGrid, style: tileGridStyle }),
  osm: new ol.layer.Vector({ source: vectorSources.osm, style: polygonStyle("#00a6ff", "rgba(0, 166, 255, 0.20)") }),
  hydrolakes: new ol.layer.Vector({ source: vectorSources.hydrolakes, style: polygonStyle("#ffd447", "rgba(255, 212, 71, 0.18)") }),
  contextOsm: new ol.layer.Vector({ source: vectorSources.contextOsm, style: polygonStyle("#0088cc", "rgba(0, 136, 204, 0.06)", [6, 5], 1.4) }),
  contextHydrolakes: new ol.layer.Vector({ source: vectorSources.contextHydrolakes, style: polygonStyle("#b28b00", "rgba(255, 212, 71, 0.06)", [6, 5], 1.4) }),
  esa: new ol.layer.Vector({ source: vectorSources.esa, style: polygonStyle("#ff4fb3", "rgba(255, 79, 179, 0.30)") }),
  jrc: new ol.layer.Vector({ source: vectorSources.jrc, style: polygonStyle("#1ab878", "rgba(44, 214, 137, 0.24)") }),
  localLabel: new ol.layer.Vector({ source: vectorSources.localLabel, style: polygonStyle("#ffffff", "rgba(0, 0, 0, 0.08)", [8, 4], 2.5) }),
  modelPrediction: new ol.layer.Vector({ source: vectorSources.modelPrediction, style: polygonStyle("#f03a47", "rgba(240, 58, 71, 0.24)", undefined, 2.6) }),
};
const map = new ol.Map({
  target: mapEl,
  layers: [
    rasterLayer,
    vectorLayers.tileGrid,
    vectorLayers.contextOsm,
    vectorLayers.contextHydrolakes,
    vectorLayers.osm,
    vectorLayers.hydrolakes,
    vectorLayers.esa,
    vectorLayers.jrc,
    vectorLayers.localLabel,
    vectorLayers.modelPrediction,
  ],
  view: new ol.View({
    center: ol.proj.fromLonLat([112.5, 28.8]),
    zoom: 8,
    minZoom: 5,
    maxZoom: 16,
  }),
});
const geojson = new ol.format.GeoJSON({
  dataProjection: "EPSG:4326",
  featureProjection: "EPSG:3857",
});

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error || response.statusText);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function patchJson(url, payload) {
  const response = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function deleteJson(url) {
  const response = await fetch(url, { method: "DELETE" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function apiPath(path) {
  return apiPathFor(state.region, path);
}

function apiPathFor(region, path) {
  return `/api/regions/${encodeURIComponent(region)}${path}`;
}

function isAllRegions() {
  return state.region === "all";
}

function activeRegionKey() {
  return state.activeRegion || state.region;
}

function activeApiPath(path) {
  return apiPathFor(activeRegionKey(), path);
}

function localModelKeyForActiveRegion(modelKey = state.selectedModel) {
  if (!isAllRegions()) return modelKey || "";
  const prefix = `${activeRegionKey()}/`;
  return modelKey?.startsWith(prefix) ? modelKey.slice(prefix.length) : modelKey || "";
}

function parseRoute() {
  const parts = window.location.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(window.location.search);
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
      route.mode = "lakes";
      route.lakeId = parts[3];
    } else if (parts[2] === "training") {
      route.mode = "training";
      route.trainingView = ["samples", "patches", "train"].includes(parts[3]) ? parts[3] : "samples";
    } else if (parts[2] === "model") {
      route.mode = "model";
      route.lakeId = parts[3] || "";
    }
  }
  for (const key of ["water_type", "area_bucket", "has_name", "has_tci", "polygon_quality", "metadata_quality"]) {
    route.filters[key] = params.get(key) || "";
  }
  return route;
}

function buildRouteUrl() {
  const region = encodeURIComponent(state.region);
  let path = `/regions/${region}`;
  if (state.sidebarMode === "training") {
    path += `/training/${["samples", "patches", "train"].includes(state.trainingView) ? state.trainingView : "samples"}`;
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
  if (isAllRegions() && state.activeRegion && state.activeId) params.set("lake_region", state.activeRegion);
  for (const [key, value] of Object.entries(state.filters)) {
    if (value) params.set(key, value);
  }
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

function updateRouteUrl({ replace = false } = {}) {
  if (state.restoringUrl) return;
  const url = buildRouteUrl();
  if (url === `${window.location.pathname}${window.location.search}`) return;
  history[replace ? "replaceState" : "pushState"]({}, "", url);
}

async function loadRegions() {
  const payload = await fetchJson("/api/regions");
  const route = parseRoute();
  state.regions = payload.items || [];
  state.region = route.region === "all"
    ? "all"
    : route.region && state.regions.some((region) => region.key === route.region)
    ? route.region
    : payload.default || state.regions[0]?.key || state.region;
  state.query = route.query;
  state.filters = route.filters;
  state.sidebarMode = route.mode;
  state.trainingView = route.trainingView;
  searchEl.value = state.query;
  applyFilterControls();
  renderRegions();
  resetSelection();
  return route;
}

function renderRegions() {
  regionSelectEl.replaceChildren();
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "全部";
  allOption.selected = state.region === "all";
  regionSelectEl.append(allOption);
  for (const region of state.regions) {
    const option = document.createElement("option");
    option.value = region.key;
    option.textContent = region.ready ? region.name : `${region.name}（未准备）`;
    option.selected = region.key === state.region;
    regionSelectEl.append(option);
  }
}

function currentRegion() {
  if (isAllRegions()) {
    return { key: "all", name: "全部区域", bounds: null, ready: true };
  }
  return state.regions.find((region) => region.key === state.region);
}

async function switchRegion(regionKey) {
  if (!regionKey || regionKey === state.region) return;
  state.region = regionKey;
  state.activeRegion = regionKey === "all" ? "" : regionKey;
  state.downloadJobs.clear();
  state.modelOptions = [];
  state.selectedModel = "";
  state.trainingRuns = [];
  state.activeTrainingJob = null;
  resetSelection();
  await loadLakes();
  if (state.sidebarMode === "training") await loadActiveTrainingView();
  const region = currentRegion();
  if (region?.bounds) fitToBounds(region.bounds);
  updateRouteUrl();
}

function resetSelection() {
  state.activeId = null;
  state.activeRegion = isAllRegions() ? "" : state.region;
  state.tileMeta = null;
  state.lake = null;
  state.loadingId = null;
  state.metaParts = {};
  state.imagery = null;
  state.localLabels = [];
  state.modelValidation = null;
  if (!state.modelOptions.length) state.selectedModel = "";
  rasterLayer.setSource(null);
  clearVectorLayers();
  sentinelPanelEl.hidden = true;
  trainingPanelEl.hidden = true;
  modelSummaryEl.textContent = "模型验证未运行";
  modelSelectEl.replaceChildren();
  modelSelectEl.disabled = true;
  modelDetailEl.textContent = "";
  sentinelProductsEl.replaceChildren();
  imageryProductEl.replaceChildren();
  resetLocalLabelSelect();
  renderTrainingReadiness();
  const region = currentRegion();
  titleEl.textContent = "选择一个湖泊";
  subtitleEl.textContent = region?.name ? `${region.name}水体和外部标注会在这里显示` : "可见光影像和 polygon 会在这里显示";
  metaEl.textContent = region?.load_error || "";
  emptyEl.hidden = false;
  mapEl.hidden = false;
  setLoading(false);
}

async function loadLakes({ append = false } = {}) {
  if (!append) {
    state.offset = 0;
    state.lakes = [];
  }
  countEl.textContent = "加载中";
  const params = new URLSearchParams({
    limit: String(state.limit),
    offset: String(state.offset),
    q: state.query,
  });
  for (const [key, value] of Object.entries(state.filters)) {
    if (value) params.set(key, value);
  }
  const payload = await fetchJson(apiPath(`/lakes?${params.toString()}`));
  state.total = payload.total;
  state.lakes = append ? state.lakes.concat(payload.items) : payload.items;
  state.offset = state.lakes.length;
  const region = currentRegion();
  countEl.textContent = isAllRegions()
    ? `${payload.total} 个湖泊，显示 ${state.lakes.length} 个`
    : region?.load_error
    ? region.load_error
    : `${payload.total} 个湖泊，显示 ${state.lakes.length} 个`;
  loadMoreEl.hidden = state.sidebarMode !== "lakes" || state.lakes.length >= payload.total;
  renderList();
}

function isCurrentTrainingLoad(loadId, view) {
  return loadId === state.trainingLoadId && state.sidebarMode === "training" && state.trainingView === view;
}

async function loadTrainingSamples(loadId = null) {
  loadId = loadId ?? ++state.trainingLoadId;
  if (isCurrentTrainingLoad(loadId, "samples")) {
    trainingListEl.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "empty-list";
    loading.textContent = "训练样本加载中";
    trainingListEl.append(loading);
    trainingSummaryEl.textContent = "训练集加载中";
  }
  const payload = await fetchJson(apiPath("/training-samples"));
  if (!isCurrentTrainingLoad(loadId, "samples")) return;
  state.trainingSamples = payload.items || [];
  const bad = state.trainingSamples.filter((item) => item.status !== "ok").length;
  trainingSummaryEl.textContent = bad
    ? `${payload.total} 个样本，${bad} 个缺文件`
    : `${payload.total} 个样本`;
  renderTrainingSamples();
}

async function loadTrainingPatches(loadId = null) {
  loadId = loadId ?? ++state.trainingLoadId;
  if (isCurrentTrainingLoad(loadId, "patches")) {
    patchReviewSummaryEl.textContent = "Patch 加载中";
    patchGridEl.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "patch-empty";
    loading.textContent = "Patch 加载中";
    patchGridEl.append(loading);
    trainingSummaryEl.textContent = "Patch 加载中";
  }
  const params = new URLSearchParams();
  if (state.patchIncludeFilter) params.set("include", state.patchIncludeFilter);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const payload = await fetchJson(apiPath(`/training-patches${suffix}`));
  if (!isCurrentTrainingLoad(loadId, "patches")) return;
  state.trainingPatches = payload.items || [];
  trainingSummaryEl.textContent = `${payload.total} 个 patch，包含 ${payload.included_count || 0}，排除 ${payload.excluded_count || 0}`;
  normalizePatchPage();
  renderPatchReview();
}

async function loadTrainingRuns(loadId = null) {
  loadId = loadId ?? ++state.trainingLoadId;
  if (isCurrentTrainingLoad(loadId, "train")) {
    trainingRunSummaryEl.textContent = "训练任务加载中";
    trainingRunBodyEl.innerHTML = `<div class="training-run-empty">训练任务加载中</div>`;
    trainingSummaryEl.textContent = "训练任务加载中";
  }
  const payload = await fetchJson(apiPath("/training-runs"));
  if (!isCurrentTrainingLoad(loadId, "train")) return;
  state.trainingDataset = payload.dataset || null;
  state.trainingRuns = payload.items || [];
  const running = state.trainingRuns.find((job) => ["queued", "configured", "running", "cancel_requested"].includes(job.status));
  state.activeTrainingJob = running || state.trainingRuns[0] || state.activeTrainingJob;
  trainingSummaryEl.textContent = running
    ? `训练中：${running.message || running.status}`
    : `${state.trainingRuns.length} 个训练任务`;
  renderTrainingRunView();
}

async function loadModelOptions(preferred = state.selectedModel) {
  modelSummaryEl.textContent = "模型列表加载中";
  const payload = await fetchJson(apiPath("/model-validation/models"));
  state.modelOptions = sortModelOptions(payload.items || []);
  state.selectedModel = preferred && state.modelOptions.some((item) => item.key === preferred)
    ? preferred
    : payload.default || state.modelOptions[0]?.key || "";
  renderModelOptions();
  modelSummaryEl.textContent = state.modelOptions.length
    ? `已加载 ${state.modelOptions.length} 个模型权重`
    : "当前区域没有模型权重";
}

function renderModelOptions() {
  modelSelectEl.replaceChildren();
  if (!state.modelOptions.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无可用模型";
    modelSelectEl.append(option);
    modelSelectEl.disabled = true;
    modelRandomEl.disabled = true;
    modelDetailEl.textContent = "无可用模型";
    return;
  }
  modelSelectEl.disabled = false;
  modelRandomEl.disabled = state.modelValidationBusy;
  for (const item of state.modelOptions) {
    const option = document.createElement("option");
    option.value = item.key;
    option.textContent = item.error
      ? `${item.label}（不可用）`
      : formatModelOptionText(item);
    option.disabled = Boolean(item.error);
    option.selected = item.key === state.selectedModel;
    option.title = item.error || item.path || item.label;
    modelSelectEl.append(option);
  }
  modelSelectEl.value = state.selectedModel;
  renderSelectedModelDetail();
}

function sortModelOptions(items) {
  return [...items].sort((a, b) => {
    const aScore = Number(a.best_iou);
    const bScore = Number(b.best_iou);
    const aHasScore = Number.isFinite(aScore);
    const bHasScore = Number.isFinite(bScore);
    if (Boolean(a.error) !== Boolean(b.error)) return a.error ? 1 : -1;
    if (aHasScore !== bHasScore) return aHasScore ? -1 : 1;
    if (aHasScore && bHasScore && bScore !== aScore) return bScore - aScore;
    if ((a.weight === "best.pt") !== (b.weight === "best.pt")) return a.weight === "best.pt" ? -1 : 1;
    return String(a.label || "").localeCompare(String(b.label || ""), "zh-CN");
  });
}

function formatModelOptionText(item) {
  const parts = [item.label];
  const score = Number(item.best_iou);
  if (Number.isFinite(score)) parts.push(`best IoU ${formatNumber(score, 4)}`);
  if (item.epoch) parts.push(`epoch ${item.epoch}`);
  if (item.in_channels) parts.push(`${item.in_channels} band`);
  return parts.filter(Boolean).join(" · ");
}

function renderSelectedModelDetail() {
  const model = state.modelOptions.find((item) => item.key === state.selectedModel);
  modelDetailEl.replaceChildren();
  if (!model) {
    modelDetailEl.textContent = "未选择模型";
    return;
  }
  const title = document.createElement("div");
  title.className = "model-detail-title";
  title.innerHTML = `<strong>${escapeHtml(model.name || model.label || model.key)}</strong><span>${escapeHtml(model.weight || "")}</span>`;
  modelDetailEl.append(title);
  if (model.error) {
    const error = document.createElement("div");
    error.className = "model-detail-error";
    error.textContent = model.error;
    modelDetailEl.append(error);
    return;
  }
  const latest = model.latest || {};
  const train = latest.train || {};
  const val = latest.val || {};
  const dataset = model.dataset || {};
  const config = model.config || {};
  const fields = [
    ["范围", model.scope === "all" ? "全部区域" : currentRegionName(model.scope || model.region)],
    ["best IoU", Number.isFinite(Number(model.best_iou)) ? formatNumber(Number(model.best_iou), 4) : ""],
    ["best epoch", model.best_epoch || ""],
    ["最新 val IoU", Number.isFinite(Number(val.iou)) ? formatNumber(Number(val.iou), 4) : ""],
    ["最新 train IoU", Number.isFinite(Number(train.iou)) ? formatNumber(Number(train.iou), 4) : ""],
    ["epoch", model.epoch ? `${model.epoch}${config.epochs ? ` / ${config.epochs}` : ""}` : ""],
    ["输入", [model.in_channels ? `${model.in_channels} band` : "", model.base_channels ? `宽度 ${model.base_channels}` : ""].filter(Boolean).join(" · ")],
    ["训练 Patch", dataset.usable_patches != null ? `${dataset.usable_patches} 可用 / ${dataset.included_patches || 0} 包含` : ""],
    ["样本/水体", dataset.sample_count != null ? `${dataset.sample_count || 0} / ${dataset.lake_count || 0}` : ""],
    ["训练/验证", config.train_count != null || config.val_count != null ? `${config.train_count || 0} / ${config.val_count || 0}` : ""],
    ["更新时间", formatDateTimeText(model.updated_at)],
    ["路径", model.path || ""],
  ];
  const grid = document.createElement("div");
  grid.className = "model-detail-grid";
  for (const [label, value] of fields) {
    if (!value) continue;
    const row = document.createElement("div");
    row.className = "model-detail-row";
    row.innerHTML = `<span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong>`;
    grid.append(row);
  }
  modelDetailEl.append(grid);
}

function setSidebarMode(mode, options = {}) {
  const previousMode = state.sidebarMode;
  state.sidebarMode = mode;
  const trainingMode = mode === "training";
  const modelMode = mode === "model";
  const lakesMode = mode === "lakes";
  tabLakesEl.classList.toggle("active", lakesMode);
  tabTrainingEl.classList.toggle("active", trainingMode);
  tabModelEl.classList.toggle("active", modelMode);
  lakeSidebarPanelEl.hidden = !lakesMode;
  listEl.hidden = !lakesMode;
  loadMoreEl.hidden = !lakesMode || state.lakes.length >= state.total;
  trainingSidebarPanelEl.hidden = !trainingMode;
  modelSidebarPanelEl.hidden = !modelMode;
  renderTrainingViewMode();
  if (!trainingMode && previousMode === "training") state.trainingLoadId += 1;
  if (trainingMode && options.load !== false) loadActiveTrainingView().catch(showError);
  if (modelMode && !state.modelOptions.length) loadModelOptions().catch(showError);
  updateRouteUrl();
}

function setTrainingView(view, options = {}) {
  const validView = ["samples", "patches", "train"].includes(view) ? view : "samples";
  const changed = state.trainingView !== validView;
  state.trainingView = validView;
  if (changed) state.trainingLoadId += 1;
  renderTrainingViewMode();
  if (state.sidebarMode === "training" && options.load !== false && changed) loadActiveTrainingView().catch(showError);
  updateRouteUrl();
}

function renderTrainingViewMode() {
  const trainingMode = state.sidebarMode === "training";
  const modelMode = state.sidebarMode === "model";
  const samplesMode = state.trainingView === "samples";
  const patchMode = trainingMode && state.trainingView === "patches";
  const trainMode = trainingMode && state.trainingView === "train";
  trainingViewSamplesEl.classList.toggle("active", samplesMode);
  trainingViewPatchesEl.classList.toggle("active", state.trainingView === "patches");
  trainingViewTrainEl.classList.toggle("active", state.trainingView === "train");
  trainingListEl.hidden = !trainingMode || !samplesMode;
  patchControlsEl.hidden = !patchMode;
  patchExportControlsEl.hidden = !patchMode;
  trainingRunControlsEl.hidden = !trainMode;
  patchReviewEl.hidden = !patchMode;
  trainingRunViewEl.hidden = !trainMode;
  toolbarEl.hidden = patchMode || trainMode;
  mapWrapEl.hidden = patchMode || trainMode;
  toolsEl.hidden = patchMode || trainMode;
  modelPredictionToolEl.hidden = !modelMode;
  if (!modelMode) vectorLayers.modelPrediction.setVisible(false);
  if (patchMode) {
    titleEl.textContent = "Patch 审核";
    subtitleEl.textContent = "浏览训练 patch 并标记包含或排除";
    sentinelPanelEl.hidden = true;
    trainingPanelEl.hidden = true;
    metaEl.textContent = "Patch 审核";
  } else if (trainMode) {
    titleEl.textContent = "模型训练";
    subtitleEl.textContent = `当前训练范围：${currentRegion()?.name || "当前区域"}`;
    sentinelPanelEl.hidden = true;
    trainingPanelEl.hidden = true;
    metaEl.textContent = "模型训练";
    renderTrainingRunView();
  } else if (state.lake) {
    renderActiveLakeTitle();
    sentinelPanelEl.hidden = modelMode || !(state.imagery?.tiles?.length);
    trainingPanelEl.hidden = modelMode || !state.imagery;
    renderMeta();
    ensureMapVisible();
  } else {
    toolsEl.hidden = false;
    mapWrapEl.hidden = false;
    ensureMapVisible();
  }
}

function isTrainingWorkspaceView() {
  return state.sidebarMode === "training" && ["patches", "train"].includes(state.trainingView);
}

async function loadActiveTrainingView() {
  if (state.sidebarMode !== "training") return;
  const loadId = ++state.trainingLoadId;
  if (state.trainingView === "patches") {
    await loadTrainingPatches(loadId);
    return;
  }
  if (state.trainingView === "train") {
    await loadTrainingRuns(loadId);
    return;
  }
  await loadTrainingSamples(loadId);
}

async function runRandomModelValidation() {
  if (state.modelValidationBusy) return;
  const runId = state.modelValidationRunId + 1;
  state.modelValidationRunId = runId;
  state.modelValidationBusy = true;
  modelRandomEl.disabled = true;
  modelRandomEl.textContent = "验证中";
  modelSummaryEl.textContent = "模型推理中";
  setLoading(true, "模型推理中");
  try {
    rasterLayer.setSource(null);
    toggleImageEl.checked = true;
    rasterLayer.setVisible(true);
    toggleModelPredictionEl.checked = true;
    vectorLayers.modelPrediction.setVisible(true);
    const params = new URLSearchParams();
    if (state.selectedModel) params.set("model", state.selectedModel);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    let payload;
    try {
      payload = await fetchJson(apiPath(`/model-validation/random${suffix}`));
    } catch (error) {
      if (error.status === 429) {
        modelSummaryEl.textContent = error.message || "模型推理正在运行，请稍后再试";
        return;
      }
      if (error instanceof TypeError) {
        modelSummaryEl.textContent = "服务暂时不可用，请稍后再试";
        return;
      }
      throw error;
    }
    if (runId !== state.modelValidationRunId) return;
    state.modelValidation = payload;
    setSidebarMode("model");
    state.activeRegion = payload.lake?.region || payload.region || activeRegionKey();
    await selectLake(payload.lake_id, { waitForTile: false, validationRunId: runId });
    if (runId !== state.modelValidationRunId) return;
    if (state.activeId !== payload.lake_id) return;
    applyModelPrediction(payload);
    updateRouteUrl();
  } finally {
    if (runId === state.modelValidationRunId) {
      state.modelValidationBusy = false;
      modelRandomEl.disabled = false;
      modelRandomEl.textContent = "随机验证一个湖泊";
      setLoading(false);
    }
  }
}

function applyModelPrediction(payload) {
  if (!payload?.prediction) return;
  addFeatureCollection("modelPrediction", payload.prediction);
  toggleModelPredictionEl.checked = true;
  vectorLayers.modelPrediction.setVisible(true);
  fitToPredictionOrLake(payload);
  const lakeName = payload.lake?.display_name || payload.lake?.name || payload.lake_id;
  const stats = payload.stats || {};
  const model = payload.model || {};
  const modelKey = isAllRegions() && model.key && !model.key.includes("/")
    ? `${activeRegionKey()}/${model.key}`
    : model.key;
  if (modelKey && modelKey !== state.selectedModel) {
    state.selectedModel = modelKey;
    if (state.modelOptions.length) renderModelOptions();
  }
  const count = payload.prediction?.features?.length || 0;
  const area = Number(stats.area_km2 || 0);
  const ratio = Number(stats.predicted_ratio || 0) * 100;
  modelSummaryEl.textContent = `${lakeName} · ${count} 个预测斑块 · ${formatNumber(area, 3)} km²`;
  state.metaParts.model = [
    `模型预测 ${model.name || ""}`,
    `阈值 ${formatNumber(Number(model.threshold ?? stats.threshold ?? 0.5), 2)}`,
    `水体像元 ${formatNumber(ratio, 1)}%`,
    model.device ? `设备 ${model.device}` : "",
  ].filter(Boolean).join(" | ");
  renderMeta();
}

function fitToPredictionOrLake(payload) {
  const extent = vectorSources.modelPrediction.getExtent();
  if (extent && extent.every(Number.isFinite) && !ol.extent.isEmpty(extent)) {
    ensureMapVisible();
    map.getView().fit(extent, {
      padding: [48, 48, 48, 48],
      duration: 180,
      maxZoom: 14,
    });
    return;
  }
  const bounds = payload?.lake?.bbox || state.lake?.bbox || state.tileMeta?.lake_bounds;
  if (bounds) fitToBounds(bounds);
}

function renderList() {
  listEl.replaceChildren();
  for (const lake of state.lakes) {
    const button = document.createElement("button");
    button.className = `lake-item${state.activeId === lake.object_id ? " active" : ""}${state.loadingId === lake.object_id ? " loading" : ""}`;
    button.type = "button";
    button.addEventListener("click", () => openLakeFromList(lake).catch(showError));
    const label = lake.display_name || lake.name || lake.object_id;
    button.innerHTML = `
      <div class="lake-row">
        <div class="lake-id">${escapeHtml(label)}</div>
        <div class="badge">${escapeHtml(isAllRegions() ? lake.region_name || lake.region || "" : typeLabel(lake.water_type))}${lake.has_tci ? " · TCI" : ""}</div>
      </div>
      <div class="lake-detail">
        <span>面积 ${formatNumber(lake.area_km2, 2)} km²</span>
        <span>tile ${escapeHtml((lake.tiles || []).slice(0, 3).join(", "))}</span>
        <span>${formatNumber(lake.center[0], 4)}, ${formatNumber(lake.center[1], 4)}</span>
        <span>${escapeHtml(lake.best_tci_date || "")} · ${escapeHtml(lake.metadata_quality || "")}</span>
      </div>
    `;
    listEl.append(button);
  }
}

async function openLakeFromList(lake) {
  state.activeRegion = lake.region || state.region;
  await selectLake(lake.object_id);
}

function renderTrainingSamples() {
  trainingListEl.replaceChildren();
  if (!state.trainingSamples.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "暂无训练样本";
    trainingListEl.append(empty);
    return;
  }
  for (const sample of state.trainingSamples) {
    const item = document.createElement("div");
    item.className = `training-item${sample.status === "ok" ? "" : " missing"}`;
    const name = sample.lake_display_name || sample.lake_name || sample.lake_id || sample.sample_id;
    item.innerHTML = `
      <div class="training-top">
        <div class="training-name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
        <div class="badge">${escapeHtml(sample.status === "ok" ? "ok" : "缺文件")}</div>
      </div>
      <div class="training-meta-line">${escapeHtml(formatTrainingLabelSource(sample.label_source, sample.label_threshold))} · ${escapeHtml(formatLabelScope(sample.label_scope))} · ${escapeHtml(formatMaskPolicy(sample.mask_policy))}</div>
      <div class="training-meta-line">${escapeHtml(sample.tile_count || 0)} tile · ${escapeHtml(formatImageryAssetLabels(sample.imagery_asset_labels || sample.imagery_asset_label))} · ${escapeHtml(sample.product_date || "")}</div>
      <div class="training-meta-line" title="${escapeHtml(sample.sample_id || "")}">${escapeHtml(sample.sample_id || "")}</div>
    `;
    const edit = document.createElement("div");
    edit.className = "training-edit";
    const split = makeSelect(["", "train", "val", "test"], sample.split || "");
    const notes = document.createElement("input");
    notes.type = "text";
    notes.value = sample.notes || "";
    notes.placeholder = "备注";
    edit.append(split, notes);

    const actions = document.createElement("div");
    actions.className = "training-actions";
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "定位";
    open.addEventListener("click", () => {
      state.activeRegion = sample.region || state.region;
      selectLake(sample.lake_id).catch(showError);
    });
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "保存";
    save.addEventListener("click", () => updateTrainingSample(sample.sample_id, {
      split: split.value,
      notes: notes.value,
    }).catch(showError));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "删除";
    remove.addEventListener("click", () => deleteTrainingSample(sample.sample_id).catch(showError));
    actions.append(open, save, remove);
    item.append(edit, actions);
    trainingListEl.append(item);
  }
}

function renderPatchReview() {
  patchGridEl.replaceChildren();
  const patches = filteredTrainingPatches();
  normalizePatchPage(patches.length);
  const pageCount = Math.max(1, Math.ceil(patches.length / state.patchPageSize));
  const start = (state.patchPage - 1) * state.patchPageSize;
  const pageItems = patches.slice(start, start + state.patchPageSize);
  patchReviewSummaryEl.textContent = `${patches.length} 个 patch，当前第 ${state.patchPage} 页`;
  patchPageLabelEl.textContent = `${state.patchPage} / ${pageCount}`;
  patchPrevEl.disabled = state.patchPage <= 1;
  patchNextEl.disabled = state.patchPage >= pageCount;

  if (!pageItems.length) {
    const empty = document.createElement("div");
    empty.className = "patch-empty";
    empty.textContent = isAllRegions()
      ? "暂无 patch，请先生成 patch"
      : "当前区域暂无 patch，请先生成 patch，或切换到“全部”查看已有区域的 patch";
    patchGridEl.append(empty);
    return;
  }

  for (const patch of pageItems) {
    const item = document.createElement("article");
    item.className = `patch-card${patch.included ? "" : " excluded"}${patch.preview_exists ? "" : " missing"}`;
    const name = patch.lake_display_name || patch.lake_name || patch.lake_id || patch.sample_id;
    const validRatio = formatPercentText(patch.valid_ratio);
    const waterRatio = formatPercentText(patch.water_ratio_valid);
    item.innerHTML = `
      <button class="patch-card-image" type="button" aria-label="打开 patch 预览">${patch.preview_url ? `<img src="${escapeHtml(patch.preview_url)}" alt="" loading="lazy" />` : ""}</button>
      <div class="patch-card-body">
        <div class="patch-card-top">
          <strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong>
          <span class="badge">${patch.included ? "include" : "exclude"}</span>
        </div>
        <div class="patch-card-meta">${escapeHtml(patch.patch_id || "")}</div>
        <div class="patch-card-stats">
          <span>water ${escapeHtml(waterRatio)}</span>
          <span>valid ${escapeHtml(validRatio)}</span>
          <span>ignore ${escapeHtml(patch.ignore_pixels || 0)}</span>
        </div>
      </div>
    `;
    item.querySelector(".patch-card-image").addEventListener("click", () => openPatchModal(patch));
    const actions = document.createElement("div");
    actions.className = "patch-card-actions";
    const include = document.createElement("button");
    include.type = "button";
    include.textContent = patch.included ? "排除" : "恢复包含";
    include.addEventListener("click", () => updateTrainingPatch(patch.patch_id, { include: !patch.included }).catch(showError));
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "定位水体";
    open.addEventListener("click", () => {
      setTrainingView("samples");
      state.activeRegion = patch.region || state.region;
      selectLake(patch.lake_id).catch(showError);
    });
    actions.append(include, open);
    item.append(actions);
    patchGridEl.append(item);
  }
}

function filteredTrainingPatches() {
  return state.trainingPatches.filter((patch) => {
    if (state.patchWaterFilter === "water" && Number(patch.water_pixels || 0) <= 0) return false;
    if (state.patchWaterFilter === "negative" && Number(patch.water_pixels || 0) > 0) return false;
    return true;
  });
}

function normalizePatchPage(count = filteredTrainingPatches().length) {
  const pageCount = Math.max(1, Math.ceil(count / state.patchPageSize));
  state.patchPage = Math.min(Math.max(1, state.patchPage), pageCount);
}

function makeSelect(values, selected) {
  const select = document.createElement("select");
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value || "unsplit";
    option.selected = value === selected;
    select.append(option);
  }
  return select;
}

async function updateTrainingSample(sampleId, payload) {
  const sample = state.trainingSamples.find((item) => item.sample_id === sampleId) || {};
  const region = sample.region || state.region;
  await patchJson(apiPathFor(region, `/training-samples/${encodeURIComponent(sampleId)}`), payload);
  await loadTrainingSamples();
}

async function updateTrainingPatch(patchId, payload) {
  const patch = state.trainingPatches.find((item) => item.patch_id === patchId) || state.activePatch || {};
  const region = patch.region || state.region;
  await patchJson(apiPathFor(region, `/training-patches/${encodeURIComponent(patchId)}`), payload);
  await loadTrainingPatches();
  if (state.activePatch?.patch_id === patchId) {
    state.activePatch = state.trainingPatches.find((patch) => patch.patch_id === patchId) || state.activePatch;
    renderPatchModal();
  }
}

async function startPatchExport() {
  patchExportEl.disabled = true;
  patchExportStatusEl.textContent = "提交生成任务";
  const payload = {
    patch_size: Number(patchSizeEl.value || 256),
    stride: Number(patchStrideEl.value || 128),
    preview_scale: Number(patchPreviewScaleEl.value || 2),
    overwrite: patchOverwriteEl.checked,
  };
  try {
    const job = await postJson(apiPath("/training-patches/export-jobs"), payload);
    pollPatchExportJob(job.job_id).catch(showError);
  } catch (error) {
    patchExportEl.disabled = false;
    throw error;
  }
}

async function pollPatchExportJob(jobId) {
  const job = await fetchJson(apiPath(`/training-patches/export-jobs/${encodeURIComponent(jobId)}`));
  patchExportStatusEl.textContent = job.message || job.status || "处理中";
  if (job.status === "completed") {
    patchExportEl.disabled = false;
    await loadTrainingPatches();
    return;
  }
  if (job.status === "failed") {
    patchExportEl.disabled = false;
    throw new Error(job.message || "Patch 生成失败");
  }
  setTimeout(() => pollPatchExportJob(jobId).catch(showError), 1500);
}

async function startTrainingRun() {
  trainStartEl.disabled = true;
  trainCancelEl.disabled = false;
  trainStatusEl.textContent = "提交训练任务";
  const payload = {
    run_name: trainRunNameEl.value.trim(),
    epochs: Number(trainEpochsEl.value || 30),
    batch_size: Number(trainBatchSizeEl.value || 8),
    lr: Number(trainLrEl.value || 0.001),
    base_channels: Number(trainBaseChannelsEl.value || 32),
    device: trainDeviceEl.value || "cuda",
    no_augment: trainNoAugmentEl.checked,
  };
  try {
    const job = await postJson(apiPath("/training-runs"), payload);
    state.activeTrainingJob = job;
    renderTrainingRunView();
    pollTrainingRun(job.job_id).catch(showError);
  } catch (error) {
    trainStartEl.disabled = false;
    trainCancelEl.disabled = true;
    throw error;
  }
}

async function pollTrainingRun(jobId) {
  const job = await fetchJson(apiPath(`/training-runs/${encodeURIComponent(jobId)}`));
  state.activeTrainingJob = job;
  const index = state.trainingRuns.findIndex((item) => item.job_id === job.job_id);
  if (index >= 0) state.trainingRuns[index] = job;
  else state.trainingRuns.unshift(job);
  renderTrainingRunView();
  const running = ["queued", "configured", "running", "cancel_requested"].includes(job.status);
  trainStartEl.disabled = running;
  trainCancelEl.disabled = !running;
  trainStatusEl.textContent = job.message || job.status || "处理中";
  if (running) {
    setTimeout(() => pollTrainingRun(jobId).catch(showError), 2000);
    return;
  }
  if (job.status === "completed") {
    await loadModelOptions();
  }
}

async function cancelTrainingRun() {
  if (!state.activeTrainingJob?.job_id) return;
  trainCancelEl.disabled = true;
  trainStatusEl.textContent = "正在取消";
  const job = await postJson(apiPath(`/training-runs/${encodeURIComponent(state.activeTrainingJob.job_id)}/cancel`), {});
  state.activeTrainingJob = job;
  renderTrainingRunView();
}

function renderTrainingRunView() {
  const job = state.activeTrainingJob;
  trainingRunBodyEl.replaceChildren();
  const running = job && ["queued", "configured", "running", "cancel_requested"].includes(job.status);
  trainStartEl.disabled = Boolean(running);
  trainCancelEl.disabled = !running;
  const dataset = job?.dataset || job?.result?.dataset || state.trainingDataset;
  if (!job) {
    trainingRunSummaryEl.textContent = "选择参数后开始训练";
    trainStatusEl.textContent = "未开始";
    trainingRunBodyEl.append(renderTrainingDatasetPanel(dataset, "当前训练数据"));
    return;
  }
  const result = job.result || {};
  const config = job.config || result.config || {};
  const history = job.history || result.history || [];
  const latest = history[history.length - 1] || job.record || {};
  const train = latest.train || {};
  const val = latest.val || {};
  const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
  trainingRunSummaryEl.textContent = [
    trainingStatusLabel(job.status),
    job.epoch && job.epochs ? `epoch ${job.epoch}/${job.epochs}` : "",
    result.best_model ? `best ${result.best_model}` : "",
  ].filter(Boolean).join(" · ");
  trainStatusEl.textContent = job.message || trainingStatusLabel(job.status);
  const metrics = [
    ["状态", trainingStatusLabel(job.status)],
    ["进度", `${formatNumber(progress, 0)}%`],
    ["epoch", job.epoch && job.epochs ? `${job.epoch}/${job.epochs}` : "0"],
    ["train IoU", Number.isFinite(Number(train.iou)) ? formatNumber(Number(train.iou), 4) : ""],
    ["val IoU", Number.isFinite(Number(val.iou)) ? formatNumber(Number(val.iou), 4) : ""],
    ["train Dice", Number.isFinite(Number(train.dice)) ? formatNumber(Number(train.dice), 4) : ""],
    ["val Dice", Number.isFinite(Number(val.dice)) ? formatNumber(Number(val.dice), 4) : ""],
    ["输入波段", config.in_channels || ""],
    ["模型宽度", config.base_channels || ""],
    ["设备", config.device || ""],
    ["输出", result.output_dir || job.output_dir || config.output_dir || ""],
    ["best", result.best_model || ""],
  ];
  const cards = document.createElement("div");
  cards.className = "training-run-metrics";
  for (const [label, value] of metrics) {
    const card = document.createElement("div");
    card.className = "training-run-metric";
    card.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value || "-")}</strong>`;
    cards.append(card);
  }
  const bar = document.createElement("div");
  bar.className = "training-progress";
  bar.innerHTML = `<div style="width:${progress}%"></div>`;
  const log = document.createElement("div");
  log.className = "training-run-log";
  const lines = history.slice(-12).map((record) => {
    const rTrain = record.train || {};
    const rVal = record.val || {};
    return `epoch ${record.epoch}: train_iou=${formatNumber(Number(rTrain.iou || 0), 4)} val_iou=${formatNumber(Number(rVal.iou || 0), 4)} train_loss=${formatNumber(Number(rTrain.loss || 0), 4)} val_loss=${formatNumber(Number(rVal.loss || 0), 4)}`;
  });
  log.textContent = lines.length ? lines.join("\n") : (job.message || "等待训练日志");
  const previous = document.createElement("div");
  previous.className = "training-run-history";
  const previousTitle = document.createElement("div");
  previousTitle.className = "training-run-history-title";
  previousTitle.textContent = `历史任务 ${state.trainingRuns.length}`;
  const previousList = document.createElement("div");
  previousList.className = "training-run-history-list";
  for (const item of state.trainingRuns) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = item.job_id === job.job_id ? "active" : "";
    row.title = item.result?.output_dir || item.message || item.job_id;
    row.innerHTML = formatTrainingHistoryRow(item);
    row.addEventListener("click", () => {
      state.activeTrainingJob = item;
      renderTrainingRunView();
    });
    previousList.append(row);
  }
  previous.append(previousTitle, previousList);
  trainingRunBodyEl.append(renderTrainingDatasetPanel(dataset, "本任务数据"), bar, cards, log, previous);
}

function renderTrainingDatasetPanel(dataset, title) {
  const panel = document.createElement("section");
  panel.className = "training-dataset-panel";
  const heading = document.createElement("div");
  heading.className = "training-dataset-heading";
  const scopeLabel = dataset?.scope === "all" ? "全部区域" : (currentRegionName(dataset?.scope) || dataset?.scope || "当前区域");
  heading.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(scopeLabel)}</span>`;
  panel.append(heading);
  if (!dataset || dataset.error) {
    const empty = document.createElement("div");
    empty.className = "training-run-empty";
    empty.textContent = dataset?.error || "还没有可用于训练的 patch，请先在 Patch 页生成。";
    panel.append(empty);
    return panel;
  }
  const patchSize = Array.isArray(dataset.patch_size) && dataset.patch_size.length
    ? dataset.patch_size.join(" x ")
    : "";
  const manifests = (dataset.manifests || []).map((item) => item.manifest).filter(Boolean);
  const fields = [
    ["范围", scopeLabel],
    ["区域", (dataset.regions || []).join(", ")],
    ["可用 Patch", String(dataset.usable_patches ?? dataset.included_patches ?? 0)],
    ["包含/排除", `${dataset.included_patches || 0} / ${dataset.excluded_patches || 0}`],
    ["样本", String(dataset.sample_count || 0)],
    ["水体", String(dataset.lake_count || 0)],
    ["训练/验证", dataset.train_count != null || dataset.val_count != null ? `${dataset.train_count || 0} / ${dataset.val_count || 0}` : ""],
    ["输入", [dataset.in_channels ? `${dataset.in_channels} band` : "", patchSize].filter(Boolean).join(" · ")],
    ["水体像元", formatPercentText(dataset.water_ratio)],
    ["Manifest", manifests.join(" | ")],
  ];
  const cards = document.createElement("div");
  cards.className = "training-run-metrics training-dataset-metrics";
  for (const [label, value] of fields) {
    if (!value) continue;
    const card = document.createElement("div");
    card.className = "training-run-metric";
    card.innerHTML = `<span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong>`;
    cards.append(card);
  }
  panel.append(cards);
  return panel;
}

function currentRegionName(key) {
  if (!key) return "";
  if (key === "all") return "全部区域";
  return state.regions.find((region) => region.key === key)?.name || key;
}

function trainingStatusLabel(status) {
  const labels = {
    queued: "排队中",
    configured: "已配置",
    running: "训练中",
    cancel_requested: "取消中",
    cancelled: "已取消",
    completed: "已完成",
    failed: "失败",
  };
  return labels[status] || status || "处理中";
}

function formatTrainingHistoryLabel(item) {
  const result = item.result || {};
  const config = item.config || result.config || {};
  const history = item.history || result.history || [];
  const latest = history[history.length - 1] || {};
  const val = latest.val || {};
  const iou = Number(result.best_iou ?? val.iou);
  const name = item.run_name || config.output_dir?.split("/")?.pop() || item.job_id;
  const parts = [
    name,
    trainingStatusLabel(item.status),
    Number.isFinite(iou) ? `best IoU ${formatNumber(iou, 4)}` : "",
  ].filter(Boolean);
  return parts.join(" · ");
}

function formatTrainingHistoryRow(item) {
  const result = item.result || {};
  const config = item.config || result.config || {};
  const history = item.history || result.history || [];
  const latest = history[history.length - 1] || {};
  const train = latest.train || {};
  const val = latest.val || {};
  const name = item.run_name || config.output_dir?.split("/")?.pop() || item.job_id;
  const bestIou = Number(result.best_iou ?? val.iou);
  const valIou = Number(val.iou);
  const trainIou = Number(train.iou);
  const timeText = formatDateTimeText(item.updated_at || item.created_at);
  const status = trainingStatusLabel(item.status);
  const epoch = item.epoch && item.epochs ? `${item.epoch}/${item.epochs}` : (item.epoch || "");
  const metrics = [
    epoch ? `epoch ${epoch}` : "",
    Number.isFinite(bestIou) ? `best ${formatNumber(bestIou, 4)}` : "",
    Number.isFinite(valIou) ? `val ${formatNumber(valIou, 4)}` : "",
    Number.isFinite(trainIou) ? `train ${formatNumber(trainIou, 4)}` : "",
  ].filter(Boolean).join(" · ");
  return `
    <span class="training-run-history-main">
      <strong>${escapeHtml(name)}</strong>
      <span>${escapeHtml(metrics || item.job_id)}</span>
    </span>
    <span class="training-run-history-side">
      <span>${escapeHtml(status)}</span>
      <span>${escapeHtml(timeText)}</span>
    </span>
  `;
}

function formatDateTimeText(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  const match = text.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (match) return `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}`;
  return text;
}

function openPatchModal(patch) {
  state.activePatch = patch;
  renderPatchModal();
  patchModalEl.hidden = false;
}

function closePatchModal() {
  patchModalEl.hidden = true;
  state.activePatch = null;
}

function renderPatchModal() {
  const patch = state.activePatch;
  if (!patch) return;
  const name = patch.lake_display_name || patch.lake_name || patch.lake_id || patch.sample_id;
  patchModalTitleEl.textContent = name || "Patch 预览";
  patchModalSubtitleEl.textContent = patch.patch_id || "";
  patchModalImageEl.src = patch.preview_url || "";
  patchModalImageEl.alt = name ? `${name} patch` : "patch preview";
  patchModalToggleEl.textContent = patch.included ? "排除这个 patch" : "恢复包含";
  patchModalMetaEl.replaceChildren();
  for (const [label, value] of [
    ["状态", patch.included ? "include" : "exclude"],
    ["water", formatPercentText(patch.water_ratio_valid)],
    ["valid", formatPercentText(patch.valid_ratio)],
    ["ignore", patch.ignore_pixels || 0],
    ["sample", patch.sample_id || ""],
    ["image", patch.product_name || patch.image_path || ""],
  ]) {
    const row = document.createElement("div");
    row.className = "patch-modal-meta-row";
    row.innerHTML = `<span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong>`;
    patchModalMetaEl.append(row);
  }
}

async function deleteTrainingSample(sampleId) {
  if (!confirm("删除这个训练样本记录？")) return;
  const sample = state.trainingSamples.find((item) => item.sample_id === sampleId) || {};
  const region = sample.region || state.region;
  await deleteJson(apiPathFor(region, `/training-samples/${encodeURIComponent(sampleId)}`));
  await loadTrainingSamples();
}

async function selectLake(shapeId, options = {}) {
  const waitForTile = options.waitForTile ?? state.sidebarMode !== "model";
  const validationRunId = options.validationRunId || 0;
  const keepModelPrediction = state.sidebarMode === "model" && validationRunId;
  state.activeId = shapeId;
  if (state.sidebarMode !== "model") state.sidebarMode = "lakes";
  updateRouteUrl();
  state.loadingId = shapeId;
  state.metaParts = {};
  state.imagery = null;
  state.localLabels = [];
  state.modelValidation = null;
  renderTrainingReadiness();
  state.tileMeta = null;
  sentinelPanelEl.hidden = true;
  trainingPanelEl.hidden = true;
  sentinelProductsEl.replaceChildren();
  imageryProductEl.replaceChildren();
  rasterLayer.setSource(null);
  if (!keepModelPrediction) vectorSources.modelPrediction.clear();
  resetLocalLabelSelect();
  clearVectorLayers(keepModelPrediction ? new Set(["modelPrediction"]) : undefined);
  renderList();
  titleEl.textContent = `水体 ${shapeId}`;
  subtitleEl.textContent = "加载影像和边界";
  setLoading(true, "加载地图数据");
  emptyEl.hidden = true;
  ensureMapVisible();

  const lake = await fetchJson(activeApiPath(`/lakes/${shapeId}`));
  if (state.activeId !== shapeId || (validationRunId && validationRunId !== state.modelValidationRunId)) return;
  state.lake = lake;
  renderActiveLakeTitle();
  addLayerGeometry("osm", lake.layers?.osm);
  addLayerGeometry("hydrolakes", lake.layers?.hydrolakes);
  const tilePromise = loadTileLayer(shapeId, lake).catch((error) => {
    if (!isMissingTciError(error)) throw error;
    rasterLayer.setSource(null);
    state.tileMeta = null;
    fitToBounds(lake.bbox);
    state.metaParts.base = "暂无本地 Sentinel 影像，可查询并下载产品";
    renderMeta();
    setLoading(false);
  });
  if (waitForTile) {
    await tilePromise;
  } else {
    tilePromise.catch(showError);
  }
  state.loadingId = null;
  renderList();
  const followups = [
    loadSentinelTiles(shapeId),
    loadImageryOptions(shapeId),
    loadContextWaterLayer(shapeId),
    loadEsaLayer(shapeId),
    loadJrcLayer(shapeId),
    loadLocalLabels(shapeId),
  ];
  if (validationRunId) {
    await Promise.race([
      Promise.allSettled([tilePromise, ...followups]),
      new Promise((resolve) => setTimeout(resolve, 8000)),
    ]);
    if (validationRunId !== state.modelValidationRunId) return;
    if (state.activeId === shapeId && state.sidebarMode === "model") setLoading(false);
  } else {
    for (const promise of followups) promise.catch(showError);
  }
}

function renderActiveLakeTitle() {
  if (!state.lake) return;
  titleEl.textContent = state.lake.display_name || state.lake.name || `水体 ${state.lake.object_id}`;
  const hylak = state.lake.layers?.hydrolakes?.properties?.Hylak_id;
  subtitleEl.textContent = `${typeLabel(state.lake.water_type)} · ${state.lake.lake_id}${hylak ? ` · Hylak ${hylak}` : ""}`;
}

async function loadTileLayer(shapeId, lake, options = {}) {
  const forceSource = Boolean(options.forceSource);
  setLoading(true, "加载影像瓦片");
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/tile-meta?padding=0.8&v=${Date.now()}`));
  if (state.activeId !== shapeId) return;
  state.tileMeta = payload;
  if (forceSource || state.sidebarMode !== "model" || toggleImageEl.checked) {
    rasterLayer.setSource(
      new ol.source.XYZ({
        url: activeApiPath(`/lakes/${shapeId}/tiles/{z}/{x}/{y}.png?v=${Date.now()}`),
        tileSize: 256,
        minZoom: 5,
        maxZoom: 16,
        transition: 120,
      }),
    );
  }
  rasterLayer.setVisible(toggleImageEl.checked);
  const focusBounds = payload.lake_bounds || payload.bounds || lake.bbox;
  fitToBounds(focusBounds);
  state.metaParts.base = [
    `影像 tile ${formatMetaList(payload.tiles)}`,
    `日期 ${formatMetaList(payload.dates)}`,
    `产品 ${formatProductList(payload.products)}`,
    `瓦片渲染`,
  ].join(" | ");
  renderMeta();
  setLoading(false);
}

function isMissingTciError(error) {
  const message = String(error?.message || "");
  return error?.status === 404 || message.includes("No downloaded TCI");
}

async function loadContextWaterLayer(shapeId) {
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/context-water?padding=0.8&min_area_km2=10&limit=500`));
  if (state.activeId !== shapeId) return;
  addFeatureCollection("contextOsm", payload.sources?.osm);
  addFeatureCollection("contextHydrolakes", payload.sources?.hydrolakes);
  const osmCount = payload.sources?.osm?.features?.length || 0;
  const hydroCount = payload.sources?.hydrolakes?.features?.length || 0;
  state.metaParts.context = `其他水体(≥${payload.min_area_km2} km²) OSM ${osmCount} / HydroLAKES ${hydroCount}`;
  renderMeta();
}

async function loadEsaLayer(shapeId) {
  state.metaParts.esa = "ESA 平滑边界生成中";
  renderMeta();
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/esa`));
  if (!state.lake || state.activeId !== shapeId) return;
  if (!payload.esa || !payload.esa.geometry) {
    const reason = payload.esa?.properties?.reason;
    state.metaParts.esa = reason ? `ESA 已跳过：${reason}` : "ESA 无结果";
    renderMeta();
    return;
  }
  state.lake.layers.esa = payload.esa;
  addLayerGeometry("esa", payload.esa);
  state.metaParts.esa = "ESA 平滑边界已加载";
  renderMeta();
}

async function loadJrcLayer(shapeId) {
  const threshold = Number(jrcThresholdEl.value);
  jrcThresholdValueEl.textContent = `${threshold}%`;
  if (state.lake?.layers?.jrc?.properties?.skipped) {
    const reason = state.lake.layers.jrc.properties.reason || "当前水体不支持实时生成";
    state.metaParts.jrc = `JRC ${threshold}%：${reason}`;
    renderMeta();
    return;
  }
  state.metaParts.jrc = `JRC ${threshold}% 边界生成中`;
  renderMeta();
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/jrc?threshold=${threshold}`));
  if (!state.lake || state.activeId !== shapeId) return;
  if (!payload.jrc || !payload.jrc.geometry) {
    const reason = payload.jrc?.properties?.reason || (payload.jrc?.properties?.empty ? "无匹配水体" : "");
    const available = payload.jrc?.properties?.available_thresholds || [];
    if (available.length) {
      const nearest = nearestThreshold(threshold, available);
      if (nearest !== threshold) {
        jrcThresholdEl.value = String(nearest);
        jrcThresholdValueEl.textContent = `${nearest}%`;
        state.metaParts.jrc = `JRC ${threshold}% 不支持实时生成，切换到预生成 ${nearest}%`;
        renderMeta();
        loadJrcLayer(shapeId).catch(showError);
        return;
      }
    }
    state.metaParts.jrc = `JRC ${threshold}%${reason ? `：${reason}` : " 无结果"}`;
    vectorSources.jrc.clear();
    renderMeta();
    return;
  }
  state.lake.layers.jrc = payload.jrc;
  addLayerGeometry("jrc", payload.jrc);
  state.metaParts.jrc = `JRC ${threshold}% 边界已加载`;
  renderMeta();
}

async function loadLocalLabels(shapeId) {
  resetLocalLabelSelect();
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/local-labels`));
  if (state.activeId !== shapeId) return;
  state.localLabels = payload.items || [];
  renderLocalLabelOptions();
  const first = state.localLabels[0];
  if (first) {
    localLabelSelectEl.value = first.id;
    await loadSelectedLocalLabel(shapeId);
  } else {
    state.metaParts.localLabel = "本地水体标注：无";
    renderMeta();
  }
}

async function loadSelectedLocalLabel(shapeId = state.activeId) {
  vectorSources.localLabel.clear();
  if (!shapeId || !localLabelSelectEl.value) {
    vectorLayers.localLabel.setVisible(false);
    return;
  }
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/local-labels/${encodeURIComponent(localLabelSelectEl.value)}`));
  if (state.activeId !== shapeId) return;
  addFeatureCollection("localLabel", payload.geojson);
  const count = payload.geojson?.features?.length || 0;
  state.metaParts.localLabel = `本地水体标注 ${payload.label?.name || ""} (${count})`;
  renderMeta();
}

function renderLocalLabelOptions() {
  localLabelSelectEl.replaceChildren();
  localLabelSelectEl.disabled = state.localLabels.length === 0;
  if (!state.localLabels.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无本地标注";
    localLabelSelectEl.append(option);
    return;
  }
  for (const item of state.localLabels) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.date ? `${item.date} ${item.name}` : item.name;
    option.title = item.path || item.name;
    localLabelSelectEl.append(option);
  }
}

function resetLocalLabelSelect() {
  state.localLabels = [];
  vectorSources.localLabel.clear();
  localLabelSelectEl.replaceChildren();
  const option = document.createElement("option");
  option.value = "";
  option.textContent = "无本地标注";
  localLabelSelectEl.append(option);
  localLabelSelectEl.disabled = true;
  delete state.metaParts.localLabel;
}

async function loadSentinelTiles(shapeId) {
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/sentinel/tiles`));
  if (state.activeId !== shapeId) return;
  renderTileGrid(payload.tiles || []);
  sentinelTileEl.replaceChildren();
  for (const item of payload.tiles) {
    const option = document.createElement("option");
    option.value = item.tile;
    option.textContent = item.tile;
    sentinelTileEl.append(option);
  }
  sentinelPanelEl.hidden = state.sidebarMode === "model" || payload.tiles.length === 0;
  renderImageryOptions();
}

function renderTileGrid(tiles) {
  vectorSources.tileGrid.clear();
  const features = [];
  for (const item of tiles) {
    if (!item.geometry) continue;
    const feature = geojson.readFeature({
      type: "Feature",
      geometry: item.geometry,
      properties: {
        tile: item.tile,
        aoi_coverage_ratio: item.aoi_coverage_ratio,
      },
    });
    features.push(feature);
  }
  vectorSources.tileGrid.addFeatures(features);
  vectorLayers.tileGrid.setVisible(toggleTileGridEl.checked);
}

async function loadImageryOptions(shapeId) {
  const payload = await fetchJson(activeApiPath(`/lakes/${shapeId}/imagery`));
  if (state.activeId !== shapeId) return;
  state.imagery = payload;
  trainingPanelEl.hidden = state.sidebarMode === "model";
  renderImageryOptions();
}

function renderImageryOptions() {
  imageryProductEl.replaceChildren();
  const selectedTile = sentinelTileEl.value;
  const tile = state.imagery?.tiles?.find((item) => item.tile === selectedTile);
  const products = tile?.products || [];
  imageryApplyEl.disabled = products.length === 0;
  if (!products.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无本地影像";
    imageryProductEl.append(option);
    renderTrainingReadiness();
    return;
  }
  for (const product of products) {
    const option = document.createElement("option");
    option.value = product.product;
    option.textContent = formatImageryOptionText(product);
    imageryProductEl.append(option);
    if (product.active) imageryProductEl.value = product.product;
  }
  renderTrainingReadiness();
}

function renderTrainingReadiness() {
  trainingSaveEl.disabled = !state.activeId || trainingPanelEl.hidden;
  if (trainingPanelEl.hidden) trainingStatusEl.textContent = "";
}

async function saveTrainingSample() {
  if (!state.activeId) return;
  trainingSaveEl.disabled = true;
  trainingStatusEl.textContent = "保存中";
  try {
    const payload = await postJson(activeApiPath(`/lakes/${state.activeId}/training-samples`), {
      label_source: "current_view",
      label_threshold: jrcThresholdEl.value,
      label_scope: "current_view",
      mask_policy: "current_view",
      notes: trainingNotesEl.value,
      buffer_ratio: 0.8,
      view_state: captureTrainingViewState(),
    });
    trainingStatusEl.textContent = `已加入训练区域：${payload.sample.sample_id}`;
    if (state.sidebarMode === "training") await loadTrainingSamples();
  } finally {
    trainingSaveEl.disabled = !state.activeId || trainingPanelEl.hidden;
  }
}

function captureTrainingViewState() {
  const view = map.getView();
  const center = ol.proj.toLonLat(view.getCenter());
  const size = map.getSize() || [mapEl.clientWidth, mapEl.clientHeight];
  const extent = ol.proj.transformExtent(view.calculateExtent(size), "EPSG:3857", "EPSG:4326");
  const selectedLocalLabel = state.localLabels.find((item) => item.id === localLabelSelectEl.value) || null;
  return {
    region: state.region,
    lake_id: state.activeId,
    imagery_visible: toggleImageEl.checked,
    visible_layers: {
      tile_grid: toggleTileGridEl.checked,
      osm: toggleOsmEl.checked,
      hydrolakes: toggleHydroEl.checked,
      context_osm: toggleContextWaterEl.checked,
      context_hydrolakes: toggleContextWaterEl.checked,
      esa: toggleEsaEl.checked,
      jrc: toggleJrcEl.checked,
      local_label: toggleLocalLabelEl.checked,
    },
    jrc_threshold: Number(jrcThresholdEl.value),
    selected_local_label: selectedLocalLabel
      ? {
          id: selectedLocalLabel.id,
          name: selectedLocalLabel.name,
          path: selectedLocalLabel.path,
          date: selectedLocalLabel.date || "",
        }
      : null,
    selected_tile: sentinelTileEl.value || "",
    selected_product: imageryProductEl.value || "",
    map: {
      center,
      zoom: view.getZoom(),
      extent,
    },
  };
}

async function applyImagerySelection() {
  if (!state.activeId || !sentinelTileEl.value || !imageryProductEl.value) return;
  imageryApplyEl.disabled = true;
  setLoading(true, "切换影像瓦片");
  try {
    await postJson(activeApiPath(`/lakes/${state.activeId}/imagery/active`), {
      tile: sentinelTileEl.value,
      product: imageryProductEl.value,
    });
    state.metaParts.sentinel = `已切换 ${sentinelTileEl.value} 影像`;
    renderMeta();
    await loadImageryOptions(state.activeId);
    await loadTileLayer(state.activeId, state.lake);
  } finally {
    imageryApplyEl.disabled = false;
    setLoading(false);
  }
}

async function querySentinelProducts() {
  if (!sentinelTileEl.value) return;
  sentinelQueryEl.disabled = true;
  sentinelProductsEl.textContent = "查询中";
  const params = new URLSearchParams({
    tile: sentinelTileEl.value,
    lake_id: state.activeId,
    start: sentinelStartEl.value,
    end: sentinelEndEl.value,
    cloud: sentinelCloudEl.value,
    product_type: "MSIL1C",
    limit: "50",
  });
  try {
    const payload = await fetchJson(activeApiPath(`/sentinel/products?${params.toString()}`));
    renderSentinelProducts(payload.products);
  } finally {
    sentinelQueryEl.disabled = false;
  }
}

function renderSentinelProducts(products) {
  sentinelProductsEl.replaceChildren();
  if (!products.length) {
    sentinelProductsEl.textContent = "没有符合条件的产品";
    return;
  }
  for (const product of products) {
    const row = document.createElement("div");
    row.className = "sentinel-product";
    const action = document.createElement("button");
    action.type = "button";
    action.textContent = product.downloaded ? "已下载" : "下载";
    action.disabled = Boolean(product.downloaded);
    action.addEventListener("click", () => startSentinelDownload(product, action).catch(showError));
    row.innerHTML = `
      <span>${escapeHtml(product.date || "")}</span>
      <span>${escapeHtml(product.tile || "")}</span>
      <span>云量 ${formatCloud(product.cloud_cover)}</span>
      <span>覆盖水体 ${formatPercent(product.lake_coverage_ratio)}</span>
      <span>覆盖视图 ${formatPercent(product.aoi_coverage_ratio)}</span>
      <span>非0 ${formatCoverage(product.coverage_ratio, product.coverage_basis)}</span>
      <span title="${escapeHtml(product.name || "")}">${escapeHtml(product.name || "")}</span>
    `;
    row.append(action);
    sentinelProductsEl.append(row);
  }
}

async function startSentinelDownload(product, button) {
  button.disabled = true;
  button.textContent = "排队中";
  const job = await postJson(activeApiPath("/sentinel/downloads"), { product });
  if (!job.job_id) {
    button.textContent = job.message || "已下载";
    await loadImageryOptions(state.activeId);
    return;
  }
  state.downloadJobs.set(job.job_id, { button, product, region: activeRegionKey() });
  pollDownloadJob(job.job_id).catch(showError);
}

async function pollDownloadJob(jobId) {
  const entry = state.downloadJobs.get(jobId);
  if (!entry) return;
  const job = await fetchJson(apiPathFor(entry.region, `/sentinel/downloads/${jobId}`));
  entry.button.textContent = job.status === "downloading" ? `${job.progress || 0}%` : statusLabel(job.status);
  if (job.status === "completed") {
    entry.button.textContent = "已下载";
    entry.button.disabled = true;
    state.downloadJobs.delete(jobId);
    state.metaParts.sentinel = "Sentinel 产品下载完成";
    renderMeta();
    await loadSentinelTiles(state.activeId);
    await loadImageryOptions(state.activeId);
    return;
  }
  if (job.status === "failed") {
    entry.button.textContent = "失败";
    entry.button.disabled = false;
    state.downloadJobs.delete(jobId);
    throw new Error(job.message || "下载失败");
  }
  setTimeout(() => pollDownloadJob(jobId).catch(showError), 1500);
}

function addLayerGeometry(layerName, layer) {
  const source = vectorSources[layerName];
  source.clear();
  if (!layer?.geometry) return;
  const feature = geojson.readFeature({
    type: "Feature",
    geometry: layer.geometry,
    properties: layer.properties || {},
  });
  source.addFeature(feature);
  vectorLayers[layerName].setVisible(layerVisible(layerName));
}

function addFeatureCollection(layerName, collection) {
  const source = vectorSources[layerName];
  source.clear();
  const features = collection?.features || [];
  if (!features.length) {
    vectorLayers[layerName].setVisible(layerVisible(layerName));
    return;
  }
  source.addFeatures(geojson.readFeatures({ type: "FeatureCollection", features }));
  vectorLayers[layerName].setVisible(layerVisible(layerName));
}

function ensureMapVisible() {
  if (isTrainingWorkspaceView()) return false;
  mapWrapEl.hidden = false;
  mapEl.hidden = false;
  const rect = mapEl.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    map.updateSize();
    return true;
  }
  requestAnimationFrame(() => {
    map.updateSize();
  });
  return false;
}

function clearVectorLayers(keep = new Set()) {
  for (const [name, source] of Object.entries(vectorSources)) {
    if (!keep.has(name)) source.clear();
  }
}

function fitToBounds(bounds) {
  if (!bounds || bounds.length !== 4) return;
  ensureMapVisible();
  const extent = ol.proj.transformExtent(bounds, "EPSG:4326", "EPSG:3857");
  map.getView().fit(extent, {
    padding: [36, 36, 36, 36],
    duration: 180,
    maxZoom: 14,
  });
  requestAnimationFrame(() => {
    map.updateSize();
    map.getView().fit(extent, {
      padding: [36, 36, 36, 36],
      duration: 0,
      maxZoom: 14,
    });
  });
}

function polygonStyle(stroke, fill, lineDash = undefined, width = 2) {
  return new ol.style.Style({
    stroke: new ol.style.Stroke({ color: stroke, width, lineDash }),
    fill: new ol.style.Fill({ color: fill }),
  });
}

function tileGridStyle(feature) {
  const tile = feature.get("tile") || "";
  return new ol.style.Style({
    stroke: new ol.style.Stroke({ color: "rgba(247, 125, 35, 0.95)", width: 2 }),
    fill: new ol.style.Fill({ color: "rgba(247, 125, 35, 0.04)" }),
    text: new ol.style.Text({
      text: tile,
      font: "600 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      fill: new ol.style.Fill({ color: "#743900" }),
      stroke: new ol.style.Stroke({ color: "rgba(255, 255, 255, 0.86)", width: 4 }),
      overflow: true,
    }),
  });
}

function layerVisible(layerName) {
  if (layerName === "tileGrid") return toggleTileGridEl.checked;
  if (layerName === "osm") return toggleOsmEl.checked;
  if (layerName === "hydrolakes") return toggleHydroEl.checked;
  if (layerName === "contextOsm") return toggleContextWaterEl.checked;
  if (layerName === "contextHydrolakes") return toggleContextWaterEl.checked;
  if (layerName === "esa") return toggleEsaEl.checked;
  if (layerName === "jrc") return toggleJrcEl.checked;
  if (layerName === "localLabel") return toggleLocalLabelEl.checked;
  if (layerName === "modelPrediction") return toggleModelPredictionEl.checked;
  return true;
}

function formatNumber(value, digits) {
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatMetaList(value) {
  if (Array.isArray(value)) return value.join(", ");
  return String(value || "");
}

function formatProductList(value) {
  const products = Array.isArray(value) ? value : String(value || "").split(",");
  const labels = products
    .filter(Boolean)
    .map((item) => {
      const tile = item.match(/_T([0-9A-Z]{5})_/)?.[1] || "";
      const date = item.match(/MSIL\d[AC]?_(\d{8})T/)?.[1] || "";
      const compactDate = item.match(/MSIL\d[AC]?_(\d{8})(?:_|$)/)?.[1] || "";
      const fallback = item
        .replace(/^shaanxi_\d+_/, "")
        .replace(/^gansu_\d+_/, "")
        .replace(/^yunnan_\d+_/, "")
        .replace(/^hunan_\d+_/, "");
      return [tile, date || compactDate].filter(Boolean).join("/") || fallback || item;
    })
    .filter(Boolean);
  return labels.join(", ");
}

function formatImageryOptionText(product) {
  const parts = [];
  if (product.active) parts.push("当前");
  parts.push(product.asset_label || formatImageryAssetType(product.asset_type || product.source));
  parts.push(`时间: ${formatDateText(product.date)}`);
  if (product.asset_type === "lake_native") {
    parts.push(`有效 ${formatPercent(product.valid_ratio)}`);
  } else {
    parts.push(`云量 ${formatCloud(product.cloud_cover)}`);
    parts.push(`非0 ${formatPercent(product.valid_ratio)}`);
  }
  return parts.join("，");
}

function formatImageryAssetType(value) {
  const text = String(value || "").trim();
  if (text === "lake_native" || text === "local_img") return "本体影像";
  if (text === "preloaded_tile" || text === "preloaded") return "预置 Sentinel tile";
  if (text === "sentinel_tile" || text === "user_download") return "已下载 Sentinel tile";
  return text || "影像";
}

function formatImageryAssetLabels(value) {
  const labels = Array.isArray(value) ? value : String(value || "").split(",");
  const unique = [];
  for (const item of labels) {
    const label = formatImageryAssetType(item);
    if (label && !unique.includes(label)) unique.push(label);
  }
  return unique.join(" + ") || "影像";
}

function formatCloud(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "未知";
  return `${formatNumber(number, 1)}%`;
}

function formatCoverage(value, basis) {
  const number = Number(value);
  if (!Number.isFinite(number) || basis !== "pixels") return "需下载后统计";
  return `${formatNumber(number * 100, 1)}%`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "未知";
  return `${formatNumber(number * 100, 1)}%`;
}

function formatPercentText(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0.0%";
  return `${formatNumber(number * 100, 1)}%`;
}

function formatDateText(value) {
  const text = String(value || "").trim();
  if (/^\d{8}$/.test(text)) return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
  return text || "未知";
}

function formatLabelScope(value) {
  if (value === "current_view") return "当前视图";
  if (value === "aoi_water") return "AOI 水体";
  return "当前水体";
}

function formatMaskPolicy(value) {
  if (value === "current_view") return "当前视图";
  if (value === "visible_water_positive") return "可见水体为正";
  return "其他水体忽略";
}

function formatTrainingLabelSource(source, threshold) {
  if (source === "current_view") return "当前视图";
  if (source === "jrc" && threshold) return `jrc ${threshold}`;
  return source || "";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return map[char];
  });
}

function setLoading(visible, text = "加载中") {
  loadingEl.hidden = !visible;
  loadingTextEl.textContent = text;
  if (visible) metaEl.textContent = text;
}

function renderMeta() {
  metaEl.textContent = [
    state.metaParts.base,
    state.metaParts.context,
    state.metaParts.esa,
    state.metaParts.jrc,
    state.metaParts.localLabel,
    state.metaParts.model,
    state.metaParts.sentinel,
  ]
    .filter(Boolean)
    .join(" | ");
}

function currentFilters() {
  return {
    water_type: filterTypeEl.value,
    area_bucket: filterAreaEl.value,
    has_name: filterNameEl.value,
    has_tci: filterTciEl.value,
    polygon_quality: filterPolygonQualityEl.value,
    metadata_quality: filterMetadataQualityEl.value,
  };
}

function applyFilterControls() {
  filterTypeEl.value = state.filters.water_type || "";
  filterAreaEl.value = state.filters.area_bucket || "";
  filterNameEl.value = state.filters.has_name || "";
  filterTciEl.value = state.filters.has_tci || "";
  filterPolygonQualityEl.value = state.filters.polygon_quality || "";
  filterMetadataQualityEl.value = state.filters.metadata_quality || "";
}

function typeLabel(value) {
  const labels = {
    lake: "湖泊",
    reservoir: "水库",
    pond: "坑塘",
    pond_candidate: "疑似坑塘",
    wetland: "湿地",
    aquaculture: "养殖水面",
    unknown: "未分类",
  };
  return labels[value] || value || "水体";
}

function nearestThreshold(value, thresholds) {
  return thresholds.reduce((best, item) => {
    const currentDistance = Math.abs(Number(item) - value);
    const bestDistance = Math.abs(Number(best) - value);
    return currentDistance < bestDistance ? Number(item) : Number(best);
  }, Number(thresholds[0]));
}

function statusLabel(status) {
  const labels = {
    queued: "排队中",
    authenticating: "连接中",
    downloading: "下载中",
    indexing: "登记中",
    completed: "已下载",
    failed: "失败",
  };
  return labels[status] || status || "处理中";
}

function setDefaultSentinelFilters() {
  const end = new Date();
  const start = new Date(end);
  start.setMonth(start.getMonth() - 2);
  sentinelStartEl.value = formatDateInput(start);
  sentinelEndEl.value = formatDateInput(end);
  sentinelCloudEl.value = "50";
}

function formatDateInput(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = searchEl.value;
    updateRouteUrl({ replace: true });
    loadLakes().catch(showError);
  }, 180);
});

for (const select of [filterTypeEl, filterAreaEl, filterNameEl, filterTciEl, filterPolygonQualityEl, filterMetadataQualityEl]) {
  select.addEventListener("change", () => {
    state.filters = currentFilters();
    updateRouteUrl({ replace: true });
    loadLakes().catch(showError);
  });
}

regionSelectEl.addEventListener("change", () => {
  switchRegion(regionSelectEl.value).catch(showError);
});

loadMoreEl.addEventListener("click", () => {
  loadMoreEl.disabled = true;
  loadLakes({ append: true })
    .catch(showError)
    .finally(() => {
      loadMoreEl.disabled = false;
    });
});

toggleImageEl.addEventListener("change", () => {
  rasterLayer.setVisible(toggleImageEl.checked);
  if (toggleImageEl.checked && !rasterLayer.getSource() && state.activeId && state.lake) {
    loadTileLayer(state.activeId, state.lake, { forceSource: true }).catch(showError);
  }
});

for (const [checkbox, layerName] of [
  [toggleTileGridEl, "tileGrid"],
  [toggleOsmEl, "osm"],
  [toggleHydroEl, "hydrolakes"],
  [toggleContextWaterEl, "contextOsm"],
  [toggleContextWaterEl, "contextHydrolakes"],
  [toggleEsaEl, "esa"],
  [toggleJrcEl, "jrc"],
  [toggleLocalLabelEl, "localLabel"],
  [toggleModelPredictionEl, "modelPrediction"],
]) {
  checkbox.addEventListener("change", () => {
    vectorLayers[layerName].setVisible(checkbox.checked);
  });
}

localLabelSelectEl.addEventListener("change", () => {
  loadSelectedLocalLabel().catch(showError);
});

jrcThresholdEl.addEventListener("input", () => {
  jrcThresholdValueEl.textContent = `${jrcThresholdEl.value}%`;
});

jrcThresholdEl.addEventListener("change", () => {
  if (!state.lake || !state.activeId) return;
  clearTimeout(jrcTimer);
  jrcTimer = setTimeout(() => loadJrcLayer(state.activeId).catch(showError), 120);
});

sentinelQueryEl.addEventListener("click", () => {
  querySentinelProducts().catch(showError);
});

tabLakesEl.addEventListener("click", () => {
  setSidebarMode("lakes");
});

tabTrainingEl.addEventListener("click", () => {
  setSidebarMode("training");
});

tabModelEl.addEventListener("click", () => {
  setSidebarMode("model");
});

modelRandomEl.addEventListener("click", () => {
  runRandomModelValidation().catch(showError);
});

modelSelectEl.addEventListener("change", () => {
  state.selectedModel = modelSelectEl.value;
  renderSelectedModelDetail();
  updateRouteUrl({ replace: true });
  if (state.sidebarMode === "model" && state.activeId) {
    const params = new URLSearchParams();
    const localModel = localModelKeyForActiveRegion();
    if (isAllRegions() && state.selectedModel && localModel === state.selectedModel) {
      modelSummaryEl.textContent = "当前模型不属于这个水体区域，请随机验证";
      return;
    }
    if (localModel) params.set("model", localModel);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    modelSummaryEl.textContent = "模型推理中";
    fetchJson(activeApiPath(`/lakes/${encodeURIComponent(state.activeId)}/model-prediction${suffix}`))
      .then(applyModelPrediction)
      .catch(showError);
  }
});

trainingRefreshEl.addEventListener("click", () => {
  loadActiveTrainingView().catch(showError);
});

trainingViewSamplesEl.addEventListener("click", () => {
  setTrainingView("samples");
});

trainingViewPatchesEl.addEventListener("click", () => {
  setTrainingView("patches");
});

trainingViewTrainEl.addEventListener("click", () => {
  setTrainingView("train");
});

patchIncludeFilterEl.addEventListener("change", () => {
  state.patchIncludeFilter = patchIncludeFilterEl.value;
  state.patchPage = 1;
  loadTrainingPatches().catch(showError);
});

patchWaterFilterEl.addEventListener("change", () => {
  state.patchWaterFilter = patchWaterFilterEl.value;
  state.patchPage = 1;
  renderPatchReview();
});

patchExportEl.addEventListener("click", () => {
  startPatchExport().catch(showError);
});

trainStartEl.addEventListener("click", () => {
  startTrainingRun().catch(showError);
});

trainCancelEl.addEventListener("click", () => {
  cancelTrainingRun().catch(showError);
});

patchPrevEl.addEventListener("click", () => {
  state.patchPage -= 1;
  normalizePatchPage();
  renderPatchReview();
});

patchNextEl.addEventListener("click", () => {
  state.patchPage += 1;
  normalizePatchPage();
  renderPatchReview();
});

patchModalCloseEl.addEventListener("click", () => {
  closePatchModal();
});

patchModalEl.addEventListener("click", (event) => {
  if (event.target === patchModalEl) closePatchModal();
});

patchModalToggleEl.addEventListener("click", () => {
  if (!state.activePatch) return;
  updateTrainingPatch(state.activePatch.patch_id, { include: !state.activePatch.included }).catch(showError);
});

sentinelTileEl.addEventListener("change", () => {
  renderImageryOptions();
});

imageryApplyEl.addEventListener("click", () => {
  applyImagerySelection().catch(showError);
});

trainingSaveEl.addEventListener("click", () => {
  saveTrainingSample().catch(showError);
});

zoomLakeEl.addEventListener("click", () => {
  if (!state.tileMeta) return;
  fitToBounds(state.tileMeta.lake_bounds || state.tileMeta.bounds);
});

zoomTileEl.addEventListener("click", () => {
  if (!state.tileMeta) return;
  fitToBounds(state.tileMeta.tile_bounds || state.tileMeta.bounds);
});

window.addEventListener("resize", () => {
  ensureMapVisible();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !patchModalEl.hidden) closePatchModal();
});

function showError(error) {
  console.error(error);
  setLoading(false);
  state.loadingId = null;
  renderList();
  metaEl.textContent = error.message;
}

async function restoreFromRoute(route = parseRoute()) {
  state.restoringUrl = true;
  try {
    if (route.region === "all" || (route.region && state.regions.some((region) => region.key === route.region))) {
      state.region = route.region;
    }
    state.query = route.query || "";
    state.filters = route.filters || {};
    state.sidebarMode = route.mode || "lakes";
    state.trainingView = route.trainingView || "samples";
    state.selectedModel = route.model || "";
    state.activeRegion = route.lakeRegion || (state.region === "all" ? "" : state.region);
    searchEl.value = state.query;
    applyFilterControls();
    renderRegions();
    resetSelection();
    setSidebarMode(state.sidebarMode);
    await loadLakes();
    if (state.sidebarMode === "training") {
      setTrainingView(state.trainingView);
    }
    if (state.sidebarMode === "model") {
      await loadModelOptions(route.model || state.selectedModel);
    }
    if (route.lakeId) {
      if (route.lakeRegion) state.activeRegion = route.lakeRegion;
      await selectLake(route.lakeId, { waitForTile: route.mode !== "model" });
      if (route.mode === "model") {
        state.sidebarMode = "model";
        setSidebarMode("model");
        const params = new URLSearchParams();
        const localModel = localModelKeyForActiveRegion();
        if (localModel) params.set("model", localModel);
        const suffix = params.toString() ? `?${params.toString()}` : "";
        fetchJson(activeApiPath(`/lakes/${encodeURIComponent(route.lakeId)}/model-prediction${suffix}`))
          .then((payload) => {
            if (state.activeId === route.lakeId) applyModelPrediction(payload);
          })
          .catch(showError);
      }
    }
  } finally {
    state.restoringUrl = false;
  }
  updateRouteUrl({ replace: true });
}

window.addEventListener("popstate", () => {
  restoreFromRoute().catch(showError);
});

loadRegions()
  .then((route) => restoreFromRoute(route))
  .catch(showError);
