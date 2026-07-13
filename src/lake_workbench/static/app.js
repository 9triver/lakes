import { fetchJson, patchJson, postJson } from "./api.js";
import {
  escapeHtml,
  formatCloud,
  formatDateInput,
  formatDateText,
  formatDateTimeText,
  formatImageryAssetLabels,
  formatImageryAssetType,
  formatLabelScope,
  formatMaskPolicy,
  formatMetaList,
  formatNumber,
  formatPercent,
  formatPercentText,
  formatProductList,
  formatTrainingLabelSource,
  nearestThreshold,
  trainingStatusLabel,
  typeLabel,
} from "./formatters.js";
import { createMapController } from "./map-ui.js";
import { createModelValidationController } from "./model-validation-controller.js";
import { createPatchReviewController } from "./patch-review-controller.js";
import { createPatchExportController } from "./patch-export-controller.js";
import { createTrainingRunController } from "./training-run-controller.js";
import { createTrainingSamplesController } from "./training-samples-controller.js";
import { createSentinelDownloadController } from "./sentinel-download-controller.js";
import { buildRouteUrl, parseRoute } from "./routing.js";

const state = {
  region: "gansu",
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

const mapUi = createMapController({
  target: mapEl,
  wrap: mapWrapEl,
  canShow: () => !isTrainingWorkspaceView(),
  isLayerVisible: layerVisible,
});
const {
  map,
  rasterLayer,
  vectorSources,
  vectorLayers,
  geojson,
  addLayerGeometry,
  addFeatureCollection,
  ensureMapVisible,
  clearVectorLayers,
  fitToBounds,
} = mapUi;

const {
  loadModelOptions,
  renderModelOptions,
  applyModelPrediction,
} = createModelValidationController({
  state,
  elements: {
    summary: modelSummaryEl,
    select: modelSelectEl,
    detail: modelDetailEl,
    random: modelRandomEl,
    imageToggle: toggleImageEl,
    predictionToggle: toggleModelPredictionEl,
  },
  mapUi,
  apiPath,
  activeApiPath,
  isAllRegions,
  activeRegionKey,
  localModelKeyForActiveRegion,
  currentRegionName,
  setSidebarMode,
  selectLake,
  updateRouteUrl,
  setLoading,
  renderMeta,
  showError,
});

const {
  renderPatchReview,
  normalizePatchPage,
  renderPatchModal,
} = createPatchReviewController({
  state,
  elements: {
    grid: patchGridEl,
    summary: patchReviewSummaryEl,
    pageLabel: patchPageLabelEl,
    previous: patchPrevEl,
    next: patchNextEl,
    waterFilter: patchWaterFilterEl,
    modal: patchModalEl,
    modalClose: patchModalCloseEl,
    modalTitle: patchModalTitleEl,
    modalSubtitle: patchModalSubtitleEl,
    modalImage: patchModalImageEl,
    modalMeta: patchModalMetaEl,
    modalToggle: patchModalToggleEl,
  },
  isAllRegions,
  updateTrainingPatch,
  setTrainingView,
  selectLake,
  showError,
});

const {
  loadTrainingRuns,
  renderTrainingRunView,
} = createTrainingRunController({
  state,
  elements: {
    summary: trainingRunSummaryEl,
    body: trainingRunBodyEl,
    trainingSummary: trainingSummaryEl,
    runName: trainRunNameEl,
    epochs: trainEpochsEl,
    batchSize: trainBatchSizeEl,
    lr: trainLrEl,
    baseChannels: trainBaseChannelsEl,
    device: trainDeviceEl,
    noAugment: trainNoAugmentEl,
    start: trainStartEl,
    cancel: trainCancelEl,
    status: trainStatusEl,
  },
  apiPath,
  isCurrentTrainingLoad,
  currentRegionName,
  loadModelOptions,
  showError,
});

const { loadTrainingSamples } = createTrainingSamplesController({
  state,
  list: trainingListEl,
  summary: trainingSummaryEl,
  apiPath,
  apiPathFor,
  isCurrentTrainingLoad,
  selectLake,
  showError,
});

createPatchExportController({
  elements: {
    patchSize: patchSizeEl,
    stride: patchStrideEl,
    previewScale: patchPreviewScaleEl,
    overwrite: patchOverwriteEl,
    start: patchExportEl,
    status: patchExportStatusEl,
  },
  apiPath,
  loadTrainingPatches,
  showError,
});

createSentinelDownloadController({
  state,
  elements: {
    tile: sentinelTileEl,
    startDate: sentinelStartEl,
    endDate: sentinelEndEl,
    cloud: sentinelCloudEl,
    query: sentinelQueryEl,
    products: sentinelProductsEl,
  },
  activeApiPath,
  apiPathFor,
  activeRegionKey,
  loadSentinelTiles,
  loadImageryOptions,
  renderMeta,
  showError,
});

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

function updateRouteUrl({ replace = false } = {}) {
  if (state.restoringUrl) return;
  const url = buildRouteUrl(state);
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
    trainingPanelEl.hidden = !state.imagery;
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


function currentRegionName(key) {
  if (!key) return "";
  if (key === "all") return "全部区域";
  return state.regions.find((region) => region.key === key)?.name || key;
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
  trainingPanelEl.hidden = !state.imagery;
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
      auto_patch: true,
      view_state: captureTrainingViewState(),
    });
    const similar = payload.sample.similar_samples || [];
    const warning = similar.length ? `；有 ${similar.length} 个相似视图` : "";
    if (payload.sample.duplicate) {
      trainingStatusEl.textContent = `训练区域已存在，已更新记录${warning}`;
    } else if (payload.patch_job?.job_id) {
      trainingStatusEl.textContent = `已加入训练区域，正在生成 patch${warning}`;
      pollTrainingSamplePatchJob(payload.patch_job.job_id, activeRegionKey()).catch(showError);
    } else {
      trainingStatusEl.textContent = `已加入训练区域：${payload.sample.sample_id}${warning}`;
    }
    if (state.sidebarMode === "training") await loadTrainingSamples();
  } finally {
    trainingSaveEl.disabled = !state.activeId || trainingPanelEl.hidden;
  }
}

async function pollTrainingSamplePatchJob(jobId, regionKey = activeRegionKey()) {
  const job = await fetchJson(apiPathFor(regionKey, `/training-patches/export-jobs/${encodeURIComponent(jobId)}`));
  if (job.status === "completed") {
    const count = job.result?.patches ?? 0;
    trainingStatusEl.textContent = `训练区域已保存，已生成 ${count} 个 patch`;
    if (state.sidebarMode === "training" && state.trainingView === "patches") await loadTrainingPatches();
    return;
  }
  if (job.status === "failed") {
    trainingStatusEl.textContent = `训练区域已保存，但 patch 生成失败：${job.message || "未知错误"}`;
    return;
  }
  trainingStatusEl.textContent = job.message || "正在生成 patch";
  setTimeout(() => pollTrainingSamplePatchJob(jobId, regionKey).catch(showError), 1500);
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
      model_prediction: state.sidebarMode === "model" && toggleModelPredictionEl.checked,
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
    model_prediction_excluded: true,
    model_validation: captureModelValidationState(),
    map: {
      center,
      zoom: view.getZoom(),
      extent,
    },
  };
}

function captureModelValidationState() {
  if (state.sidebarMode !== "model" && !state.modelValidation) return null;
  const validation = state.modelValidation || {};
  const model = validation.model || selectedModelOption() || {};
  const stats = validation.stats || {};
  return {
    model_key: state.selectedModel || model.key || "",
    model_name: model.name || model.label || "",
    model_path: model.path || "",
    model_weight: model.weight || "",
    threshold: Number(model.threshold ?? stats.threshold ?? 0.5),
    predicted_area_km2: Number(stats.area_km2 || 0),
    predicted_ratio: Number(stats.predicted_ratio || 0),
    prediction_feature_count: validation.prediction?.features?.length || 0,
    device: model.device || "",
  };
}

function selectedModelOption() {
  return state.modelOptions.find((item) => item.key === state.selectedModel) || null;
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

function setDefaultSentinelFilters() {
  const end = new Date();
  const start = new Date(end);
  start.setMonth(start.getMonth() - 2);
  sentinelStartEl.value = formatDateInput(start);
  sentinelEndEl.value = formatDateInput(end);
  sentinelCloudEl.value = "50";
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

tabLakesEl.addEventListener("click", () => {
  setSidebarMode("lakes");
});

tabTrainingEl.addEventListener("click", () => {
  setSidebarMode("training");
});

tabModelEl.addEventListener("click", () => {
  setSidebarMode("model");
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


trainStartEl.addEventListener("click", () => {
  startTrainingRun().catch(showError);
});

trainCancelEl.addEventListener("click", () => {
  cancelTrainingRun().catch(showError);
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
