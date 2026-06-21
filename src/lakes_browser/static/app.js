const state = {
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
  trainingReady: null,
  trainingSamples: [],
  sidebarMode: "lakes",
  downloadJobs: new Map(),
};

const tabLakesEl = document.querySelector("#tab-lakes");
const tabTrainingEl = document.querySelector("#tab-training");
const lakeSidebarPanelEl = document.querySelector("#lake-sidebar-panel");
const trainingSidebarPanelEl = document.querySelector("#training-sidebar-panel");
const listEl = document.querySelector("#lake-list");
const trainingListEl = document.querySelector("#training-list");
const countEl = document.querySelector("#count");
const trainingSummaryEl = document.querySelector("#training-summary");
const trainingRefreshEl = document.querySelector("#training-refresh");
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
const mapEl = document.querySelector("#map");
const loadingEl = document.querySelector("#loading");
const loadingTextEl = document.querySelector("#loading-text");
const toggleImageEl = document.querySelector("#toggle-image");
const toggleTileGridEl = document.querySelector("#toggle-tile-grid");
const toggleOsmEl = document.querySelector("#toggle-osm");
const toggleHydroEl = document.querySelector("#toggle-hydro");
const toggleEsaEl = document.querySelector("#toggle-esa");
const toggleJrcEl = document.querySelector("#toggle-jrc");
const jrcThresholdEl = document.querySelector("#jrc-threshold");
const jrcThresholdValueEl = document.querySelector("#jrc-threshold-value");
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
const trainingLabelSourceEl = document.querySelector("#training-label-source");
const trainingJrcThresholdEl = document.querySelector("#training-jrc-threshold");
const trainingQualityEl = document.querySelector("#training-quality");
const trainingNotesEl = document.querySelector("#training-notes");
const trainingSaveEl = document.querySelector("#training-save");
const trainingStatusEl = document.querySelector("#training-status");
const zoomLakeEl = document.querySelector("#zoom-lake");
const zoomTileEl = document.querySelector("#zoom-tile");

let searchTimer = null;
let jrcTimer = null;

setDefaultSentinelFilters();

const rasterLayer = new ol.layer.Tile({ visible: true });
const vectorSources = {
  tileGrid: new ol.source.Vector(),
  osm: new ol.source.Vector(),
  hydrolakes: new ol.source.Vector(),
  esa: new ol.source.Vector(),
  jrc: new ol.source.Vector(),
};
const vectorLayers = {
  tileGrid: new ol.layer.Vector({ source: vectorSources.tileGrid, style: tileGridStyle }),
  osm: new ol.layer.Vector({ source: vectorSources.osm, style: polygonStyle("#00a6ff", "rgba(0, 166, 255, 0.20)") }),
  hydrolakes: new ol.layer.Vector({ source: vectorSources.hydrolakes, style: polygonStyle("#ffd447", "rgba(255, 212, 71, 0.18)") }),
  esa: new ol.layer.Vector({ source: vectorSources.esa, style: polygonStyle("#ff4fb3", "rgba(255, 79, 179, 0.30)") }),
  jrc: new ol.layer.Vector({ source: vectorSources.jrc, style: polygonStyle("#1ab878", "rgba(44, 214, 137, 0.24)") }),
};
const map = new ol.Map({
  target: mapEl,
  layers: [rasterLayer, vectorLayers.tileGrid, vectorLayers.osm, vectorLayers.hydrolakes, vectorLayers.esa, vectorLayers.jrc],
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
  const payload = await fetchJson(`/api/lakes?${params.toString()}`);
  state.total = payload.total;
  state.lakes = append ? state.lakes.concat(payload.items) : payload.items;
  state.offset = state.lakes.length;
  countEl.textContent = `${payload.total} 个湖泊，显示 ${state.lakes.length} 个`;
  loadMoreEl.hidden = state.lakes.length >= payload.total;
  renderList();
}

async function loadTrainingSamples() {
  trainingSummaryEl.textContent = "训练集加载中";
  const payload = await fetchJson("/api/training-samples");
  state.trainingSamples = payload.items || [];
  const bad = state.trainingSamples.filter((item) => item.status !== "ok").length;
  trainingSummaryEl.textContent = bad
    ? `${payload.total} 个样本，${bad} 个缺文件`
    : `${payload.total} 个样本`;
  renderTrainingSamples();
}

function setSidebarMode(mode) {
  state.sidebarMode = mode;
  const trainingMode = mode === "training";
  tabLakesEl.classList.toggle("active", !trainingMode);
  tabTrainingEl.classList.toggle("active", trainingMode);
  lakeSidebarPanelEl.hidden = trainingMode;
  listEl.hidden = trainingMode;
  loadMoreEl.hidden = trainingMode || state.lakes.length >= state.total;
  trainingSidebarPanelEl.hidden = !trainingMode;
  trainingListEl.hidden = !trainingMode;
  if (trainingMode) loadTrainingSamples().catch(showError);
}

function renderList() {
  listEl.replaceChildren();
  for (const lake of state.lakes) {
    const button = document.createElement("button");
    button.className = `lake-item${state.activeId === lake.object_id ? " active" : ""}${state.loadingId === lake.object_id ? " loading" : ""}`;
    button.type = "button";
    button.addEventListener("click", () => selectLake(lake.object_id).catch(showError));
    const label = lake.display_name || lake.name || lake.object_id;
    button.innerHTML = `
      <div class="lake-row">
        <div class="lake-id">${escapeHtml(label)}</div>
        <div class="badge">${escapeHtml(typeLabel(lake.water_type))}${lake.has_tci ? " · TCI" : ""}</div>
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
      <div class="training-meta-line">${escapeHtml(sample.label_source || "")}${sample.label_threshold ? ` ${escapeHtml(sample.label_threshold)}` : ""} · ${escapeHtml(sample.tile_count || 0)} tile · ${escapeHtml(sample.product_date || "")}</div>
      <div class="training-meta-line" title="${escapeHtml(sample.sample_id || "")}">${escapeHtml(sample.sample_id || "")}</div>
    `;
    const edit = document.createElement("div");
    edit.className = "training-edit";
    const quality = makeSelect(["good", "usable", "needs_edit", "bad"], sample.quality || "good");
    const split = makeSelect(["", "train", "val", "test"], sample.split || "");
    const notes = document.createElement("input");
    notes.type = "text";
    notes.value = sample.notes || "";
    notes.placeholder = "备注";
    edit.append(quality, split, notes);

    const actions = document.createElement("div");
    actions.className = "training-actions";
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "定位";
    open.addEventListener("click", () => selectLake(sample.lake_id).catch(showError));
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "保存";
    save.addEventListener("click", () => updateTrainingSample(sample.sample_id, {
      quality: quality.value,
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
  await patchJson(`/api/training-samples/${encodeURIComponent(sampleId)}`, payload);
  await loadTrainingSamples();
}

async function deleteTrainingSample(sampleId) {
  if (!confirm("删除这个训练样本记录？")) return;
  await deleteJson(`/api/training-samples/${encodeURIComponent(sampleId)}`);
  await loadTrainingSamples();
}

async function selectLake(shapeId) {
  state.activeId = shapeId;
  state.loadingId = shapeId;
  state.metaParts = {};
  state.imagery = null;
  state.trainingReady = null;
  renderTrainingReadiness();
  state.tileMeta = null;
  sentinelPanelEl.hidden = true;
  trainingPanelEl.hidden = true;
  sentinelProductsEl.replaceChildren();
  imageryProductEl.replaceChildren();
  clearVectorLayers();
  renderList();
  titleEl.textContent = `水体 ${shapeId}`;
  subtitleEl.textContent = "加载影像和边界";
  setLoading(true, "加载地图数据");
  emptyEl.hidden = true;
  mapEl.hidden = false;
  map.updateSize();

  const lake = await fetchJson(`/api/lakes/${shapeId}`);
  if (state.activeId !== shapeId) return;
  state.lake = lake;
  titleEl.textContent = lake.display_name || lake.name || `水体 ${lake.object_id}`;
  const hylak = lake.layers?.hydrolakes?.properties?.Hylak_id;
  subtitleEl.textContent = `${typeLabel(lake.water_type)} · ${lake.lake_id}${hylak ? ` · Hylak ${hylak}` : ""}`;
  addLayerGeometry("osm", lake.layers?.osm);
  addLayerGeometry("hydrolakes", lake.layers?.hydrolakes);
  await loadTileLayer(shapeId, lake).catch((error) => {
    if (!isMissingTciError(error)) throw error;
    rasterLayer.setSource(null);
    state.tileMeta = null;
    fitToBounds(lake.bbox);
    state.metaParts.base = "暂无本地 Sentinel 影像，可查询并下载产品";
    renderMeta();
    setLoading(false);
  });
  state.loadingId = null;
  renderList();
  loadSentinelTiles(shapeId).catch(showError);
  loadImageryOptions(shapeId).catch(showError);
  loadEsaLayer(shapeId).catch(showError);
  loadJrcLayer(shapeId).catch(showError);
}

async function loadTileLayer(shapeId, lake) {
  setLoading(true, "加载影像瓦片");
  const payload = await fetchJson(`/api/lakes/${shapeId}/tile-meta?padding=0.8&v=${Date.now()}`);
  if (state.activeId !== shapeId) return;
  state.tileMeta = payload;
  rasterLayer.setSource(
    new ol.source.XYZ({
      url: `/api/lakes/${shapeId}/tiles/{z}/{x}/{y}.png?v=${Date.now()}`,
      tileSize: 256,
      minZoom: 5,
      maxZoom: 16,
      transition: 120,
    }),
  );
  rasterLayer.setVisible(toggleImageEl.checked);
  fitToBounds(payload.lake_bounds || payload.bounds || lake.bbox);
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

async function loadEsaLayer(shapeId) {
  state.metaParts.esa = "ESA 平滑边界生成中";
  renderMeta();
  const payload = await fetchJson(`/api/lakes/${shapeId}/esa`);
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
  const payload = await fetchJson(`/api/lakes/${shapeId}/jrc?threshold=${threshold}`);
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

async function loadSentinelTiles(shapeId) {
  const payload = await fetchJson(`/api/lakes/${shapeId}/sentinel/tiles`);
  if (state.activeId !== shapeId) return;
  renderTileGrid(payload.tiles || []);
  sentinelTileEl.replaceChildren();
  for (const item of payload.tiles) {
    const option = document.createElement("option");
    option.value = item.tile;
    option.textContent = item.tile;
    sentinelTileEl.append(option);
  }
  sentinelPanelEl.hidden = payload.tiles.length === 0;
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
  const payload = await fetchJson(`/api/lakes/${shapeId}/imagery`);
  if (state.activeId !== shapeId) return;
  state.imagery = payload;
  trainingPanelEl.hidden = false;
  renderImageryOptions();
  await loadTrainingReadiness(shapeId);
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
    option.textContent = `${product.active ? "当前，" : ""}时间: ${formatDateText(product.date)}，云量 ${formatCloud(product.cloud_cover)}`;
    imageryProductEl.append(option);
    if (product.active) imageryProductEl.value = product.product;
  }
  renderTrainingReadiness();
}

async function loadTrainingReadiness(shapeId) {
  state.trainingReady = null;
  renderTrainingReadiness();
  const payload = await fetchJson(`/api/lakes/${shapeId}/training-samples/readiness?buffer_ratio=0.8`);
  if (state.activeId !== shapeId) return;
  state.trainingReady = payload;
  renderTrainingReadiness();
}

function renderTrainingReadiness() {
  const ready = state.trainingReady;
  if (!ready) {
    trainingSaveEl.disabled = true;
    trainingStatusEl.textContent = trainingPanelEl.hidden ? "" : "检查训练数据";
    return;
  }
  trainingSaveEl.disabled = !ready.ready;
  if (ready.ready) {
    const tiles = (ready.required_tiles || []).join(", ");
    trainingStatusEl.textContent = `训练数据完备：${ready.ready_count}/${ready.required_count} 个 tile 已设为影像；${tiles}`;
    return;
  }
  const missing = (ready.missing_tiles || []).join(", ");
  const selected = `${ready.ready_count || 0}/${ready.required_count || 0}`;
  trainingStatusEl.textContent = missing
    ? `训练数据不完备：${selected} 个 tile 已设为影像；缺少 ${missing}`
    : `训练数据不完备：${selected} 个 tile 已设为影像`;
}

function syncTrainingLabelControls() {
  const isJrc = trainingLabelSourceEl.value === "jrc";
  trainingJrcThresholdEl.hidden = !isJrc;
  trainingJrcThresholdEl.disabled = !isJrc;
}

async function saveTrainingSample() {
  if (!state.activeId) return;
  trainingSaveEl.disabled = true;
  trainingStatusEl.textContent = "检查训练数据";
  await loadTrainingReadiness(state.activeId);
  if (!state.trainingReady?.ready) {
    renderTrainingReadiness();
    return;
  }
  trainingSaveEl.disabled = true;
  trainingStatusEl.textContent = "保存中";
  const labelSource = trainingLabelSourceEl.value;
  try {
    const payload = await postJson(`/api/lakes/${state.activeId}/training-samples`, {
      label_source: labelSource,
      label_threshold: labelSource === "jrc" ? trainingJrcThresholdEl.value : "",
      quality: trainingQualityEl.value,
      notes: trainingNotesEl.value,
      buffer_ratio: 0.8,
    });
    trainingStatusEl.textContent = `已加入训练集：${payload.sample.sample_id}`;
    if (state.sidebarMode === "training") await loadTrainingSamples();
  } finally {
    trainingSaveEl.disabled = !state.trainingReady?.ready;
  }
}

async function applyImagerySelection() {
  if (!state.activeId || !sentinelTileEl.value || !imageryProductEl.value) return;
  imageryApplyEl.disabled = true;
  setLoading(true, "切换影像瓦片");
  try {
    await postJson(`/api/lakes/${state.activeId}/imagery/active`, {
      tile: sentinelTileEl.value,
      product: imageryProductEl.value,
    });
    state.metaParts.sentinel = `已切换 ${sentinelTileEl.value} 影像`;
    renderMeta();
    await loadImageryOptions(state.activeId);
    await loadTileLayer(state.activeId, state.lake);
    await loadTrainingReadiness(state.activeId);
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
    const payload = await fetchJson(`/api/sentinel/products?${params.toString()}`);
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
  const job = await postJson("/api/sentinel/downloads", { product });
  if (!job.job_id) {
    button.textContent = job.message || "已下载";
    await loadImageryOptions(state.activeId);
    return;
  }
  state.downloadJobs.set(job.job_id, { button, product });
  pollDownloadJob(job.job_id).catch(showError);
}

async function pollDownloadJob(jobId) {
  const entry = state.downloadJobs.get(jobId);
  if (!entry) return;
  const job = await fetchJson(`/api/sentinel/downloads/${jobId}`);
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

function clearVectorLayers() {
  for (const source of Object.values(vectorSources)) source.clear();
}

function fitToBounds(bounds) {
  if (!bounds || bounds.length !== 4) return;
  const extent = ol.proj.transformExtent(bounds, "EPSG:4326", "EPSG:3857");
  map.updateSize();
  map.getView().fit(extent, {
    padding: [36, 36, 36, 36],
    duration: 180,
    maxZoom: 14,
  });
}

function polygonStyle(stroke, fill) {
  return new ol.style.Style({
    stroke: new ol.style.Stroke({ color: stroke, width: 2 }),
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
  if (layerName === "esa") return toggleEsaEl.checked;
  if (layerName === "jrc") return toggleJrcEl.checked;
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
  return products
    .filter(Boolean)
    .map((item) => {
      const tile = item.match(/_T([0-9A-Z]{5})_/)?.[1] || "";
      const date = item.match(/MSIL\d[AC]?_(\d{8})T/)?.[1] || "";
      return [tile, date].filter(Boolean).join("/");
    })
    .filter(Boolean)
    .join(", ");
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

function formatDateText(value) {
  const text = String(value || "").trim();
  if (/^\d{8}$/.test(text)) return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
  return text || "未知";
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
  metaEl.textContent = [state.metaParts.base, state.metaParts.esa, state.metaParts.jrc, state.metaParts.sentinel]
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
    loadLakes().catch(showError);
  }, 180);
});

for (const select of [filterTypeEl, filterAreaEl, filterNameEl, filterTciEl, filterPolygonQualityEl, filterMetadataQualityEl]) {
  select.addEventListener("change", () => {
    state.filters = currentFilters();
    loadLakes().catch(showError);
  });
}

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
});

for (const [checkbox, layerName] of [
  [toggleTileGridEl, "tileGrid"],
  [toggleOsmEl, "osm"],
  [toggleHydroEl, "hydrolakes"],
  [toggleEsaEl, "esa"],
  [toggleJrcEl, "jrc"],
]) {
  checkbox.addEventListener("change", () => {
    vectorLayers[layerName].setVisible(checkbox.checked);
  });
}

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

trainingRefreshEl.addEventListener("click", () => {
  loadTrainingSamples().catch(showError);
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

trainingLabelSourceEl.addEventListener("change", () => {
  syncTrainingLabelControls();
});

syncTrainingLabelControls();

zoomLakeEl.addEventListener("click", () => {
  if (!state.tileMeta) return;
  fitToBounds(state.tileMeta.lake_bounds || state.tileMeta.bounds);
});

zoomTileEl.addEventListener("click", () => {
  if (!state.tileMeta) return;
  fitToBounds(state.tileMeta.tile_bounds || state.tileMeta.bounds);
});

window.addEventListener("resize", () => {
  map.updateSize();
});

function showError(error) {
  console.error(error);
  setLoading(false);
  state.loadingId = null;
  renderList();
  metaEl.textContent = error.message;
}

loadLakes().catch(showError);
