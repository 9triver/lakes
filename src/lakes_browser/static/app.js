const state = {
  lakes: [],
  activeId: null,
  imageMeta: null,
  lake: null,
  zoom: 1,
  baseScale: 1,
  panX: 0,
  panY: 0,
  imageWidth: 0,
  imageHeight: 0,
  loadingId: null,
  total: 0,
  offset: 0,
  limit: 300,
  query: "",
  filters: {},
  metaParts: {},
  imagery: null,
  downloadJobs: new Map(),
};

const listEl = document.querySelector("#lake-list");
const countEl = document.querySelector("#count");
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
const mapWrapEl = document.querySelector(".map-wrap");
const loadingEl = document.querySelector("#loading");
const loadingTextEl = document.querySelector("#loading-text");
const rasterEl = document.querySelector("#raster");
const overlayEl = document.querySelector("#overlay");
const toggleImageEl = document.querySelector("#toggle-image");
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

let searchTimer = null;
let jrcTimer = null;
let dragState = null;

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
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
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
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
        <span>tile ${escapeHtml((lake.best_tci_tile || lake.tiles.slice(0, 2).join(", ")).toString())}</span>
        <span>${formatNumber(lake.center[0], 4)}, ${formatNumber(lake.center[1], 4)}</span>
        <span>${escapeHtml(lake.best_tci_date || "")} · ${escapeHtml(lake.metadata_quality || "")}</span>
      </div>
    `;
    listEl.append(button);
  }
}

async function selectLake(shapeId) {
  state.activeId = shapeId;
  state.loadingId = shapeId;
  state.zoom = 1;
  state.baseScale = 1;
  state.panX = 0;
  state.panY = 0;
  state.imageWidth = 0;
  state.imageHeight = 0;
  state.metaParts = {};
  state.imagery = null;
  sentinelPanelEl.hidden = true;
  sentinelProductsEl.replaceChildren();
  imageryProductEl.replaceChildren();
  renderList();
  titleEl.textContent = `水体 ${shapeId}`;
  subtitleEl.textContent = "加载影像和边界";
  setLoading(true, "加载水体边界");
  emptyEl.hidden = true;
  mapEl.hidden = false;
  overlayEl.replaceChildren();
  applyZoom();
  renderList();

  const lake = await fetchJson(`/api/lakes/${shapeId}`);
  if (state.activeId !== shapeId) return;
  setLoading(true, "裁剪可见光影像");
  state.lake = lake;
  await loadLakeImage(shapeId, lake);
}

async function loadLakeImage(shapeId, lake) {
  const imageUrl = `/api/lakes/${shapeId}/image.png?size=1000&padding=0.8&v=${Date.now()}`;
  const response = await fetch(imageUrl, { cache: "no-store" });
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.error || response.statusText);
  }
  state.imageMeta = JSON.parse(response.headers.get("X-Image-Meta"));
  const blob = await response.blob();
  rasterEl.src = URL.createObjectURL(blob);
  rasterEl.onload = () => {
    if (state.activeId !== shapeId) return;
    layoutMap(state.imageMeta.width, state.imageMeta.height);
    drawLayers(lake.layers, state.imageMeta.bounds);
    setLoading(false);
    state.loadingId = null;
    renderList();
    loadSentinelTiles(shapeId).catch(showError);
    loadImageryOptions(shapeId).catch(showError);
    loadEsaLayer(shapeId).catch(showError);
    loadJrcLayer(shapeId).catch(showError);
  };
  titleEl.textContent = lake.display_name || lake.name || `水体 ${lake.object_id}`;
  const hylak = lake.layers?.hydrolakes?.properties?.Hylak_id;
  subtitleEl.textContent = `${typeLabel(lake.water_type)} · ${lake.lake_id}${hylak ? ` · Hylak ${hylak}` : ""}`;
  state.metaParts.base = [
    `影像 tile ${formatMetaList(state.imageMeta.tiles || state.imageMeta.tile)}`,
    `日期 ${state.imageMeta.date}`,
    `产品 ${formatProductList(state.imageMeta.products || state.imageMeta.product)}`,
    `覆盖率 ${formatNumber(Number(state.imageMeta.valid_ratio) * 100, 1)}%`,
    `范围 ${state.imageMeta.bounds.map((v) => formatNumber(v, 5)).join(", ")}`,
  ].join(" | ");
  renderMeta();
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
  drawLayers(state.lake.layers, state.imageMeta.bounds);
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
    state.lake.layers.jrc = payload.jrc;
    drawLayers(state.lake.layers, state.imageMeta.bounds);
    renderMeta();
    return;
  }
  state.lake.layers.jrc = payload.jrc;
  drawLayers(state.lake.layers, state.imageMeta.bounds);
  state.metaParts.jrc = `JRC ${threshold}% 边界已加载`;
  renderMeta();
}

async function loadSentinelTiles(shapeId) {
  const payload = await fetchJson(`/api/lakes/${shapeId}/sentinel/tiles`);
  if (state.activeId !== shapeId) return;
  sentinelTileEl.replaceChildren();
  for (const item of payload.tiles) {
    const option = document.createElement("option");
    option.value = item.tile;
    option.textContent = `${item.tile}${item.downloaded ? " 已下载" : " 未下载"}${item.date ? ` ${item.date}` : ""}`;
    sentinelTileEl.append(option);
  }
  sentinelPanelEl.hidden = payload.tiles.length === 0;
  renderImageryOptions();
}

async function loadImageryOptions(shapeId) {
  const payload = await fetchJson(`/api/lakes/${shapeId}/imagery`);
  if (state.activeId !== shapeId) return;
  state.imagery = payload;
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
    return;
  }
  for (const product of products) {
    const option = document.createElement("option");
    option.value = product.product;
    option.textContent = `${product.active ? "当前 " : ""}${product.date || ""} ${product.source || ""} ${formatNumber(Number(product.valid_ratio) * 100, 1)}%`;
    imageryProductEl.append(option);
    if (product.active) imageryProductEl.value = product.product;
  }
}

async function applyImagerySelection() {
  if (!state.activeId || !sentinelTileEl.value || !imageryProductEl.value) return;
  imageryApplyEl.disabled = true;
  setLoading(true, "切换影像");
  try {
    await postJson(`/api/lakes/${state.activeId}/imagery/active`, {
      tile: sentinelTileEl.value,
      product: imageryProductEl.value,
    });
    state.metaParts.sentinel = `已切换 ${sentinelTileEl.value} 影像`;
    renderMeta();
    await loadImageryOptions(state.activeId);
    await loadLakeImage(state.activeId, state.lake);
    setLoading(false);
  } finally {
    imageryApplyEl.disabled = false;
  }
}

async function querySentinelProducts() {
  if (!sentinelTileEl.value) return;
  sentinelQueryEl.disabled = true;
  sentinelProductsEl.textContent = "查询中";
  const params = new URLSearchParams({
    tile: sentinelTileEl.value,
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

function layoutMap(width, height) {
  state.imageWidth = width;
  state.imageHeight = height;
  mapEl.style.width = `${width}px`;
  mapEl.style.height = `${height}px`;
  overlayEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
  requestAnimationFrame(() => {
    fitMapToView();
    applyZoom();
  });
}

function drawLayers(layers, bounds) {
  overlayEl.replaceChildren();
  const items = [
    ["osm", layers?.osm?.geometry, "polygon polygon-osm", toggleOsmEl.checked],
    ["hydrolakes", layers?.hydrolakes?.geometry, "polygon polygon-hydro", toggleHydroEl.checked],
    ["esa", layers?.esa?.geometry, "polygon polygon-esa", toggleEsaEl.checked],
    ["jrc", layers?.jrc?.geometry, "polygon polygon-jrc", toggleJrcEl.checked],
  ];
  for (const [, geometry, className, visible] of items) {
    if (!visible || !geometry) continue;
    drawPolygon(geometry, bounds, className);
  }
}

function drawPolygon(geometry, bounds, className) {
  const rings = geometryToRings(geometry);
  for (const ring of rings) {
    const points = ring.map(([lon, lat]) => projectPoint(lon, lat, bounds, state.imageMeta.width, state.imageMeta.height));
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", className);
    path.setAttribute("d", pointsToPath(points));
    overlayEl.append(path);
  }
}

function geometryToRings(geometry) {
  if (!geometry) return [];
  if (geometry.type === "Polygon") {
    return [geometry.coordinates[0]];
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.map((poly) => poly[0]);
  }
  return [];
}

function projectPoint(lon, lat, bounds, width, height) {
  const [west, south, east, north] = bounds;
  const x = ((lon - west) / (east - west)) * width;
  const y = ((north - lat) / (north - south)) * height;
  return [x, y];
}

function pointsToPath(points) {
  if (!points.length) return "";
  const [first, ...rest] = points;
  return `M ${first[0].toFixed(2)} ${first[1].toFixed(2)} ${rest
    .map(([x, y]) => `L ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ")} Z`;
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

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return map[char];
  });
}

function setLoading(visible, text = "加载中") {
  loadingEl.hidden = !visible;
  loadingTextEl.textContent = text;
  if (visible) {
    metaEl.textContent = text;
  }
}

function renderMeta() {
  metaEl.textContent = [state.metaParts.base, state.metaParts.esa, state.metaParts.jrc, state.metaParts.sentinel]
    .filter(Boolean)
    .join(" | ");
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
  rasterEl.style.visibility = toggleImageEl.checked ? "visible" : "hidden";
});

for (const checkbox of [toggleOsmEl, toggleHydroEl, toggleEsaEl, toggleJrcEl]) {
  checkbox.addEventListener("change", () => {
    if (!state.lake || !state.imageMeta) return;
    drawLayers(state.lake.layers, state.imageMeta.bounds);
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

sentinelTileEl.addEventListener("change", () => {
  renderImageryOptions();
});

imageryApplyEl.addEventListener("click", () => {
  applyImagerySelection().catch(showError);
});

rasterEl.addEventListener("dragstart", (event) => event.preventDefault());

window.addEventListener("resize", () => {
  if (!state.imageMeta) return;
  fitMapToView();
  applyZoom();
});

mapWrapEl.addEventListener(
  "wheel",
  (event) => {
    if (mapEl.hidden || !state.imageMeta) return;
    event.preventDefault();
    const viewport = mapWrapEl.getBoundingClientRect();
    const pointerX = event.clientX - viewport.left;
    const pointerY = event.clientY - viewport.top;
    const mapX = (pointerX - state.panX) / currentScale();
    const mapY = (pointerY - state.panY) / currentScale();
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    const nextZoom = clamp(state.zoom * factor, 0.65, 12);
    if (Math.abs(nextZoom - state.zoom) < 0.001) return;
    state.zoom = nextZoom;
    state.panX = pointerX - mapX * currentScale();
    state.panY = pointerY - mapY * currentScale();
    applyZoom();
  },
  { passive: false },
);

mapWrapEl.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || mapEl.hidden || !state.imageMeta) return;
  event.preventDefault();
  dragState = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    panX: state.panX,
    panY: state.panY,
  };
  mapWrapEl.classList.add("dragging");
  mapWrapEl.setPointerCapture(event.pointerId);
});

mapWrapEl.addEventListener("pointermove", (event) => {
  if (!dragState || dragState.pointerId !== event.pointerId) return;
  event.preventDefault();
  state.panX = dragState.panX + event.clientX - dragState.x;
  state.panY = dragState.panY + event.clientY - dragState.y;
  applyZoom();
});

function endDrag(event) {
  if (!dragState || dragState.pointerId !== event.pointerId) return;
  dragState = null;
  mapWrapEl.classList.remove("dragging");
}

mapWrapEl.addEventListener("pointerup", endDrag);
mapWrapEl.addEventListener("pointercancel", endDrag);
mapWrapEl.addEventListener("lostpointercapture", () => {
  dragState = null;
  mapWrapEl.classList.remove("dragging");
});

mapWrapEl.addEventListener("dblclick", () => {
  if (!state.imageMeta) return;
  fitMapToView();
  applyZoom();
});

function applyZoom() {
  mapEl.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${currentScale()})`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
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

function currentScale() {
  return state.baseScale * state.zoom;
}

function fitMapToView() {
  const wrap = mapWrapEl.getBoundingClientRect();
  if (!state.imageWidth || !state.imageHeight || !wrap.width || !wrap.height) return;
  state.baseScale = Math.min(wrap.width / state.imageWidth, wrap.height / state.imageHeight) * 0.94;
  state.zoom = 1;
  const scale = currentScale();
  state.panX = (wrap.width - state.imageWidth * scale) / 2;
  state.panY = (wrap.height - state.imageHeight * scale) / 2;
}

function showError(error) {
  console.error(error);
  setLoading(false);
  state.loadingId = null;
  renderList();
  metaEl.textContent = error.message;
}

loadLakes().catch(showError);
