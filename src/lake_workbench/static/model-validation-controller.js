import { fetchJson } from "./api.js";
import { formatNumber } from "./formatters.js";
import { formatModelOptionText, renderModelDetail, sortModelOptions } from "./model-ui.js";

export function createModelValidationController({
  state,
  elements,
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
  ol = globalThis.ol,
}) {
  const { summary, select, detail, random, imageToggle, predictionToggle } = elements;

  async function loadModelOptions(preferred = state.selectedModel) {
    summary.textContent = "模型列表加载中";
    const payload = await fetchJson(apiPath("/model-validation/models"));
    state.modelOptions = sortModelOptions(payload.items || []);
    state.selectedModel = preferred && state.modelOptions.some((item) => item.key === preferred)
      ? preferred
      : payload.default || state.modelOptions[0]?.key || "";
    renderModelOptions();
    summary.textContent = state.modelOptions.length
      ? `已加载 ${state.modelOptions.length} 个模型权重`
      : "当前区域没有模型权重";
  }

  function renderModelOptions() {
    select.replaceChildren();
    if (!state.modelOptions.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "无可用模型";
      select.append(option);
      select.disabled = true;
      random.disabled = true;
      detail.textContent = "无可用模型";
      return;
    }
    select.disabled = false;
    random.disabled = state.modelValidationBusy;
    for (const item of state.modelOptions) {
      const option = document.createElement("option");
      option.value = item.key;
      option.textContent = item.error ? `${item.label}（不可用）` : formatModelOptionText(item);
      option.disabled = Boolean(item.error);
      option.selected = item.key === state.selectedModel;
      option.title = item.error || item.path || item.label;
      select.append(option);
    }
    select.value = state.selectedModel;
    renderSelectedModelDetail();
  }

  function renderSelectedModelDetail() {
    const model = state.modelOptions.find((item) => item.key === state.selectedModel);
    renderModelDetail(detail, model, currentRegionName);
  }

  async function runRandomModelValidation() {
    if (state.modelValidationBusy) return;
    const runId = state.modelValidationRunId + 1;
    state.modelValidationRunId = runId;
    state.modelValidationBusy = true;
    random.disabled = true;
    random.textContent = "验证中";
    summary.textContent = "模型推理中";
    setLoading(true, "模型推理中");
    try {
      mapUi.rasterLayer.setSource(null);
      imageToggle.checked = true;
      mapUi.rasterLayer.setVisible(true);
      predictionToggle.checked = true;
      mapUi.vectorLayers.modelPrediction.setVisible(true);
      const params = new URLSearchParams();
      if (state.selectedModel) params.set("model", state.selectedModel);
      const suffix = params.toString() ? `?${params.toString()}` : "";
      let payload;
      try {
        payload = await fetchJson(apiPath(`/model-validation/random${suffix}`));
      } catch (error) {
        if (error.status === 429) {
          summary.textContent = error.message || "模型推理正在运行，请稍后再试";
          return;
        }
        if (error instanceof TypeError) {
          summary.textContent = "服务暂时不可用，请稍后再试";
          return;
        }
        throw error;
      }
      if (runId !== state.modelValidationRunId) return;
      state.modelValidation = payload;
      setSidebarMode("model");
      state.activeRegion = payload.lake?.region || payload.region || activeRegionKey();
      await selectLake(payload.lake_id, { waitForTile: false, validationRunId: runId });
      if (runId !== state.modelValidationRunId || state.activeId !== payload.lake_id) return;
      applyModelPrediction(payload);
      updateRouteUrl();
    } finally {
      if (runId === state.modelValidationRunId) {
        state.modelValidationBusy = false;
        random.disabled = false;
        random.textContent = "随机验证一个湖泊";
        setLoading(false);
      }
    }
  }

  function applyModelPrediction(payload) {
    if (!payload?.prediction) return;
    state.modelValidation = payload;
    mapUi.addFeatureCollection("modelPrediction", payload.prediction);
    predictionToggle.checked = true;
    mapUi.vectorLayers.modelPrediction.setVisible(true);
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
    summary.textContent = `${lakeName} · ${count} 个预测斑块 · ${formatNumber(area, 3)} km²`;
    state.metaParts.model = [
      `模型预测 ${model.name || ""}`,
      `阈值 ${formatNumber(Number(model.threshold ?? stats.threshold ?? 0.5), 2)}`,
      `水体像元 ${formatNumber(ratio, 1)}%`,
      model.device ? `设备 ${model.device}` : "",
    ].filter(Boolean).join(" | ");
    renderMeta();
  }

  function fitToPredictionOrLake(payload) {
    const extent = mapUi.vectorSources.modelPrediction.getExtent();
    if (extent && extent.every(Number.isFinite) && !ol.extent.isEmpty(extent)) {
      mapUi.ensureMapVisible();
      mapUi.map.getView().fit(extent, { padding: [48, 48, 48, 48], duration: 180, maxZoom: 14 });
      return;
    }
    const bounds = payload?.lake?.bbox || state.lake?.bbox || state.tileMeta?.lake_bounds;
    if (bounds) mapUi.fitToBounds(bounds);
  }

  async function handleModelSelection() {
    state.selectedModel = select.value;
    renderSelectedModelDetail();
    updateRouteUrl({ replace: true });
    if (state.sidebarMode !== "model" || !state.activeId) return;
    const params = new URLSearchParams();
    const localModel = localModelKeyForActiveRegion();
    if (isAllRegions() && state.selectedModel && localModel === state.selectedModel) {
      summary.textContent = "当前模型不属于这个水体区域，请随机验证";
      return;
    }
    if (localModel) params.set("model", localModel);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    summary.textContent = "模型推理中";
    applyModelPrediction(await fetchJson(activeApiPath(`/lakes/${encodeURIComponent(state.activeId)}/model-prediction${suffix}`)));
  }

  random.addEventListener("click", () => runRandomModelValidation().catch(showError));
  select.addEventListener("change", () => handleModelSelection().catch(showError));

  return {
    loadModelOptions,
    renderModelOptions,
    renderSelectedModelDetail,
    runRandomModelValidation,
    applyModelPrediction,
  };
}
