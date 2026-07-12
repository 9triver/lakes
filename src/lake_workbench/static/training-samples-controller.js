import { deleteJson, fetchJson, patchJson } from "./api.js";
import {
  escapeHtml,
  formatImageryAssetLabels,
  formatLabelScope,
  formatMaskPolicy,
  formatPercentText,
  formatTrainingLabelSource,
} from "./formatters.js";

export function createTrainingSamplesController({
  state,
  list,
  summary,
  apiPath,
  apiPathFor,
  isCurrentTrainingLoad,
  selectLake,
  showError,
  confirmDelete = (message) => globalThis.confirm(message),
}) {
  async function loadTrainingSamples(loadId = null) {
    loadId = loadId ?? ++state.trainingLoadId;
    if (isCurrentTrainingLoad(loadId, "samples")) {
      list.replaceChildren();
      const loading = document.createElement("div");
      loading.className = "empty-list";
      loading.textContent = "训练样本加载中";
      list.append(loading);
      summary.textContent = "训练集加载中";
    }
    const payload = await fetchJson(apiPath("/training-samples"));
    if (!isCurrentTrainingLoad(loadId, "samples")) return;
    state.trainingSamples = payload.items || [];
    const missing = state.trainingSamples.filter((item) => item.status !== "ok").length;
    summary.textContent = missing ? `${payload.total} 个样本，${missing} 个缺文件` : `${payload.total} 个样本`;
    renderTrainingSamples();
  }

  function renderTrainingSamples() {
    list.replaceChildren();
    if (!state.trainingSamples.length) {
      const empty = document.createElement("div");
      empty.className = "empty-list";
      empty.textContent = "暂无训练样本";
      list.append(empty);
      return;
    }
    for (const sample of state.trainingSamples) {
      const item = document.createElement("div");
      item.className = `training-item${sample.status === "ok" ? "" : " missing"}`;
      const name = sample.lake_display_name || sample.lake_name || sample.lake_id || sample.sample_id;
      item.innerHTML = `
        <div class="training-top"><div class="training-name" title="${escapeHtml(name)}">${escapeHtml(name)}</div><div class="badge">${escapeHtml(sample.status === "ok" ? "ok" : "缺文件")}</div></div>
        <div class="training-meta-line">${escapeHtml(formatTrainingLabelSource(sample.label_source, sample.label_threshold))} · ${escapeHtml(formatLabelScope(sample.label_scope))} · ${escapeHtml(formatMaskPolicy(sample.mask_policy))}</div>
        <div class="training-meta-line">${escapeHtml(sample.tile_count || 0)} tile · ${escapeHtml(formatImageryAssetLabels(sample.imagery_asset_labels || sample.imagery_asset_label))} · ${escapeHtml(sample.product_date || "")}</div>
        ${sample.diagnostic_model_name || sample.diagnostic_model_key ? `<div class="training-meta-line">诊断模型 ${escapeHtml(sample.diagnostic_model_name || sample.diagnostic_model_key)} · 预测水体 ${escapeHtml(formatPercentText(sample.diagnostic_prediction_ratio))}</div>` : ""}
        <div class="training-meta-line" title="${escapeHtml(sample.sample_id || "")}">${escapeHtml(sample.sample_id || "")}</div>`;
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
      const open = button("定位", () => {
        state.activeRegion = sample.region || state.region;
        selectLake(sample.lake_id).catch(showError);
      });
      const save = button("保存", () => updateTrainingSample(sample.sample_id, { split: split.value, notes: notes.value }).catch(showError));
      const remove = button("删除", () => deleteTrainingSample(sample.sample_id).catch(showError), "danger");
      actions.append(open, save, remove);
      item.append(edit, actions);
      list.append(item);
    }
  }

  async function updateTrainingSample(sampleId, payload) {
    const sample = state.trainingSamples.find((item) => item.sample_id === sampleId) || {};
    await patchJson(apiPathFor(sample.region || state.region, `/training-samples/${encodeURIComponent(sampleId)}`), payload);
    await loadTrainingSamples();
  }

  async function deleteTrainingSample(sampleId) {
    if (!confirmDelete("删除这个训练样本记录？")) return;
    const sample = state.trainingSamples.find((item) => item.sample_id === sampleId) || {};
    await deleteJson(apiPathFor(sample.region || state.region, `/training-samples/${encodeURIComponent(sampleId)}`));
    await loadTrainingSamples();
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

  function button(label, onClick, className = "") {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.className = className;
    element.addEventListener("click", onClick);
    return element;
  }

  return { loadTrainingSamples, renderTrainingSamples };
}
