import { fetchJson, postJson } from "./api.js";
import { escapeHtml, formatDateTimeText, formatNumber, formatPercentText, trainingStatusLabel } from "./formatters.js";

const RUNNING_STATUSES = new Set(["queued", "configured", "running", "cancel_requested"]);

export function createTrainingRunController({
  state,
  elements,
  apiPath,
  isCurrentTrainingLoad,
  currentRegionName,
  loadModelOptions,
  showError,
}) {
  const { summary, body, trainingSummary, runName, epochs, batchSize, lr, baseChannels, device, noAugment, start, cancel, status } = elements;

  async function loadTrainingRuns(loadId = null) {
    loadId = loadId ?? ++state.trainingLoadId;
    if (isCurrentTrainingLoad(loadId, "train")) {
      summary.textContent = "训练任务加载中";
      body.innerHTML = `<div class="training-run-empty">训练任务加载中</div>`;
      trainingSummary.textContent = "训练任务加载中";
    }
    const payload = await fetchJson(apiPath("/training-runs"));
    if (!isCurrentTrainingLoad(loadId, "train")) return;
    state.trainingDataset = payload.dataset || null;
    state.trainingRuns = payload.items || [];
    const running = state.trainingRuns.find((job) => RUNNING_STATUSES.has(job.status));
    state.activeTrainingJob = running || state.trainingRuns[0] || state.activeTrainingJob;
    trainingSummary.textContent = running ? `训练中：${running.message || running.status}` : `${state.trainingRuns.length} 个训练任务`;
    renderTrainingRunView();
  }

  async function startTrainingRun() {
    start.disabled = true;
    cancel.disabled = false;
    status.textContent = "提交训练任务";
    const payload = {
      run_name: runName.value.trim(),
      epochs: Number(epochs.value || 30),
      batch_size: Number(batchSize.value || 8),
      lr: Number(lr.value || 0.001),
      base_channels: Number(baseChannels.value || 32),
      device: device.value || "cuda",
      no_augment: noAugment.checked,
    };
    try {
      const job = await postJson(apiPath("/training-runs"), payload);
      state.activeTrainingJob = job;
      renderTrainingRunView();
      pollTrainingRun(job.job_id).catch(showError);
    } catch (error) {
      start.disabled = false;
      cancel.disabled = true;
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
    const running = RUNNING_STATUSES.has(job.status);
    start.disabled = running;
    cancel.disabled = !running;
    status.textContent = job.message || job.status || "处理中";
    if (running) {
      setTimeout(() => pollTrainingRun(jobId).catch(showError), 2000);
    } else if (job.status === "completed") {
      await loadModelOptions();
    }
  }

  async function cancelTrainingRun() {
    if (!state.activeTrainingJob?.job_id) return;
    cancel.disabled = true;
    status.textContent = "正在取消";
    state.activeTrainingJob = await postJson(apiPath(`/training-runs/${encodeURIComponent(state.activeTrainingJob.job_id)}/cancel`), {});
    renderTrainingRunView();
  }

  function renderTrainingRunView() {
    const job = state.activeTrainingJob;
    body.replaceChildren();
    const running = job && RUNNING_STATUSES.has(job.status);
    start.disabled = Boolean(running);
    cancel.disabled = !running;
    const dataset = job?.dataset || job?.result?.dataset || state.trainingDataset;
    if (!job) {
      summary.textContent = "选择参数后开始训练";
      status.textContent = "未开始";
      body.append(renderTrainingDatasetPanel(dataset, "当前训练数据"));
      return;
    }
    const result = job.result || {};
    const config = job.config || result.config || {};
    const history = job.history || result.history || [];
    const latest = history[history.length - 1] || job.record || {};
    const train = latest.train || {};
    const val = latest.val || {};
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    summary.textContent = [
      trainingStatusLabel(job.status),
      job.epoch && job.epochs ? `epoch ${job.epoch}/${job.epochs}` : "",
      result.best_model ? `best ${result.best_model}` : "",
    ].filter(Boolean).join(" · ");
    status.textContent = job.message || trainingStatusLabel(job.status);
    const metrics = [
      ["状态", trainingStatusLabel(job.status)], ["进度", `${formatNumber(progress, 0)}%`],
      ["epoch", job.epoch && job.epochs ? `${job.epoch}/${job.epochs}` : "0"],
      ["train IoU", finiteMetric(train.iou)], ["val IoU", finiteMetric(val.iou)],
      ["train Dice", finiteMetric(train.dice)], ["val Dice", finiteMetric(val.dice)],
      ["输入波段", config.in_channels || ""], ["模型宽度", config.base_channels || ""],
      ["设备", config.device || ""], ["输出", result.output_dir || job.output_dir || config.output_dir || ""],
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
    const lines = history.slice(-12).map((record) => `epoch ${record.epoch}: train_iou=${formatNumber(Number(record.train?.iou || 0), 4)} val_iou=${formatNumber(Number(record.val?.iou || 0), 4)} train_loss=${formatNumber(Number(record.train?.loss || 0), 4)} val_loss=${formatNumber(Number(record.val?.loss || 0), 4)}`);
    log.textContent = lines.length ? lines.join("\n") : (job.message || "等待训练日志");
    const previous = document.createElement("div");
    previous.className = "training-run-history";
    previous.innerHTML = `<div class="training-run-history-title">历史任务 ${state.trainingRuns.length}</div>`;
    const list = document.createElement("div");
    list.className = "training-run-history-list";
    for (const item of state.trainingRuns) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = item.job_id === job.job_id ? "active" : "";
      row.title = item.result?.output_dir || item.message || item.job_id;
      row.innerHTML = formatTrainingHistoryRow(item);
      row.addEventListener("click", () => { state.activeTrainingJob = item; renderTrainingRunView(); });
      list.append(row);
    }
    previous.append(list);
    body.append(renderTrainingDatasetPanel(dataset, "本任务数据"), bar, cards, log, previous);
  }

  function renderTrainingDatasetPanel(dataset, title) {
    const panel = document.createElement("section");
    panel.className = "training-dataset-panel";
    const scopeLabel = dataset?.scope === "all" ? "全部区域" : (currentRegionName(dataset?.scope) || dataset?.scope || "当前区域");
    panel.innerHTML = `<div class="training-dataset-heading"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(scopeLabel)}</span></div>`;
    if (!dataset || dataset.error) {
      const empty = document.createElement("div");
      empty.className = "training-run-empty";
      empty.textContent = dataset?.error || "还没有可用于训练的 patch，请先在 Patch 页生成。";
      panel.append(empty);
      return panel;
    }
    const patchSize = Array.isArray(dataset.patch_size) ? dataset.patch_size.join(" x ") : "";
    const fields = [
      ["范围", scopeLabel], ["区域", (dataset.regions || []).join(", ")],
      ["可用 Patch", String(dataset.usable_patches ?? dataset.included_patches ?? 0)],
      ["包含/排除", `${dataset.included_patches || 0} / ${dataset.excluded_patches || 0}`],
      ["样本", String(dataset.sample_count || 0)], ["水体", String(dataset.lake_count || 0)],
      ["训练/验证", dataset.train_count != null || dataset.val_count != null ? `${dataset.train_count || 0} / ${dataset.val_count || 0}` : ""],
      ["输入", [dataset.in_channels ? `${dataset.in_channels} band` : "", patchSize].filter(Boolean).join(" · ")],
      ["水体像元", formatPercentText(dataset.water_ratio)],
      ["Manifest", (dataset.manifests || []).map((item) => item.manifest).filter(Boolean).join(" | ")],
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

  function formatTrainingHistoryRow(item) {
    const result = item.result || {};
    const config = item.config || result.config || {};
    const history = item.history || result.history || [];
    const latest = history[history.length - 1] || {};
    const name = item.run_name || config.output_dir?.split("/")?.pop() || item.job_id;
    const metrics = [
      item.epoch ? `epoch ${item.epochs ? `${item.epoch}/${item.epochs}` : item.epoch}` : "",
      Number.isFinite(Number(result.best_iou ?? latest.val?.iou)) ? `best ${formatNumber(Number(result.best_iou ?? latest.val?.iou), 4)}` : "",
      Number.isFinite(Number(latest.val?.iou)) ? `val ${formatNumber(Number(latest.val.iou), 4)}` : "",
      Number.isFinite(Number(latest.train?.iou)) ? `train ${formatNumber(Number(latest.train.iou), 4)}` : "",
    ].filter(Boolean).join(" · ");
    return `<span class="training-run-history-main"><strong>${escapeHtml(name)}</strong><span>${escapeHtml(metrics || item.job_id)}</span></span><span class="training-run-history-side"><span>${escapeHtml(trainingStatusLabel(item.status))}</span><span>${escapeHtml(formatDateTimeText(item.updated_at || item.created_at))}</span></span>`;
  }

  function finiteMetric(value) {
    return Number.isFinite(Number(value)) ? formatNumber(Number(value), 4) : "";
  }

  start.addEventListener("click", () => startTrainingRun().catch(showError));
  cancel.addEventListener("click", () => cancelTrainingRun().catch(showError));
  return { loadTrainingRuns, renderTrainingRunView };
}
