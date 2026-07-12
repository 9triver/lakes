import { escapeHtml, formatDateTimeText, formatNumber } from "./formatters.js";

export function sortModelOptions(items) {
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

export function formatModelOptionText(item) {
  const parts = [item.label];
  const score = Number(item.best_iou);
  if (Number.isFinite(score)) parts.push(`best IoU ${formatNumber(score, 4)}`);
  if (item.epoch) parts.push(`epoch ${item.epoch}`);
  if (item.in_channels) parts.push(`${item.in_channels} band`);
  return parts.filter(Boolean).join(" · ");
}

export function renderModelDetail(container, model, regionName) {
  container.replaceChildren();
  if (!model) {
    container.textContent = "未选择模型";
    return;
  }
  const title = document.createElement("div");
  title.className = "model-detail-title";
  title.innerHTML = `<strong>${escapeHtml(model.name || model.label || model.key)}</strong><span>${escapeHtml(model.weight || "")}</span>`;
  container.append(title);
  if (model.error) {
    const error = document.createElement("div");
    error.className = "model-detail-error";
    error.textContent = model.error;
    container.append(error);
    return;
  }
  const latest = model.latest || {};
  const train = latest.train || {};
  const val = latest.val || {};
  const dataset = model.dataset || {};
  const config = model.config || {};
  const fields = [
    ["范围", model.scope === "all" ? "全部区域" : regionName(model.scope || model.region)],
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
  container.append(grid);
}
