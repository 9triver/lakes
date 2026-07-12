import { fetchJson, postJson } from "./api.js";
import { escapeHtml, formatCloud, formatCoverage, formatPercent, statusLabel } from "./formatters.js";

export function createSentinelDownloadController({ state, elements, activeApiPath, apiPathFor, activeRegionKey, loadSentinelTiles, loadImageryOptions, renderMeta, showError }) {
  const { tile, startDate, endDate, cloud, query, products } = elements;

  async function querySentinelProducts() {
    if (!tile.value) return;
    query.disabled = true;
    products.textContent = "查询中";
    const params = new URLSearchParams({
      tile: tile.value,
      lake_id: state.activeId,
      start: startDate.value,
      end: endDate.value,
      cloud: cloud.value,
      product_type: "MSIL1C",
      limit: "50",
    });
    try {
      const payload = await fetchJson(activeApiPath(`/sentinel/products?${params.toString()}`));
      renderSentinelProducts(payload.products || []);
    } finally {
      query.disabled = false;
    }
  }

  function renderSentinelProducts(items) {
    products.replaceChildren();
    if (!items.length) {
      products.textContent = "没有符合条件的产品";
      return;
    }
    for (const product of items) {
      const row = document.createElement("div");
      row.className = "sentinel-product";
      const action = document.createElement("button");
      action.type = "button";
      action.textContent = product.downloaded ? "已下载" : "下载";
      action.disabled = Boolean(product.downloaded);
      action.addEventListener("click", () => startSentinelDownload(product, action).catch(showError));
      row.innerHTML = `<span>${escapeHtml(product.date || "")}</span><span>${escapeHtml(product.tile || "")}</span><span>云量 ${formatCloud(product.cloud_cover)}</span><span>覆盖水体 ${formatPercent(product.lake_coverage_ratio)}</span><span>覆盖视图 ${formatPercent(product.aoi_coverage_ratio)}</span><span>非0 ${formatCoverage(product.coverage_ratio, product.coverage_basis)}</span><span title="${escapeHtml(product.name || "")}">${escapeHtml(product.name || "")}</span>`;
      row.append(action);
      products.append(row);
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

  query.addEventListener("click", () => querySentinelProducts().catch(showError));
  return { querySentinelProducts, renderSentinelProducts };
}
