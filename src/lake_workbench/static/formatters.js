export function formatNumber(value, digits) {
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

export function formatMetaList(value) {
  if (Array.isArray(value)) return value.join(", ");
  return String(value || "");
}

export function formatProductList(value) {
  const products = Array.isArray(value) ? value : String(value || "").split(",");
  return products
    .filter(Boolean)
    .map((item) => {
      const tile = item.match(/_T([0-9A-Z]{5})_/)?.[1] || "";
      const date = item.match(/MSIL\d[AC]?_(\d{8})T/)?.[1] || "";
      const compactDate = item.match(/MSIL\d[AC]?_(\d{8})(?:_|$)/)?.[1] || "";
      const fallback = item
        .replace(/^shaanxi_\d+_/, "")
        .replace(/^gansu_\d+_/, "")
        .replace(/^yunnan_\d+_/, "");
      return [tile, date || compactDate].filter(Boolean).join("/") || fallback || item;
    })
    .filter(Boolean)
    .join(", ");
}

export function formatImageryAssetType(value) {
  const text = String(value || "").trim();
  if (text === "lake_native" || text === "local_img") return "本体影像";
  if (text === "preloaded_tile" || text === "preloaded") return "预置 Sentinel tile";
  if (text === "sentinel_tile" || text === "user_download") return "已下载 Sentinel tile";
  return text || "影像";
}

export function formatImageryAssetLabels(value) {
  const labels = Array.isArray(value) ? value : String(value || "").split(",");
  const unique = [];
  for (const item of labels) {
    const label = formatImageryAssetType(item);
    if (label && !unique.includes(label)) unique.push(label);
  }
  return unique.join(" + ") || "影像";
}

export function formatCloud(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${formatNumber(number, 1)}%` : "未知";
}

export function formatCoverage(value, basis) {
  const number = Number(value);
  if (!Number.isFinite(number) || basis !== "pixels") return "需下载后统计";
  return `${formatNumber(number * 100, 1)}%`;
}

export function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${formatNumber(number * 100, 1)}%` : "未知";
}

export function formatPercentText(value) {
  const number = Number(value);
  return `${formatNumber(Number.isFinite(number) ? number * 100 : 0, 1)}%`;
}

export function formatDateText(value) {
  const text = String(value || "").trim();
  if (/^\d{8}$/.test(text)) return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
  return text || "未知";
}

export function formatDateTimeText(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  const match = text.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  return match ? `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}` : text;
}

export function formatDateInput(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function formatLabelScope(value) {
  if (value === "current_view") return "当前视图";
  if (value === "aoi_water") return "AOI 水体";
  return "当前水体";
}

export function formatMaskPolicy(value) {
  if (value === "current_view") return "当前视图";
  if (value === "visible_water_positive") return "可见水体为正";
  return "其他水体忽略";
}

export function formatTrainingLabelSource(source, threshold) {
  if (source === "current_view") return "当前视图";
  if (source === "jrc" && threshold) return `jrc ${threshold}`;
  return source || "";
}

export function typeLabel(value) {
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

export function nearestThreshold(value, thresholds) {
  return thresholds.reduce((best, item) => {
    return Math.abs(Number(item) - value) < Math.abs(Number(best) - value) ? Number(item) : Number(best);
  }, Number(thresholds[0]));
}

export function statusLabel(status) {
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

export function trainingStatusLabel(status) {
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

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
