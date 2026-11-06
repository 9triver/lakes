import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Alert, Autocomplete, Box, Button, Checkbox, Chip, CircularProgress, Dialog, DialogContent, DialogTitle, FormControlLabel, IconButton, LinearProgress, MenuItem, Slider, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { ChevronLeft, ChevronRight, Download, Play, Square, X, ZoomIn, ZoomOut } from "lucide-react";
import { getJson } from "../../api/client";
import { LargePreviewMap } from "./LargePreviewMap";
import { useValidationModels } from "./api";
import type { ModelOption } from "./api";
import { workspaceRegionApi } from "../workspaces/api";
import { useRegions } from "../regions/api";
import { isRgbNirSlope } from "./modelSelection";
import { streamJson } from "./streamJson";
import { useComparisonSites, type ComparisonSite } from "./useComparisonSites";


interface Context {
  context_id: string; site_id: string; site_bounds_wgs84?: number[]; site_geometry?: object;
  imagery: { asset: string; name: string; date: string; width: number; height: number; crs: string; resolution: number[]; bounds_wgs84?: number[]; bounds_native?: number[] };
  assets: { key: string; name: string; date: string }[];
  valid_pixels: number;
  reference: { available: boolean; name: string; kind: string; description: string; scored_pixels?: number; water_pixels?: number; background_pixels?: number; ignored_pixels?: number };
  preview: string; reference_overlay?: string; reference_mask?: string; ignored_overlay?: string | null;
}
interface Preparation {
  id: string; status: "queued" | "running" | "ready" | "failed"; context?: Context;
  error?: string; warning?: string;
  progress: Record<string, { stage: string; message: string; completed?: number | null; total?: number | null }>;
}
const readinessLabel = (item: ComparisonSite) => item.cache_checked === false ? "缓存状态检查中（可先选择）" : item.seed_cached && item.slope_cached ? "坡度＋高置信种子已缓存" : item.seed_cached ? "高置信种子已缓存" : item.imagery_available ? item.slope_cached ? "坡度已缓存，种子待生成" : "待准备" : "无可用影像";
const readinessGroups = ["坡度＋高置信种子已缓存", "高置信种子已缓存", "其他区域"];
interface Result {
  context_id: string; cached: boolean;
  model: { key: string; epoch: number; architecture: string; input_bands: string[] };
  metrics: { iou: number | null; f1: number | null; precision: number | null; recall: number | null; tp: number; fp: number; fn: number; tn: number; scored_pixels: number } | null;
  stats: { valid_pixels: number; predicted_pixels: number; predicted_ratio: number; area_km2: number | null; seconds: number };
  prediction_overlay: string; error_overlay?: string;
}
interface Outcome { key: string; result?: Result; error?: string }
interface SavedEvent {
  type: "progress" | "context" | "result" | "complete" | "error";
  context_id?: string; item?: Outcome; stage?: string; message?: string;
  completed?: number; total?: number | null; loaded?: number;
}
interface SavedProgress { message: string; completed: number; total: number | null; loaded: number }
interface Props {
  workspaceId: string; scope: string; routeSiteId?: string; routeSiteRegion?: string;
  onRouteResult: (siteId: string, region: string, model: string) => void;
}
const percentage = (value?: number | null) => value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(2)}%`;
const number = (value?: number | null, digits = 2) => value == null || !Number.isFinite(value) ? "—" : value.toLocaleString("zh-CN", { maximumFractionDigits: digits });

type SortMetric = "validation_iou" | "site_precision" | "site_iou";
const sortLabels: Record<SortMetric, string> = { validation_iou: "原模型最高验证IoU", site_precision: "当前site Precision", site_iou: "当前site IoU" };
const highestValidationIou = (model?: ModelOption) => model?.max_validation_iou !== undefined ? model.max_validation_iou : model?.best_iou;
const excludedModel = (model: ModelOption) => [model.label, model.name, model.path].some((value) => /gan|fold/i.test(value || ""));

interface PreviewProps { context: Context; overlay?: string | null; label: string; base?: string; showIgnored?: boolean; overlayColor?: string }
interface LargePreview { title: string; context: Context; result?: Result; modelKey?: string }
type LargePreviewMode = "prediction" | "errors" | "reference" | "reference_overlay" | "original" | "osm";

const previewArrowSx = {
  width: { xs: 54, sm: 68 }, height: { xs: 54, sm: 68 },
  bgcolor: "rgba(15,23,42,.82)", color: "#fff", border: "2px solid rgba(255,255,255,.85)", boxShadow: 4,
  "&:hover": { bgcolor: "rgba(15,23,42,.98)" },
  "&.Mui-disabled": { bgcolor: "rgba(15,23,42,.3)", color: "rgba(255,255,255,.45)" },
} as const;

function Preview({ context, overlay, label, base, showIgnored, overlayColor, onOpen, large = false }: PreviewProps & { onOpen?: () => void; large?: boolean }) {
  return <Box role={onOpen ? "button" : undefined} tabIndex={onOpen ? 0 : undefined} aria-label={onOpen ? `放大查看${label}` : undefined} aria-haspopup={onOpen ? "dialog" : undefined} title={onOpen ? "点击查看大图" : undefined} onClick={onOpen} onKeyDown={onOpen ? (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onOpen(); } } : undefined} sx={{ position: "relative", bgcolor: "#181e26", width: "100%", height: large ? "100%" : undefined, aspectRatio: large ? undefined : `${context.imagery.width}/${context.imagery.height}`, maxHeight: large ? undefined : 560, overflow: "hidden", cursor: onOpen ? "zoom-in" : undefined, "&:focus-visible": { outline: "3px solid", outlineColor: "primary.main", outlineOffset: -3 } }}>
    <img src={base || context.preview} alt={base ? label : `${label}原始影像`} style={{ position: "absolute", width: "100%", height: "100%", objectFit: "contain" }} />
    {overlay && (overlayColor ? <Box role="img" aria-label={label} sx={{ position: "absolute", width: "100%", height: "100%", bgcolor: overlayColor, maskImage: `url("${overlay}")`, WebkitMaskImage: `url("${overlay}")`, maskSize: "contain", maskPosition: "center", maskRepeat: "no-repeat" }} /> : <img src={overlay} alt={label} style={{ position: "absolute", width: "100%", height: "100%", objectFit: "contain" }} />)}
    {showIgnored && context.ignored_overlay && <img src={context.ignored_overlay} alt="255忽略区覆盖层" style={{ position: "absolute", width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none" }} />}
  </Box>;
}

export function ModelComparisonView({ workspaceId, scope, routeSiteId = "", routeSiteRegion = "", onRouteResult }: Props) {
  const models = useValidationModels(workspaceId, "all", "current");
  const regions = useRegions();
  const [region, setRegion] = useState(routeSiteRegion || (scope === "all" ? "gansu" : scope));
  const sites = useComparisonSites(workspaceId, region);
  const options = sites.data?.items || [];
  const selectedSite = options.find((item) => item.site_id === routeSiteId) || null;
  const [asset, setAsset] = useState("");
  const [reference, setReference] = useState("spectral_osm_intersection");
  const [threshold, setThreshold] = useState(.5);
  const [thresholdDraft, setThresholdDraft] = useState(.5);
  const [thresholdEditing, setThresholdEditing] = useState(false);
  const thresholdCommitTimer = useRef<ReturnType<typeof window.setTimeout> | undefined>(undefined);
  const savedResultsRequest = useRef(0);
  const savedResultsAbort = useRef<AbortController | null>(null);
  const [savedProgress, setSavedProgress] = useState<SavedProgress | null>(null);
  const [filter, setFilter] = useState("");
  const [rgbNirSlopeOnly, setRgbNirSlopeOnly] = useState(true);
  const [completedOnly, setCompletedOnly] = useState(true);
  const [selection, setSelection] = useState<string[]>([]);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [running, setRunning] = useState(false);
  const [loadingSaved, setLoadingSaved] = useState(false);
  const [active, setActive] = useState("");
  const [mode, setMode] = useState("prediction");
  const [showIgnored, setShowIgnored] = useState(false);
  const [largePreview, setLargePreview] = useState<LargePreview | null>(null);
  const [previewZoom, setPreviewZoom] = useState(1);
  const [largePreviewMode, setLargePreviewMode] = useState<LargePreviewMode>("prediction");
  const [largeShowIgnored, setLargeShowIgnored] = useState(false);
  const [sortMetric, setSortMetric] = useState<SortMetric>("site_iou");
  const [sortDirection, setSortDirection] = useState("desc");
  const [globalError, setGlobalError] = useState("");
  const cancelled = useRef(false);
  const scopeIdentity = `${workspaceId}:${region}:${routeSiteId}:${asset}:${reference}:${threshold}`;
  const currentIdentity = useRef(scopeIdentity);
  currentIdentity.current = scopeIdentity;
  const [preparationRetry, setPreparationRetry] = useState(0);
  const preparation = useQuery({
    queryKey: ["comparison-preparation", workspaceId, region, routeSiteId, asset, reference, preparationRetry],
    queryFn: () => getJson<Preparation>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(routeSiteId)}/model-comparison/preparation?${new URLSearchParams({ asset, reference, retry: String(preparationRetry) })}`)),
    enabled: Boolean(workspaceId && region && routeSiteId), retry: false,
    refetchInterval: (query) => ["queued", "running"].includes(query.state.data?.status || "") ? 1000 : false,
    refetchOnWindowFocus: false,
  });
  const preparing = preparation.isLoading || ["queued", "running"].includes(preparation.data?.status || "");
  const context = { data: preparation.data?.context, isLoading: preparing, isFetching: preparing,
    isError: preparation.isError || preparation.data?.status === "failed",
    error: preparation.error || new Error(preparation.data?.error || "数据准备失败") };
  useEffect(() => { if (preparation.data?.status === "ready") void sites.refetch(); }, [preparation.data?.id, preparation.data?.status]);
  useEffect(() => {
    if (routeSiteRegion) setRegion(routeSiteRegion);
  }, [routeSiteRegion]);
  useEffect(() => {
    cancelled.current = true;
    savedResultsAbort.current?.abort();
    savedResultsRequest.current += 1;
    if (loadingSaved) { setRunning(false); setLoadingSaved(false); }
    setOutcomes([]); setGlobalError(""); setLargePreview(null);
  }, [scopeIdentity]);
  useEffect(() => () => {
    cancelled.current = true;
    window.clearTimeout(thresholdCommitTimer.current);
    savedResultsAbort.current?.abort();
  }, []);
  useEffect(() => { setSelection([]); }, [workspaceId]);
  const all = (models.data?.items || []).filter((item) => !excludedModel(item));
  const allowedKeys = new Set(all.map((item) => item.key));
  const visible = all.filter((item) => (!rgbNirSlopeOnly || isRgbNirSlope(item)) && (!completedOnly || item.completed !== false) && `${item.label} ${item.architecture_label}`.toLowerCase().includes(filter.toLowerCase()));
  const option = (key: string): ModelOption | undefined => all.find((item) => item.key === key);
  const sortValue = (item: Outcome) => sortMetric === "validation_iou" ? highestValidationIou(option(item.key)) : sortMetric === "site_precision" ? item.result?.metrics?.precision : item.result?.metrics?.iou;
  const orderedOutcomes = outcomes.filter((item) => allowedKeys.has(item.key)).sort((a, b) => {
    const av = sortValue(a); const bv = sortValue(b);
    const aValid = av != null && Number.isFinite(av); const bValid = bv != null && Number.isFinite(bv);
    if (aValid !== bValid) return aValid ? -1 : 1;
    if (aValid && bValid && av !== bv) return (sortDirection === "desc" ? -1 : 1) * (av! - bv!);
    return (option(a.key)?.label || a.key).localeCompare(option(b.key)?.label || b.key, "zh-CN") || a.key.localeCompare(b.key);
  });
  const successful = orderedOutcomes.filter((item) => item.result);
  const previewModelIndex = successful.findIndex((item) => item.key === largePreview?.modelKey);
  const canSwitchPreview = Boolean(largePreview) && successful.length > 0 && (previewModelIndex < 0 || successful.length > 1);
  const switchPreview = (direction: -1 | 1) => {
    if (!largePreview || !successful.length) return;
    const nextIndex = previewModelIndex < 0
      ? direction > 0 ? 0 : successful.length - 1
      : (previewModelIndex + direction + successful.length) % successful.length;
    const item = successful[nextIndex];
    setLargePreview({ title: option(item.key)?.label || item.key, context: largePreview.context, result: item.result, modelKey: item.key });
    if (largePreviewMode === "errors" && !item.result?.error_overlay) setLargePreviewMode("prediction");
  };
  const changeThreshold = (value: number) => {
    window.clearTimeout(thresholdCommitTimer.current);
    savedResultsRequest.current += 1;
    savedResultsAbort.current?.abort();
    setThresholdDraft(value); setThresholdEditing(true);
    if (loadingSaved) { setRunning(false); setLoadingSaved(false); }
  };
  const commitThreshold = (value: number) => {
    changeThreshold(value);
    thresholdCommitTimer.current = window.setTimeout(() => {
      setThreshold(value); setThresholdEditing(false);
    }, 650);
  };
  const loadSaved = async () => {
    if (!context.data || running || thresholdEditing) return;
    const identity = scopeIdentity;
    const fixedContext = context.data;
    const request = ++savedResultsRequest.current;
    const controller = new AbortController();
    savedResultsAbort.current?.abort(); savedResultsAbort.current = controller;
    setRunning(true); setLoadingSaved(true); setGlobalError(""); setOutcomes([]); setSelection([]);
    setSavedProgress({ message: "正在连接已保存结果", completed: 0, total: null, loaded: 0 });
    let loaded = 0; let complete = false; let contextConfirmed = false;
    try {
      const params = new URLSearchParams({ asset: fixedContext.imagery.asset, reference, threshold: String(threshold), stream: "1" });
      await streamJson<SavedEvent>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(routeSiteId)}/model-comparison/saved?${params}`), (event) => {
        if (currentIdentity.current !== identity || request !== savedResultsRequest.current) {
          controller.abort(); return;
        }
        if (event.type === "error") throw new Error(event.message || "读取已保存结果失败");
        if (event.context_id && event.context_id !== fixedContext.context_id) throw new Error("影像或标注已变化，请重新载入");
        if (event.type === "context") contextConfirmed = true;
        if (event.type === "result" && event.item && allowedKeys.has(event.item.key)) {
          if (!contextConfirmed || event.item.result?.context_id !== fixedContext.context_id) throw new Error("保存结果不属于当前影像，请重新载入");
          const item = event.item; loaded += 1;
          setOutcomes((current) => [...current.filter((entry) => entry.key !== item.key), item]);
          setSelection((current) => current.includes(item.key) ? current : [...current, item.key]);
        }
        setSavedProgress({ message: event.message || (event.type === "complete" ? "已保存结果加载完成" : "正在检查并读取已保存结果"),
          completed: event.completed ?? 0, total: event.total ?? null, loaded });
        if (event.type === "complete") {
          complete = true;
          if (!loaded) setGlobalError("当前影像、参考和阈值尚无符合筛选条件的已保存结果，请选择模型后运行。");
        }
      }, controller.signal);
      if (!complete && !controller.signal.aborted) throw new Error("结果传输中断，已收到的结果已保留，请重试");
    } catch (error) {
      if (!controller.signal.aborted && currentIdentity.current === identity && request === savedResultsRequest.current) setGlobalError(error instanceof Error ? error.message : String(error));
    } finally {
      if (savedResultsAbort.current === controller) savedResultsAbort.current = null;
      if (request === savedResultsRequest.current) { setRunning(false); setLoadingSaved(false); }
    }
  };
  const run = async () => {
    if (!context.data || running || thresholdEditing) return;
    const fixedContext = context.data;
    const identity = scopeIdentity;
    const selected = selection.filter((key) => allowedKeys.has(key) && !option(key)?.error);
    cancelled.current = false; setRunning(true); setOutcomes([]); setGlobalError("");
    try {
      for (const key of selected) {
        if (cancelled.current || currentIdentity.current !== identity) break;
        setActive(key);
        let outcome: Outcome;
        try {
          const params = new URLSearchParams({ model: key, asset: fixedContext.imagery.asset, reference, threshold: String(threshold) });
          const result = await getJson<Result>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(routeSiteId)}/model-comparison?${params}`));
          if (result.context_id !== fixedContext.context_id) throw new Error("影像或标注已变化，请重新载入后运行，不能混合比较");
          outcome = { key, result };
        } catch (error) { outcome = { key, error: error instanceof Error ? error.message : String(error) }; }
        if (currentIdentity.current === identity) setOutcomes((current) => [...current, outcome]);
      }
    } catch (error) { setGlobalError(String(error)); }
    finally { setActive(""); setRunning(false); }
  };
  const presetModels = all.filter((item) => item.completed && !item.error && item.path?.includes("guizhou_checkpoint_selection_lr0005_seed42_20261024"));
  const preset = (both: boolean) => setSelection(presetModels.filter((item) => both || item.weight === "best.pt").map((item) => item.key));
  const openPreview = (preview: LargePreview) => {
    setPreviewZoom(1);
    setLargePreviewMode(mode === "original" ? "original" : preview.result ? mode === "errors" ? "errors" : "prediction" : "reference");
    setLargeShowIgnored(showIgnored);
    setLargePreview(preview);
  };
  const exportCsv = () => {
    const rows: (string | number | null | undefined)[][] = [["model", "weight", "epoch", "IoU", "F1", "precision", "recall", "TP", "FP", "FN", "TN", "scored_pixels", "water_area_km2", "reference", "reference_kind", "ignored_pixels", "image", "threshold"]];
    for (const item of successful) {
      const r = item.result!; const m = r.metrics;
      rows.push([option(item.key)?.label || item.key, option(item.key)?.weight, r.model.epoch, m?.iou, m?.f1, m?.precision, m?.recall, m?.tp, m?.fp, m?.fn, m?.tn, m?.scored_pixels, r.stats.area_km2, context.data?.reference.name, context.data?.reference.kind, context.data?.reference.ignored_pixels, context.data?.imagery.name, threshold]);
    }
    const text = "\ufeff" + rows.map((row) => row.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = `${routeSiteId}-model-comparison.csv`; link.click(); URL.revokeObjectURL(url);
  };
  return <Box data-testid="model-comparison" sx={{ height: "100%", overflow: "auto", p: { xs: 1, md: 2 }, bgcolor: "#f5f7fa" }}>
    <Typography variant="h6" sx={{ mb: .5 }}>同一区域，多模型对比</Typography>
    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>固定影像、参考标签和阈值，逐个运行所选权重。原始网格计分，重叠窗口融合后每个像素只计一次。</Typography>
    <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1.5 }}>
      <TextField select label="区域" size="small" value={region} disabled={running} sx={{ minWidth: 120 }} onChange={(event) => { setRegion(event.target.value); setAsset(""); onRouteResult("", event.target.value, ""); }}>
        {(regions.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key}>{item.name}</MenuItem>)}
      </TextField>
      <Autocomplete sx={{ minWidth: 250, flex: 1 }} disabled={running} options={options} value={selectedSite} loading={sites.isFetching && options.length === 0}
        getOptionLabel={(item) => `${item.display_name || item.site_id} · ${item.site_id} · ${readinessLabel(item)}`}
        getOptionKey={(item) => item.site_id}
        groupBy={(item) => readinessGroups[item.rank] || readinessGroups[2]}
        isOptionEqualToValue={(a, b) => a.site_id === b.site_id}
        noOptionsText={sites.isFetching ? "正在读取区域目录…" : "没有匹配区域"}
        onChange={(_, value) => { setAsset(""); onRouteResult(value?.site_id || "", region, ""); }}
        renderInput={(params) => <TextField {...params} size="small" label="选择观测区域" helperText={sites.isFetching ? `已显示 ${options.length} 个区域，可先选择；缓存已检查 ${sites.data?.checked || 0}/${sites.data?.total ?? "待统计"}` : `全部区域可选（${options.length}个）；按坡度＋种子、种子、其他排序`} />} />
      {context.data && <TextField select label="共同影像" size="small" disabled={running} value={asset || context.data.imagery.asset} onChange={(event) => setAsset(event.target.value)} sx={{ minWidth: 230 }}>
        {context.data.assets.map((item) => <MenuItem key={item.key} value={item.key}>{item.name}</MenuItem>)}
      </TextField>}
      <TextField select label="参考标签" size="small" value={reference} disabled={running} onChange={(event) => setReference(event.target.value)} sx={{ minWidth: 190 }}>
        <MenuItem value="spectral_osm_intersection">光谱 ∩ OSM（高置信种子）</MenuItem><MenuItem value="local">同日期本地水体标注（未核验）</MenuItem><MenuItem value="none">不使用参考，仅预测</MenuItem>
      </TextField>
      <Box sx={{ minWidth: 170, px: 1 }}><Typography variant="caption">共同阈值 {thresholdDraft.toFixed(2)}{thresholdEditing ? " · 调整中" : ""}</Typography><Slider aria-label="共同阈值" disabled={running && !loadingSaved} value={thresholdDraft} min={.05} max={.95} step={.05} onChange={(_, value) => changeThreshold(value as number)} onChangeCommitted={(_, value) => commitThreshold(value as number)} /></Box>
    </Box>
    {sites.isFetching && <Box role="status" aria-live="polite" sx={{ mb: 1.5 }}>
      <Typography variant="caption" color="text.secondary">{sites.data?.total == null ? "正在读取区域目录…" : `区域列表已可选择；正在检查缓存 ${sites.data.checked}/${sites.data.total}`}</Typography>
      <LinearProgress sx={{ mt: .5 }} variant={sites.data?.total == null ? "indeterminate" : "determinate"}
        value={sites.data?.total ? Math.min(100, sites.data.checked / sites.data.total * 100) : 100} />
    </Box>}
    {sites.isError && <Alert severity="error">区域缓存检查中断，已显示区域仍可选择：{sites.error.message}<Button size="small" onClick={() => void sites.refetch()}>重试检查</Button></Alert>}
    {routeSiteId && (preparing || context.isError || preparation.data?.warning) && <Box sx={{ p: 1.5, mb: 1, border: 1, borderColor: "divider", bgcolor: "background.paper" }}>
      <Typography variant="subtitle2">区域数据准备{preparation.data?.status === "queued" ? "：等待处理" : ""}</Typography>
      {Object.entries(preparation.data?.progress || {}).map(([key, event]) => <Box key={key} sx={{ my: 1 }}>
        <Typography variant="body2">{key === "seed" ? "光谱 ∩ OSM 高置信种子" : key === "terrain" ? "坡度数据" : "共同预览"}：{event.message}{event.total != null && event.completed != null && event.total > 1 ? `（${event.completed}/${event.total}）` : ""}</Typography>
        <LinearProgress variant={event.total && event.completed != null || !preparing ? "determinate" : "indeterminate"} value={event.total && event.completed != null ? Math.min(100, event.completed / event.total * 100) : undefined} />
      </Box>)}
      {!preparation.data && preparing && <LinearProgress />}
      {!preparing && <Button size="small" onClick={() => setPreparationRetry((value) => value + 1)}>重试数据准备</Button>}
    </Box>}
    {preparation.data?.warning && <Alert severity="warning" sx={{ mb: 1 }}>{preparation.data.warning}</Alert>}
    {context.isError && <Alert severity="error">{context.error.message}</Alert>}
    {context.data && <Alert severity={context.data.reference.available ? "info" : "warning"} sx={{ mb: 1.5 }}>
      {context.data.imagery.name} · {context.data.imagery.width}×{context.data.imagery.height} · {context.data.imagery.resolution.map((v) => number(v)).join("×")} 米 · {number(context.data.valid_pixels, 0)} 有效像素。{context.data.reference.description}
      {context.data.reference.scored_pixels != null && <Typography variant="body2">共同评分像素 {number(context.data.reference.scored_pixels, 0)}（有效像素的 {percentage(context.data.reference.scored_pixels / context.data.valid_pixels)}）：水体 {number(context.data.reference.water_pixels, 0)}，背景 {number(context.data.reference.background_pixels, 0)}；忽略 {number(context.data.reference.ignored_pixels, 0)}。</Typography>}
    </Alert>}
    <Box sx={{ bgcolor: "background.paper", border: 1, borderColor: "divider", borderRadius: 1, p: 1.5, mb: 2 }}>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap", mb: 1 }}>
        <TextField size="small" label="筛选模型 / A、B组 / 编码器 / 学习率" value={filter} onChange={(event) => setFilter(event.target.value)} sx={{ flex: 1, minWidth: 250 }} />
        <FormControlLabel control={<Checkbox checked={rgbNirSlopeOnly} onChange={(_, checked) => setRgbNirSlopeOnly(checked)} />} label="仅 RGB-NIR-SLOPE" />
        <FormControlLabel control={<Checkbox checked={completedOnly} onChange={(_, checked) => setCompletedOnly(checked)} />} label="仅完整训练" />
        <Chip size="small" label={`${visible.length} 个权重 · 已选 ${selection.length}`} />
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>已排除名称或路径包含 gan、fold 的模型（不区分大小写）。</Typography>
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1 }}>
        {presetModels.some((item) => item.weight === "best.pt") && <Button size="small" disabled={running || models.isLoading} onClick={() => preset(false)}>本轮{presetModels.filter((item) => item.weight === "best.pt").length}个模型</Button>}
        {presetModels.some((item) => item.weight === "best_loss.pt") && <Button size="small" disabled={running || models.isLoading} onClick={() => preset(true)}>本轮两种权重，共{presetModels.length}个</Button>}
        <Button size="small" disabled={running || models.isLoading} onClick={() => setSelection(visible.filter((item) => !item.error).map((item) => item.key))}>全选</Button>
        <Button size="small" disabled={running} onClick={() => setSelection([])}>清空选择</Button>
      </Box>
      {models.isLoading && <Box><Typography variant="body2" color="text.secondary">正在扫描模型目录并读取训练信息…</Typography><LinearProgress /></Box>}
      {models.isError && <Alert severity="error">{models.error.message}</Alert>}
      <Box sx={{ maxHeight: 265, overflow: "auto" }}>
        {visible.map((item) => <Box key={item.key} sx={{ display: "flex", alignItems: "center", borderBottom: "1px solid #edf0f3", py: .25 }}>
          <Checkbox size="small" disabled={running || Boolean(item.error)} checked={selection.includes(item.key)} slotProps={{ input: { "aria-label": item.label } }} onChange={(_, checked) => setSelection((current) => checked ? [...current, item.key] : current.filter((key) => key !== item.key))} />
          <Box sx={{ minWidth: 0, flex: 1 }}><Typography variant="body2" sx={{ overflowWrap: "anywhere" }}>{item.label}</Typography><Typography variant="caption" color={item.error ? "error" : "text.secondary"}>{item.error || `${item.architecture_label || "U-Net"} · ${item.in_channels} 通道 · 轮次 ${item.epoch} · 选中轮次验证IoU ${percentage(item.selected_validation_iou ?? item.best_iou)}`}</Typography></Box>
          {item.completed === false && <Chip size="small" color="warning" label="训练未完成" />}
        </Box>)}
      </Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
        <Button variant="contained" startIcon={running ? <CircularProgress size={15} color="inherit" /> : <Play size={16} />} disabled={running || thresholdEditing || !selection.length || !context.data || context.isFetching} onClick={run}>{running ? loadingSaved ? "加载已保存结果…" : `运行中 ${outcomes.length}/${selection.length}` : "运行所选模型"}</Button>
        {running && loadingSaved && <Button startIcon={<Square size={15} />} onClick={() => savedResultsAbort.current?.abort()}>停止加载（保留已加载结果）</Button>}
        {running && !loadingSaved && <Button startIcon={<Square size={15} />} onClick={() => { cancelled.current = true; }}>停止后续模型</Button>}
        <Button startIcon={<Download size={15} />} disabled={!successful.length || running || thresholdEditing} onClick={exportCsv}>导出指标 CSV</Button>
        <Button disabled={running || thresholdEditing || !context.data || context.isFetching || models.isLoading} onClick={loadSaved}>查看已保存结果</Button>
        <Typography variant="caption" color="text.secondary">{running ? loadingSaved ? `已加载 ${savedProgress?.loaded || 0} 个结果` : option(active)?.label : outcomes.length ? `已返回 ${outcomes.length} 个结果，成功 ${successful.length} 个` : "旧实验结果不会被覆盖；未完成权重不会作为完整实验展示。"}</Typography>
      </Box>
      {running && !loadingSaved && <LinearProgress variant="determinate" value={100 * outcomes.length / Math.max(selection.length, 1)} sx={{ mt: 1 }} />}
    </Box>
    {loadingSaved && savedProgress && <Box role="status" aria-live="polite" sx={{ mb: 2, p: 1.5, border: 1, borderColor: "divider", bgcolor: "background.paper" }}>
      <Typography variant="body2">{savedProgress.message} · 已加载 {savedProgress.loaded} 个模型结果{savedProgress.total != null ? ` · 已检查 ${savedProgress.completed}/${savedProgress.total} 个权重` : ""}</Typography>
      <LinearProgress sx={{ mt: 1 }} variant={savedProgress.total != null ? "determinate" : "indeterminate"}
        value={savedProgress.total ? Math.min(100, savedProgress.completed / savedProgress.total * 100) : 100} />
    </Box>}
    {globalError && <Alert severity="error">{globalError}</Alert>}
    {!!outcomes.length && <Typography variant="caption" color="text.secondary">指标表与视觉对照按{sortLabels[sortMetric]}{sortDirection === "desc" ? "从高到低" : "从低到高"}排列；缺失排序指标的结果置后。</Typography>}
    {!!outcomes.length && <TableContainer sx={{ bgcolor: "background.paper", mb: 2, maxHeight: 430 }}><Table size="small" stickyHeader aria-label="模型比较指标"><TableHead><TableRow>
      {["模型 / 权重", "轮次", "原模型最高验证IoU", "IoU", "F1", "Precision", "Recall", "误报 FP", "漏检 FN", "评分像素", "全图预测面积 km²"].map((label) => <TableCell key={label} sx={{ whiteSpace: "nowrap" }}>{label}</TableCell>)}
    </TableRow></TableHead><TableBody>{orderedOutcomes.map((item) => <TableRow key={item.key} data-model-key={item.key}><TableCell sx={{ minWidth: 280, maxWidth: 420, overflowWrap: "anywhere" }}>{option(item.key)?.label || item.key}{item.error && <Typography variant="caption" color="error" display="block">{item.error}</Typography>}</TableCell><TableCell>{item.result?.model.epoch ?? "—"}</TableCell>
      {[highestValidationIou(option(item.key)), item.result?.metrics?.iou, item.result?.metrics?.f1, item.result?.metrics?.precision, item.result?.metrics?.recall].map((value, i) => <TableCell key={i}>{percentage(value)}</TableCell>)}
      <TableCell>{number(item.result?.metrics?.fp, 0)}</TableCell><TableCell>{number(item.result?.metrics?.fn, 0)}</TableCell><TableCell>{number(item.result?.metrics?.scored_pixels, 0)}</TableCell><TableCell>{number(item.result?.stats.area_km2)}</TableCell>
    </TableRow>)}</TableBody></Table></TableContainer>}
    {context.data && <>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap", mb: 1 }}>
        <Typography variant="subtitle1">同范围视觉对照</Typography>
        <ToggleButtonGroup size="small" exclusive value={mode} onChange={(_, value) => { if (value) setMode(value); }}>
          <ToggleButton value="prediction">预测叠加</ToggleButton><ToggleButton value="errors" disabled={!context.data.reference.available}>误报 / 漏检</ToggleButton><ToggleButton value="original">原图</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButton size="small" value="ignored" selected={showIgnored} disabled={!context.data.ignored_overlay} aria-label="显示255忽略区" onChange={() => setShowIgnored((value) => !value)}>显示255忽略区</ToggleButton>
        <TextField select size="small" label="对照排序" value={sortMetric} sx={{ minWidth: 205 }} onChange={(event) => setSortMetric(event.target.value as SortMetric)}>
          {(Object.entries(sortLabels) as [SortMetric, string][]).map(([key, label]) => <MenuItem key={key} value={key}>{label}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="排序方向" value={sortDirection} sx={{ minWidth: 120 }} onChange={(event) => setSortDirection(event.target.value)}>
          <MenuItem value="desc">从高到低</MenuItem><MenuItem value="asc">从低到高</MenuItem>
        </TextField>
        <Typography variant="caption" color="text.secondary">参考 mask：绿色为水体，深色为背景，灰色为忽略区；预测叠加：亮青色为预测水体；误报 / 漏检：蓝色为命中，粉色为误报，橙色为漏检。忽略区不计分，预览缩放不影响原始网格计分。</Typography>
      </Box>
      {showIgnored && context.data.ignored_overlay && <Typography variant="body2" sx={{ mb: 1, color: "#7e22ce" }}>紫色覆盖：标签255忽略区，共 {number(context.data.reference.ignored_pixels, 0)} 像素，不参与 IoU / Precision 评分；无效影像区域不着色。</Typography>}
      {sortMetric === "validation_iou" && <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>原模型最高验证IoU取该次训练完整历史的最高值；最低验证损失权重所选轮次的验证IoU可能与之不同。</Typography>}
      <Box data-testid="comparison-previews" sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(min(100%,300px),1fr))", gap: 1.5 }}>
        <Box sx={{ bgcolor: "background.paper", border: 1, borderColor: "divider" }}><Typography sx={{ p: 1 }} variant="body2">{context.data.reference.kind === "spectral_osm_intersection" ? "光谱 ∩ OSM（高置信种子）" : "共同影像与参考标注"}</Typography><Preview context={context.data} showIgnored={showIgnored} base={mode === "original" ? undefined : context.data.reference_mask} label={context.data.reference.kind === "spectral_osm_intersection" ? "光谱 ∩ OSM（高置信种子）mask" : "参考标注mask"} onOpen={() => openPreview({ title: context.data!.reference.kind === "spectral_osm_intersection" ? "光谱 ∩ OSM（高置信种子）" : "共同影像与参考标注", context: context.data! })} /></Box>
        {successful.map((item) => <Box key={item.key} data-testid="comparison-model-card" data-model-key={item.key} sx={{ bgcolor: "background.paper", border: 1, borderColor: "divider" }}>
          <Typography sx={{ p: 1, minHeight: 75, overflowWrap: "anywhere" }} variant="body2">{option(item.key)?.label}<Typography component="span" variant="caption" display="block" color="text.secondary">原模型最高验证IoU {percentage(highestValidationIou(option(item.key)))} · Site Precision {percentage(item.result?.metrics?.precision)} · Site IoU {percentage(item.result?.metrics?.iou)} · 全图预测面积 {number(item.result?.stats.area_km2)} km² · {item.result?.cached ? "缓存" : `${number(item.result?.stats.seconds, 1)} 秒`}</Typography></Typography>
          <Preview context={context.data!} showIgnored={showIgnored} overlayColor={mode === "prediction" ? "#00ffff" : undefined} overlay={mode === "original" ? undefined : mode === "errors" ? item.result?.error_overlay : item.result?.prediction_overlay} label={mode === "errors" ? "误报漏检" : "模型预测"} onOpen={() => openPreview({ title: option(item.key)?.label || item.key, context: context.data!, result: item.result, modelKey: item.key })} />
        </Box>)}
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: "block" }}>该页面用于指定区域对照；部分历史模型可能训练过该区域，不能自动当作独立测试成绩。无参考标签时不显示准确率指标。</Typography>
    </>}
    <Dialog open={Boolean(largePreview)} onClose={() => setLargePreview(null)} onKeyDown={(event) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target;
      if (target instanceof HTMLElement && target.closest('input, textarea, select, [role="slider"], [role="combobox"], [role="listbox"], [role="tablist"], [contenteditable="true"]')) return;
      if (canSwitchPreview && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        event.preventDefault(); event.stopPropagation(); switchPreview(event.key === "ArrowRight" ? 1 : -1);
      }
    }} fullWidth maxWidth="xl" aria-labelledby="comparison-preview-title" slotProps={{ paper: { sx: { height: "90dvh", maxHeight: "90dvh" } } }}>
      <DialogTitle id="comparison-preview-header" component="div" sx={{ display: "flex", alignItems: "center", gap: 1, py: 1, px: 2, flexShrink: 0 }}>
        <Typography id="comparison-preview-title" component="h2" variant="subtitle1" sx={{ flex: 1, minWidth: 0, overflowWrap: "anywhere" }}>{largePreview?.title}</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: "nowrap" }} aria-live="polite">{previewModelIndex >= 0 ? `${previewModelIndex + 1} / ${successful.length}` : `参考图 · ${successful.length} 个模型`}</Typography>
        <IconButton aria-label="关闭大图" onClick={() => setLargePreview(null)}><X size={22} /></IconButton>
      </DialogTitle>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 2, pb: 1, flexWrap: "wrap", flexShrink: 0 }}>
        <ToggleButtonGroup size="small" exclusive value={largePreviewMode} aria-label="大图显示图层" onChange={(_, value: LargePreviewMode | null) => { if (value) setLargePreviewMode(value); }}>
          {largePreview?.result && <ToggleButton value="prediction">预测叠加</ToggleButton>}
          <ToggleButton value="original">原图</ToggleButton>
          <ToggleButton value="osm">OSM 地图</ToggleButton>
          {largePreview?.result && <ToggleButton value="errors" disabled={!largePreview.result.error_overlay}>误报 / 漏检</ToggleButton>}
          <ToggleButton value="reference" disabled={!largePreview?.context.reference_mask}>参考标注</ToggleButton>
          <ToggleButton value="reference_overlay" disabled={!largePreview?.context.reference_overlay}>标注叠加</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButton size="small" value="ignored" selected={largeShowIgnored} disabled={largePreviewMode === "osm" || !largePreview?.context.ignored_overlay} aria-label="显示255忽略区" onChange={() => setLargeShowIgnored((value) => !value)}>显示255忽略区</ToggleButton>
        <IconButton size="small" aria-label="缩小大图" disabled={previewZoom <= 1} onClick={() => setPreviewZoom((value) => Math.max(1, value - .25))}><ZoomOut size={20} /></IconButton>
        <Typography variant="body2">{Math.round(previewZoom * 100)}%</Typography>
        <IconButton size="small" aria-label="放大大图" disabled={previewZoom >= 4} onClick={() => setPreviewZoom((value) => Math.min(4, value + .25))}><ZoomIn size={20} /></IconButton>
        <Button size="small" onClick={() => setPreviewZoom(1)}>适应窗口</Button>
        <Typography variant="caption" color="text.secondary">滚轮缩放、拖动平移；← / → 按当前排序循环切换模型；Esc 关闭</Typography>
      </Box>
      <DialogContent sx={{ p: 0, flex: "1 1 0", minHeight: 0, overflow: "hidden", position: "relative", bgcolor: "#181e26" }}>
        {largePreview && <LargePreviewMap width={largePreview.context.imagery.width} height={largePreview.context.imagery.height}
          navigation={successful.length > 0 ? <>
            <IconButton aria-label="上一个模型预测" title="上一个模型预测（←）" disabled={!canSwitchPreview} onClick={() => switchPreview(-1)} sx={previewArrowSx}><ChevronLeft size={42} strokeWidth={2.5} /></IconButton>
            <IconButton aria-label="下一个模型预测" title="下一个模型预测（→）" disabled={!canSwitchPreview} onClick={() => switchPreview(1)} sx={previewArrowSx}><ChevronRight size={42} strokeWidth={2.5} /></IconButton>
          </> : undefined}
          imageTileUrl={workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(largePreview.context.site_id)}/tiles/{z}/{x}/{y}.png?${new URLSearchParams({ asset: largePreview.context.imagery.asset })}`)}
          previewBase={largePreviewMode === "reference"}
          base={largePreviewMode === "reference" ? largePreview.context.reference_mask || largePreview.context.preview : largePreview.context.preview}
          overlayKey={`${largePreview.modelKey || "reference"}:${largePreviewMode}`}
          overlay={largePreviewMode === "prediction" ? largePreview.result?.prediction_overlay : largePreviewMode === "errors" ? largePreview.result?.error_overlay : largePreviewMode === "reference_overlay" ? largePreview.context.reference_overlay : undefined}
          overlayColor={largePreviewMode === "prediction" ? "#00ffff" : undefined}
          ignored={largeShowIgnored ? largePreview.context.ignored_overlay : undefined}
          crs={largePreview.context.imagery.crs} nativeBounds={largePreview.context.imagery.bounds_native} siteBounds={largePreview.context.site_bounds_wgs84} siteGeometry={largePreview.context.site_geometry}
          osm={largePreviewMode === "osm"} bounds={largePreview.context.imagery.bounds_wgs84} zoom={previewZoom} onZoom={setPreviewZoom} />}
      </DialogContent>
    </Dialog>
  </Box>;
}
