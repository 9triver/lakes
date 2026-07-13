import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, Box, Button, CircularProgress, FormControl, IconButton, InputLabel, LinearProgress, MenuItem, Select, Slider, Typography } from "@mui/material";
import { ArrowLeft, Dices } from "lucide-react";
import { useContextWater, useEsaLayer, useJrcLayer, useSite, useLocalLabel, useLocalLabels, useTileMeta } from "../sites/api";
import { ImageryPanel } from "../imagery/ImageryPanel";
import { useSentinelTiles } from "../imagery/api";
import { SiteMap, type SiteMapHandle } from "../map/SiteMap";
import { TrainingCapture } from "../training-samples/TrainingCapture";
import { type ModelOption, useLakeModelPrediction, useRandomModelValidation, useValidationModels } from "./api";

function metric(value?: number, digits = 4) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-"; }

function ModelDetail({ model }: { model?: ModelOption }) {
  if (!model) return null;
  const config = model.config || {};
  const fields = [
    ["范围", model.scope === "all" ? "全部区域" : model.scope || model.region || "-"],
    ["Best IoU", metric(model.best_iou)], ["Best epoch", model.best_epoch || "-"],
    ["最新 val IoU", metric(model.latest?.val?.iou)], ["最新 train IoU", metric(model.latest?.train?.iou)],
    ["输入", `${model.in_channels || "-"} 波段 · 宽度 ${model.base_channels || "-"}`],
    ["训练 / 验证", config.train_count != null ? `${config.train_count} / ${config.val_count || 0}` : "-"],
    ["权重", model.weight || "-"], ["路径", model.path || "-"],
  ];
  return <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", border: 1, borderColor: "divider", mt: 1 }}>
    {fields.map(([label, value]) => <Box key={label} sx={{ px: 1.25, py: .75, minWidth: 0, borderRight: 1, borderBottom: 1, borderColor: "divider" }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" noWrap title={String(value)}>{value}</Typography></Box>)}
  </Box>;
}

function ValidationWorkspace({ result, threshold }: { result: NonNullable<ReturnType<typeof useRandomModelValidation>["data"]>; threshold: number }) {
  const region = result.site?.region || result.lake?.region || result.region;
  const lakeId = result.site_id || result.lake_id;
  const mapRef = useRef<SiteMapHandle>(null);
  const [jrcThreshold, setJrcThreshold] = useState(75);
  const [selectedLocalLabel, setSelectedLocalLabel] = useState("");
  const [imagerySelection, setImagerySelection] = useState({ tile: result.imagery?.tile || result.imagery?.tiles?.[0] || "", product: result.imagery?.product || result.imagery?.product_name || result.imagery?.products?.[0] || "" });
  const lake = useSite(region, lakeId);
  const tileMeta = useTileMeta(region, lakeId);
  const sentinelTiles = useSentinelTiles(region, lakeId);
  const context = useContextWater(region, lakeId);
  const esa = useEsaLayer(region, lakeId);
  const jrc = useJrcLayer(region, lakeId, jrcThreshold);
  const localLabels = useLocalLabels(region, lakeId);
  const localLabel = useLocalLabel(region, lakeId, selectedLocalLabel);
  useEffect(() => { const items = localLabels.data || []; if (!items.some((item) => item.id === selectedLocalLabel)) setSelectedLocalLabel(items[0]?.id || ""); }, [localLabels.data, selectedLocalLabel]);
  const handleImagery = useCallback((selection: { tile: string; product: string }) => setImagerySelection(selection), []);
  if (lake.isLoading) return <Box sx={{ display: "grid", placeItems: "center", minHeight: 360 }}><CircularProgress size={28} /></Box>;
  if (!lake.data) return <Alert severity="error">无法加载推理区域</Alert>;
  return <Box sx={{ display: "grid", gridTemplateRows: "minmax(420px,1fr) auto", minHeight: 0 }}>
    <SiteMap ref={mapRef} lake={lake.data} tileMeta={tileMeta.data} sentinelTiles={sentinelTiles.data} contextOsm={context.data?.sources.osm} contextHydro={context.data?.sources.hydrolakes} esa={esa.data} jrc={jrc.data} localLabel={localLabel.data} localLabels={localLabels.data || []} selectedLocalLabel={selectedLocalLabel} onLocalLabelChange={setSelectedLocalLabel} jrcThreshold={jrcThreshold} onJrcThresholdChange={setJrcThreshold} modelPrediction={result.prediction} />
    <Box sx={{ maxHeight: "38vh", overflow: "auto" }}>
      <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider" }}><Typography variant="body2">{lake.data.display_name || lake.data.name || lakeId} · 模型 {result.model.name} · 阈值 {threshold.toFixed(2)} · 水体像元 {metric(Number(result.stats.predicted_ratio || 0) * 100, 1)}% · {result.model.device || ""}</Typography></Box>
      <ImageryPanel region={region} lakeId={lakeId} onSelectionChange={handleImagery} />
      <TrainingCapture region={region} lakeId={lakeId} jrcThreshold={jrcThreshold} localLabel={(localLabels.data || []).find((item) => item.id === selectedLocalLabel)} imagery={imagerySelection} mapHandle={mapRef} modelValidation={{ model_key: result.model.key, model_name: result.model.name, model_path: result.model.path || "", threshold: result.model.threshold ?? threshold, predicted_area_km2: Number(result.stats.area_km2 || 0), predicted_ratio: Number(result.stats.predicted_ratio || 0), prediction_feature_count: result.prediction.features.length, device: result.model.device || "" }} />
    </Box>
  </Box>;
}

interface ModelValidationViewProps {
  scope: string;
  routeLakeId?: string;
  routeLakeRegion?: string;
  routeModel?: string;
  onBack: () => void;
  onRouteResult: (lakeId: string, lakeRegion: string, model: string) => void;
  onRouteModel: (model: string) => void;
}

export function ModelValidationView({ scope, routeLakeId = "", routeLakeRegion = "", routeModel = "", onBack, onRouteResult, onRouteModel }: ModelValidationViewProps) {
  const models = useValidationModels(scope);
  const random = useRandomModelValidation(scope);
  const [selectedModel, setSelectedModel] = useState("");
  const [threshold, setThreshold] = useState(.5);
  useEffect(() => {
    const items = models.data?.items || [];
    if (!items.some((item) => item.key === selectedModel && !item.error)) {
      const preferred = items.some((item) => item.key === routeModel && !item.error) ? routeModel : models.data?.default;
      setSelectedModel(preferred || items.find((item) => !item.error)?.key || "");
    }
  }, [models.data, routeModel, selectedModel]);
  const selected = models.data?.items.find((item) => item.key === selectedModel);
  const predictionRegion = routeLakeRegion || (scope === "all" ? "" : scope);
  const randomModelMatches = Boolean(random.data && (random.data.model.key === selectedModel || selectedModel === `${random.data.site?.region || random.data.lake?.region || random.data.region}/${random.data.model.key}`));
  const randomMatchesRoute = Boolean(random.data && (random.data.site_id || random.data.lake_id) === routeLakeId && (random.data.site?.region || random.data.lake?.region || random.data.region) === predictionRegion && randomModelMatches);
  const prediction = useLakeModelPrediction(predictionRegion, routeLakeId, selectedModel, threshold, Boolean(routeLakeId && !randomMatchesRoute));
  const result = randomMatchesRoute ? random.data : prediction.data;
  return <Box sx={{ height: "100%", minHeight: 0, overflow: "auto", display: "grid", gridTemplateRows: "auto minmax(0,1fr)" }}>
    <Box sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider" }}>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
        <IconButton sx={{ display: { md: "none" } }} onClick={onBack}><ArrowLeft size={19} /></IconButton>
        <FormControl sx={{ minWidth: { xs: 230, sm: 300 }, flex: 1 }}><InputLabel>模型权重</InputLabel><Select label="模型权重" value={selectedModel} onChange={(event) => { const model = event.target.value; setSelectedModel(model); random.reset(); onRouteModel(model); }}>{(models.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key} disabled={Boolean(item.error)}>{item.label}{Number.isFinite(Number(item.best_iou)) ? ` · IoU ${metric(item.best_iou)}` : ""}{item.error ? " · 不可用" : ""}</MenuItem>)}</Select></FormControl>
        <Box sx={{ width: 180, display: "flex", alignItems: "center", gap: 1 }}><Typography variant="caption">阈值</Typography><Slider min={.05} max={.95} step={.05} value={threshold} onChange={(_, value) => setThreshold(value as number)} /><Typography variant="caption">{threshold.toFixed(2)}</Typography></Box>
        <Button variant="contained" startIcon={<Dices size={16} />} disabled={!selectedModel || random.isPending || prediction.isFetching || models.isLoading} onClick={() => random.mutate({ model: selectedModel, threshold }, { onSuccess: (value) => onRouteResult(value.site_id || value.lake_id, value.site?.region || value.lake?.region || value.region, selectedModel) })}>{random.isPending ? "推理中" : "随机验证一个区域"}</Button>
      </Box>
      {models.isLoading ? <LinearProgress sx={{ mt: 1 }} /> : models.isError ? <Alert severity="error" sx={{ mt: 1 }}>{models.error.message}</Alert> : <ModelDetail model={selected} />}
      {random.isError && <Alert severity="error" sx={{ mt: 1 }}>{random.error.message}</Alert>}
      {prediction.isError && <Alert severity="error" sx={{ mt: 1 }}>{prediction.error.message}</Alert>}
    </Box>
    {prediction.isFetching ? <Box sx={{ display: "grid", placeItems: "center", minHeight: 320 }}><CircularProgress size={30} /></Box> : result ? <ValidationWorkspace key={`${result.region}:${result.site_id || result.lake_id}:${result.model.key}:${result.model.threshold}`} result={result} threshold={Number(result.model.threshold ?? threshold)} /> : <Box sx={{ display: "grid", placeItems: "center", minHeight: 320 }}><Typography color="text.secondary">选择模型后随机验证一个有可用影像的观测区域</Typography></Box>}
  </Box>;
}
