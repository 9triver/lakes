import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, Box, Button, CircularProgress, FormControl, IconButton, InputLabel, LinearProgress, MenuItem, Select, Slider, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { ArrowLeft, Dices } from "lucide-react";
import { useContextWater, useEsaLayer, useHydrolakesLayer, useJrcLayer, useOsmLayer, useSite, useLocalLabel, useTileMeta } from "../sites/api";
import { ImageryPanel } from "../imagery/ImageryPanel";
import type { ImagerySelection } from "../imagery/ImageryPanel";
import { useSentinelTiles } from "../imagery/api";
import { DEFAULT_SITE_LAYER_VISIBILITY, SiteMap, type BasemapType, type SiteLayerVisibility, type SiteMapHandle } from "../map/SiteMap";
import { SiteMapToolbar } from "../map/SiteMapToolbar";
import { TrainingCaptureButton, TrainingCaptureStatus, useTrainingCapture } from "../training-samples/TrainingCapture";
import { type ModelOption, useSiteModelPrediction, useRandomModelValidation, useValidationModels } from "./api";
import type { GeneratedLabelResults } from "../../api/types";
import { GeneratedLabelStatus, useGeneratedLabel } from "../automatic-labels/GeneratedLabel";

function metric(value?: number, digits = 4) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-"; }
function modelLabel(value?: string) { return value === "pixel_mlp" ? "Pixel MLP" : "U-Net"; }

function ModelDetail({ model }: { model?: ModelOption }) {
  if (!model) return null;
  const config = model.config || {};
  const fields = [
    ["范围", model.scope === "all" ? "全部区域" : model.scope || model.region || "-"],
    ["Best IoU", metric(model.best_iou)], ["Best epoch", model.best_epoch || "-"],
    ["最新 val IoU", metric(model.latest?.val?.iou)], ["最新 train IoU", metric(model.latest?.train?.iou)],
    ["模型", modelLabel(model.model_type)],
    ["输入", `${model.in_channels || "-"} 波段 · ${model.architecture_label || `U-Net (base ${model.base_channels || "-"})`}`],
    ["训练 / 验证", config.train_count != null ? `${config.train_count} / ${config.val_count || 0}` : "-"],
    ["权重", model.weight || "-"], ["路径", model.path || "-"],
  ];
  return <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", border: 1, borderColor: "divider", mt: 1 }}>
    {fields.map(([label, value]) => <Box key={label} sx={{ px: 1.25, py: .75, minWidth: 0, borderRight: 1, borderBottom: 1, borderColor: "divider" }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" noWrap title={String(value)}>{value}</Typography></Box>)}
  </Box>;
}

function ValidationWorkspace({ workspaceId, result, threshold, onTrainingDataGenerated }: { workspaceId: string; result: NonNullable<ReturnType<typeof useRandomModelValidation>["data"]>; threshold: number; onTrainingDataGenerated?: (region: string, sampleId: string, siteId: string) => void }) {
  const region = result.site?.region || result.region;
  const siteId = result.site_id;
  const mapRef = useRef<SiteMapHandle>(null);
  const [jrcThreshold, setJrcThreshold] = useState(75);
  const [basemap, setBasemap] = useState<BasemapType>("satellite");
  const [layerVisibility, setLayerVisibility] = useState<SiteLayerVisibility>(() => ({ ...DEFAULT_SITE_LAYER_VISIBILITY }));
  const [imagerySelection, setImagerySelection] = useState<ImagerySelection>({ assetId: "", tile: result.imagery?.tile || result.imagery?.tiles?.[0] || "", product: result.imagery?.product || result.imagery?.product_name || result.imagery?.products?.[0] || "", localLabelId: "", localLabel: null });
  const [generatedLabels, setGeneratedLabels] = useState<GeneratedLabelResults>({});
  const site = useSite(region, siteId, workspaceId);
  const tileMeta = useTileMeta(region, siteId);
  const sentinelTiles = useSentinelTiles(region, siteId);
  const context = useContextWater(region, siteId);
  const osm = useOsmLayer(region, siteId);
  const hydrolakes = useHydrolakesLayer(region, siteId);
  const esa = useEsaLayer(region, siteId);
  const jrc = useJrcLayer(region, siteId, jrcThreshold);
  const localLabel = useLocalLabel(region, siteId, imagerySelection.localLabelId);
  const spectralWaterGeneration = useGeneratedLabel({
    workspaceId,
    region,
    siteId,
    source: "spectral_water",
    imagery: imagerySelection,
    mapHandle: mapRef,
    onGenerated: (value) => setGeneratedLabels((current) => ({ ...current, spectral_water: value })),
  });
  const spectralOsmIntersectionGeneration = useGeneratedLabel({
    workspaceId,
    region,
    siteId,
    source: "spectral_osm_intersection",
    imagery: imagerySelection,
    mapHandle: mapRef,
    onGenerated: (value) => setGeneratedLabels((current) => ({ ...current, spectral_osm_intersection: value })),
  });
  const spectralOsmConsensusGeneration = useGeneratedLabel({
    workspaceId,
    region,
    siteId,
    source: "spectral_osm_consensus",
    imagery: imagerySelection,
    mapHandle: mapRef,
    onGenerated: (value) => setGeneratedLabels((current) => ({ ...current, spectral_osm_consensus: value })),
  });
  const osmSpectralConsensusGeneration = useGeneratedLabel({
    workspaceId,
    region,
    siteId,
    source: "osm_spectral_consensus",
    imagery: imagerySelection,
    mapHandle: mapRef,
    onGenerated: (value) => setGeneratedLabels((current) => ({ ...current, osm_spectral_consensus: value })),
  });
  const handleImagery = useCallback((selection: ImagerySelection) => {
    setImagerySelection(selection);
  }, []);
  useEffect(() => {
    setGeneratedLabels({});
    setLayerVisibility((current) => ({ ...current, spectralWater: false, spectralOsmIntersection: false, spectralOsmConsensus: false, osmSpectralConsensus: false }));
  }, [imagerySelection.assetId]);
  const trainingCapture = useTrainingCapture({
    workspaceId,
    region,
    siteId,
    jrcThreshold,
    localLabel: imagerySelection.localLabel || undefined,
    imagery: imagerySelection,
    mapHandle: mapRef,
    generatedLabels,
    modelValidation: { model_key: result.model.key, model_name: result.model.name, model_path: result.model.path || "", threshold: result.model.threshold ?? threshold, predicted_area_km2: Number(result.stats.area_km2 || 0), predicted_ratio: Number(result.stats.predicted_ratio || 0), prediction_feature_count: result.prediction.features.length, device: result.model.device || "" },
    onComplete: (sampleId) => onTrainingDataGenerated?.(region, sampleId, result.site_id),
  });
  if (site.isLoading) return <Box sx={{ display: "grid", placeItems: "center", minHeight: 360 }}><CircularProgress size={28} /></Box>;
  if (!site.data) return <Alert severity="error">无法加载推理区域</Alert>;
  return <Box sx={{ display: "grid", gridTemplateRows: "auto minmax(420px,1fr) auto", minHeight: 0 }}>
    <SiteMapToolbar
      title={site.data.display_name || siteId}
      subtitle={`模型 ${result.model.name} · 阈值 ${threshold.toFixed(2)} · 水体像元 ${metric(Number(result.stats.predicted_ratio || 0) * 100, 1)}%`}
      imageryControl={<ImageryPanel region={region} siteId={siteId} compact onSelectionChange={handleImagery} />}
      basemap={basemap}
      onBasemapChange={setBasemap}
      visibility={layerVisibility}
      onVisibilityChange={(layer, visible) => {
        setLayerVisibility((current) => ({ ...current, [layer]: visible }));
        if (!visible) return;
        if (layer === "spectralWater" && !generatedLabels.spectral_water) spectralWaterGeneration.mutate();
        if (layer === "spectralOsmIntersection" && !generatedLabels.spectral_osm_intersection) spectralOsmIntersectionGeneration.mutate();
        if (layer === "spectralOsmConsensus" && !generatedLabels.spectral_osm_consensus) spectralOsmConsensusGeneration.mutate();
        if (layer === "osmSpectralConsensus" && !generatedLabels.osm_spectral_consensus) osmSpectralConsensusGeneration.mutate();
      }}
      jrcThreshold={jrcThreshold}
      onJrcThresholdChange={setJrcThreshold}
      trainingAction={<TrainingCaptureButton controller={trainingCapture} />}
      showPrediction
      canFitSite={Boolean(tileMeta.data?.site_bounds || site.data.bbox)}
      canFitTile={Boolean(tileMeta.data?.tile_bounds)}
      onFitSite={() => mapRef.current?.fitSite()}
      onFitTile={() => mapRef.current?.fitTile()}
    />
    <SiteMap ref={mapRef} site={site.data} basemap={basemap} visibility={layerVisibility} tileMeta={tileMeta.data} sentinelTiles={sentinelTiles.data} osm={osm.data} hydrolakes={hydrolakes.data} contextOsm={context.data?.sources.osm} contextHydro={context.data?.sources.hydrolakes} esa={esa.data} jrc={jrc.data} localLabel={localLabel.data} spectralWater={generatedLabels.spectral_water?.label} spectralOsmIntersection={generatedLabels.spectral_osm_intersection?.label} spectralOsmConsensus={generatedLabels.spectral_osm_consensus?.label} osmSpectralConsensus={generatedLabels.osm_spectral_consensus?.label} modelPrediction={result.prediction} />
    <Box sx={{ maxHeight: "38vh", overflow: "auto" }}>
      <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider" }}><Typography variant="body2">{site.data.display_name || siteId} · 模型 {result.model.name} · 阈值 {threshold.toFixed(2)} · 水体像元 {metric(Number(result.stats.predicted_ratio || 0) * 100, 1)}% · {result.model.device || ""}</Typography></Box>
      <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <TrainingCaptureStatus controller={trainingCapture} />
        {layerVisibility.spectralWater && <GeneratedLabelStatus source="spectral_water" result={generatedLabels.spectral_water} pending={spectralWaterGeneration.isPending} error={spectralWaterGeneration.error} />}
        {layerVisibility.spectralOsmIntersection && <GeneratedLabelStatus source="spectral_osm_intersection" result={generatedLabels.spectral_osm_intersection} pending={spectralOsmIntersectionGeneration.isPending} error={spectralOsmIntersectionGeneration.error} />}
        {layerVisibility.spectralOsmConsensus && <GeneratedLabelStatus source="spectral_osm_consensus" result={generatedLabels.spectral_osm_consensus} pending={spectralOsmConsensusGeneration.isPending} error={spectralOsmConsensusGeneration.error} />}
        {layerVisibility.osmSpectralConsensus && <GeneratedLabelStatus source="osm_spectral_consensus" result={generatedLabels.osm_spectral_consensus} pending={osmSpectralConsensusGeneration.isPending} error={osmSpectralConsensusGeneration.error} />}
      </Box>
    </Box>
  </Box>;
}

interface ModelValidationViewProps {
  workspaceId: string;
  scope: string;
  routeSiteId?: string;
  routeSiteRegion?: string;
  routeModel?: string;
  onBack: () => void;
  onRouteResult: (siteId: string, siteRegion: string, model: string) => void;
  onRouteModel: (model: string) => void;
  allowForeignModels?: boolean;
  onTrainingDataGenerated?: (region: string, sampleId: string, siteId: string) => void;
}

export function ModelValidationView({ workspaceId, scope, routeSiteId = "", routeSiteRegion = "", routeModel = "", onBack, onRouteResult, onRouteModel, allowForeignModels = false, onTrainingDataGenerated }: ModelValidationViewProps) {
  const [visibility, setVisibility] = useState<"current" | "all">("current");
  const models = useValidationModels(workspaceId, scope, visibility);
  const random = useRandomModelValidation(workspaceId, scope);
  const [selectedModel, setSelectedModel] = useState("");
  const [threshold, setThreshold] = useState(.5);
  useEffect(() => {
    const items = models.data?.items || [];
    if (!items.some((item) => item.key === selectedModel && !item.error)) {
      const routeMatch = items.find((item) => !item.error && (item.key === routeModel || item.key.endsWith(`/${routeModel}`)));
      const preferred = routeMatch?.key || models.data?.default;
      setSelectedModel(preferred || items.find((item) => !item.error)?.key || "");
    }
  }, [models.data, routeModel, selectedModel]);
  const selected = models.data?.items.find((item) => item.key === selectedModel);
  const predictionRegion = routeSiteRegion || (scope === "all" ? "" : scope);
  const randomModelMatches = Boolean(random.data && (random.data.model.key === selectedModel || selectedModel === `${random.data.site?.region || random.data.region}/${random.data.model.key}`));
  const randomMatchesRoute = Boolean(random.data && random.data.site_id === routeSiteId && (random.data.site?.region || random.data.region) === predictionRegion && randomModelMatches);
  const prediction = useSiteModelPrediction(workspaceId, predictionRegion, routeSiteId, selectedModel, threshold, Boolean(routeSiteId && !randomMatchesRoute));
  const result = randomMatchesRoute ? random.data : prediction.data;
  return <Box sx={{ height: "100%", minHeight: 0, overflow: "auto", display: "grid", gridTemplateRows: "auto minmax(0,1fr)" }}>
    <Box sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider" }}>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
        <IconButton sx={{ display: { md: "none" } }} onClick={onBack}><ArrowLeft size={19} /></IconButton>
        <FormControl sx={{ minWidth: { xs: 230, sm: 300 }, flex: 1 }}><InputLabel id="model-weight-label">模型权重</InputLabel><Select id="model-weight" labelId="model-weight-label" label="模型权重" value={selectedModel} onChange={(event) => { const model = event.target.value; setSelectedModel(model); random.reset(); onRouteModel(model); }}>{(models.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key} disabled={Boolean(item.error)}>{item.label}{Number.isFinite(Number(item.best_iou)) ? ` · IoU ${metric(item.best_iou)}` : ""}{item.error ? " · 不可用" : ""}</MenuItem>)}</Select></FormControl>
        {allowForeignModels && <ToggleButtonGroup exclusive size="small" value={visibility} onChange={(_, value) => { if (value) { setVisibility(value); setSelectedModel(""); } }}><ToggleButton value="current">当前工作区</ToggleButton><ToggleButton value="all">全部工作区</ToggleButton></ToggleButtonGroup>}
        <Box sx={{ width: 180, display: "flex", alignItems: "center", gap: 1 }}><Typography variant="caption">阈值</Typography><Slider min={.05} max={.95} step={.05} value={threshold} onChange={(_, value) => setThreshold(value as number)} /><Typography variant="caption">{threshold.toFixed(2)}</Typography></Box>
        <Button variant="contained" startIcon={<Dices size={16} />} disabled={!selectedModel || random.isPending || prediction.isFetching || models.isLoading} onClick={() => random.mutate({ model: selectedModel, threshold }, { onSuccess: (value) => onRouteResult(value.site_id, value.site?.region || value.region, selectedModel) })}>{random.isPending ? "推理中" : "随机验证一个区域"}</Button>
      </Box>
      {models.isLoading ? <LinearProgress sx={{ mt: 1 }} /> : models.isError ? <Alert severity="error" sx={{ mt: 1 }}>{models.error.message}</Alert> : <ModelDetail model={selected} />}
      {random.isError && <Alert severity="error" sx={{ mt: 1 }}>{random.error.message}</Alert>}
      {prediction.isError && <Alert severity="error" sx={{ mt: 1 }}>{prediction.error.message}</Alert>}
    </Box>
    {prediction.isFetching ? <Box sx={{ display: "grid", placeItems: "center", minHeight: 320 }}><CircularProgress size={30} /></Box> : result ? <ValidationWorkspace workspaceId={workspaceId} key={`${result.region}:${result.site_id}:${result.model.key}:${result.model.threshold}`} result={result} threshold={Number(result.model.threshold ?? threshold)} onTrainingDataGenerated={onTrainingDataGenerated} /> : <Box sx={{ display: "grid", placeItems: "center", minHeight: 320 }}><Typography color="text.secondary">选择模型后随机验证一个有可用影像的观测区域</Typography></Box>}
  </Box>;
}
