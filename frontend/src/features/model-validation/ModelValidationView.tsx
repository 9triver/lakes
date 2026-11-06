import { preferredValidationModel } from "./modelSelection";
import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import { workspaceRegionApi } from "../workspaces/api";
import { Alert, Autocomplete, Box, Button, CircularProgress, FormControl, IconButton, InputLabel, LinearProgress, MenuItem, Select, Slider, TextField, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { ArrowLeft, Dices } from "lucide-react";
import { useContextWater, useEsaLayer, useHydrolakesLayer, useJrcLayer, useOsmLayer, useSite, useSites, useLocalLabel, useTileMeta } from "../sites/api";
import { useRegions } from "../regions/api";
import { ImageryPanel } from "../imagery/ImageryPanel";
import type { ImagerySelection } from "../imagery/ImageryPanel";
import { useSentinelTiles } from "../imagery/api";
import { DEFAULT_SITE_LAYER_VISIBILITY, SiteMap, type BasemapType, type SiteLayerVisibility, type SiteMapHandle } from "../map/SiteMap";
import { SiteMapToolbar } from "../map/SiteMapToolbar";
import { TrainingCaptureButton, TrainingCaptureStatus, useTrainingCapture } from "../training-samples/TrainingCapture";
import { type ModelValidationResult, type ModelOption, useSiteModelPrediction, useRandomModelValidation, useRandomModelValidationTarget, useValidationModels } from "./api";
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

interface TerrainPreparation {
  status: "queued" | "running" | "ready" | "failed";
  error?: string;
  progress: Record<string, { stage: string; message: string; completed?: number | null; total?: number | null }>;
}

function requiresSlope(model?: ModelOption): boolean {
  const config = model?.config || {};
  const bands = config.input_bands || config.input_features;
  return Array.isArray(bands) ? bands.includes("slope_degrees") : model?.in_channels === 6 && Boolean(config.slope_features);
}

function ValidationWorkspace({ slopeRequired, workspaceId, result, threshold, hasPrediction = true, onPredict, predictionPending, canPredict, onTrainingDataGenerated, onImageryReady }: { onImageryReady?: () => void; slopeRequired: boolean; onPredict: () => void; predictionPending: boolean; canPredict: boolean; hasPrediction?: boolean; workspaceId: string; result: NonNullable<ReturnType<typeof useRandomModelValidation>["data"]>; threshold: number; onTrainingDataGenerated?: (region: string, sampleId: string, siteId: string) => void }) {
  const region = result.site?.region || result.region;
  const siteId = result.site_id;
  const [terrainRetry, setTerrainRetry] = useState(0);
  const terrain = useQuery({
    queryKey: ["single-model-terrain", workspaceId, region, siteId, terrainRetry],
    queryFn: () => getJson<TerrainPreparation>(workspaceRegionApi(workspaceId, region,
      `/sites/${encodeURIComponent(siteId)}/model-comparison/preparation?${new URLSearchParams({ reference: "single_model_terrain", retry: String(terrainRetry) })}`)),
    enabled: slopeRequired && Boolean(workspaceId && region && siteId),
    retry: false,
    refetchInterval: (query) => ["queued", "running"].includes(query.state.data?.status || "") ? 1000 : false,
    refetchOnWindowFocus: false, refetchOnReconnect: false,
  });
  const terrainReady = !slopeRequired || terrain.data?.status === "ready";
  const terrainFailed = terrain.isError || terrain.data?.status === "failed";
  const terrainProgress = terrain.data?.progress.terrain;
  const terrainFraction = terrainProgress?.total && terrainProgress.completed != null
    ? Math.min(100, Math.max(0, terrainProgress.completed / terrainProgress.total * 100)) : undefined;

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
    modelValidation: hasPrediction ? { model_key: result.model.key, model_name: result.model.name, model_path: result.model.path || "", threshold: result.model.threshold ?? threshold, predicted_area_km2: Number(result.stats.area_km2 || 0), predicted_ratio: Number(result.stats.predicted_ratio || 0), prediction_feature_count: result.prediction.features.length, device: result.model.device || "" } : undefined,
    onComplete: (sampleId) => onTrainingDataGenerated?.(region, sampleId, result.site_id),
  });
  if (site.isLoading) return <Box sx={{ display: "grid", placeItems: "center", minHeight: 360 }}><CircularProgress size={28} /></Box>;
  if (!site.data) return <Alert severity="error">无法加载推理区域</Alert>;
  return <Box sx={{ display: "grid", gridTemplateRows: `${slopeRequired ? "auto " : ""}auto minmax(420px,1fr) auto`, gridTemplateColumns: "minmax(0,1fr)", minWidth: 0, minHeight: 0, height: "100%" }}>
    {slopeRequired && <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider" }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography variant="body2" color={terrainFailed ? "error" : "text.primary"} sx={{ flex: 1 }}>
          {terrainFailed ? `坡度准备失败：${terrain.error?.message || terrain.data?.error || "未知错误"}`
            : terrainReady ? "坡度已就绪，可运行模型预测"
            : `此模型需要坡度 · ${terrainProgress?.message || (terrain.data?.status === "queued" ? "正在排队准备坡度" : "正在检查坡度缓存")}`}
        </Typography>
        {terrainFailed && <Button size="small" onClick={() => setTerrainRetry((value) => value + 1)}>重试坡度准备</Button>}
      </Box>
      {!terrainReady && !terrainFailed && <LinearProgress sx={{ mt: 1 }} variant={terrainFraction == null ? "indeterminate" : "determinate"} value={terrainFraction} />}
    </Box>}
    <SiteMapToolbar
      title={site.data.display_name || siteId}
      subtitle={predictionPending ? "地图已加载，正在生成预测结果…" : hasPrediction ? `模型 ${result.model.name} · 阈值 ${threshold.toFixed(2)} · 水体像元 ${metric(Number(result.stats.predicted_ratio || 0) * 100, 1)}%` : "地图已加载，点击“模型预测”后生成预测结果"}
      imageryControl={<ImageryPanel region={region} siteId={siteId} compact onSelectionChange={handleImagery} />}
      predictionAction={<Button variant="contained" disabled={!canPredict || predictionPending || !terrainReady} onClick={onPredict}>{predictionPending ? "预测中" : !terrainReady ? "坡度未就绪" : "模型预测"}</Button>}
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
      trainingAction={hasPrediction ? <TrainingCaptureButton controller={trainingCapture} /> : undefined}
      showPrediction={hasPrediction}
      canFitSite={Boolean(tileMeta.data?.site_bounds || site.data.bbox)}
      canFitTile={Boolean(tileMeta.data?.tile_bounds)}
      onFitSite={() => mapRef.current?.fitSite()}
      onFitTile={() => mapRef.current?.fitTile()}
    />
    <SiteMap ref={mapRef} onImageryReady={onImageryReady} site={site.data} basemap={basemap} visibility={layerVisibility} tileMeta={tileMeta.data} sentinelTiles={sentinelTiles.data} osm={osm.data} hydrolakes={hydrolakes.data} contextOsm={context.data?.sources.osm} contextHydro={context.data?.sources.hydrolakes} esa={esa.data} jrc={jrc.data} localLabel={localLabel.data} spectralWater={generatedLabels.spectral_water?.label} spectralOsmIntersection={generatedLabels.spectral_osm_intersection?.label} spectralOsmConsensus={generatedLabels.spectral_osm_consensus?.label} osmSpectralConsensus={generatedLabels.osm_spectral_consensus?.label} modelPrediction={hasPrediction ? result.prediction : undefined} />
    <Box sx={{ maxHeight: "38vh", overflow: "auto" }}>
      <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider" }}><Typography variant="body2">{hasPrediction ? `${site.data.display_name || siteId} · 模型 ${result.model.name} · 阈值 ${threshold.toFixed(2)} · 水体像元 ${metric(Number(result.stats.predicted_ratio || 0) * 100, 1)}% · ${result.model.device || ""}` : `${site.data.display_name || siteId} · 尚未运行模型预测`}</Typography></Box>
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
  const random = useRandomModelValidationTarget(workspaceId);
  const queryClient = useQueryClient();
  const selectionVersion = useRef(0);
  const [selectedModel, setSelectedModel] = useState("");
  const [threshold, setThreshold] = useState(.5);
  const [requestedIdentity, setRequestedIdentity] = useState("");
  const [imageryReadyIdentity, setImageryReadyIdentity] = useState("");
  const [imageryTimedOutIdentity, setImageryTimedOutIdentity] = useState("");
  const [validationRegion, setValidationRegion] = useState(routeSiteRegion || (scope === "all" ? "" : scope));
  const [siteQuery, setSiteQuery] = useState("");
  const [sitePickerOpen, setSitePickerOpen] = useState(false);
  const regions = useRegions();
  const validationSites = useSites(workspaceId, validationRegion, siteQuery, sitePickerOpen);
  const currentSite = useSite(routeSiteRegion || (scope === "all" ? "" : scope), routeSiteId, workspaceId);
  const loadedOptions = validationSites.data?.pages.flatMap((page) => page.items) || [];
  const selectedSite = currentSite.data?.region === validationRegion ? currentSite.data :
    loadedOptions.find((site) => site.site_id === routeSiteId && site.region === validationRegion) || null;
  const siteOptions = selectedSite && !loadedOptions.some((site) => site.site_id === selectedSite.site_id && site.region === selectedSite.region)
    ? [selectedSite, ...loadedOptions] : loadedOptions;
  const siteListLoading = sitePickerOpen && validationSites.isFetching;
  useEffect(() => {
    if (routeSiteRegion) setValidationRegion(routeSiteRegion);
    setSitePickerOpen(false);
    setSiteQuery("");
  }, [routeSiteRegion, routeSiteId]);
  useEffect(() => {
    const items = models.data?.items || [];
    if (!items.some((item) => item.key === selectedModel && !item.error)) {
      const routeMatch = items.find((item) => !item.error && (item.key === routeModel || item.key.endsWith(`/${routeModel}`)));
      const preferred = routeMatch?.key || preferredValidationModel(items, models.data?.default);
      setSelectedModel(preferred || items.find((item) => !item.error)?.key || "");
    }
  }, [models.data, routeModel, selectedModel]);
  const selected = models.data?.items.find((item) => item.key === selectedModel);
  const predictionRegion = routeSiteRegion || (scope === "all" ? "" : scope);
  const predictionIdentity = `${workspaceId}:${predictionRegion}:${routeSiteId}:${selectedModel}:${threshold}`;
  const predictionRequested = requestedIdentity === predictionIdentity;
  const imageryIdentity = `${workspaceId}:${predictionRegion}:${routeSiteId}`;
  const handleImageryReady = useCallback(() => setImageryReadyIdentity(imageryIdentity), [imageryIdentity]);
  const waitingForImagery = predictionRequested && imageryReadyIdentity !== imageryIdentity && imageryTimedOutIdentity !== imageryIdentity;
  useEffect(() => {
    if (!waitingForImagery) return;
    const timer = window.setTimeout(() => setImageryTimedOutIdentity(imageryIdentity), 15000);
    return () => window.clearTimeout(timer);
  }, [waitingForImagery, imageryIdentity]);
  const prediction = useSiteModelPrediction(workspaceId, predictionRegion, routeSiteId, selectedModel, threshold, Boolean(routeSiteId && predictionRequested && !waitingForImagery));
  const result = predictionRequested && !waitingForImagery && prediction.isSuccess && !prediction.isFetching ? prediction.data : undefined;
  const mapResult: ModelValidationResult | undefined = result || (routeSiteId && predictionRegion ? {
    region: predictionRegion, site_id: routeSiteId, model: { key: selectedModel, name: selected?.label || "尚未运行预测" },
    prediction: { type: "FeatureCollection", features: [] }, stats: {},
  } : undefined);
  return <Box sx={{ height: "100%", minHeight: 0, overflow: "auto", display: "grid", gridTemplateRows: "auto minmax(0,1fr)", gridTemplateColumns: "minmax(0,1fr)", minWidth: 0 }}>
    <Box sx={{ p: { xs: 1.5, sm: 2 }, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider" }}>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
        <IconButton sx={{ display: { md: "none" } }} onClick={onBack}><ArrowLeft size={19} /></IconButton>
        <FormControl sx={{ minWidth: { xs: 230, sm: 300 }, flex: 1 }}><InputLabel id="model-weight-label">模型权重</InputLabel><Select id="model-weight" labelId="model-weight-label" label="模型权重" value={selectedModel} onChange={(event) => { const model = event.target.value; setSelectedModel(model); selectionVersion.current += 1; setRequestedIdentity(""); random.reset(); onRouteModel(model); }}>{(models.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key} disabled={Boolean(item.error)}>{item.label}{Number.isFinite(Number(item.best_iou)) ? ` · IoU ${metric(item.best_iou)}` : ""}{item.error ? " · 不可用" : ""}</MenuItem>)}</Select></FormControl>
        {allowForeignModels && <ToggleButtonGroup exclusive size="small" value={visibility} onChange={(_, value) => { if (value) { setVisibility(value); setSelectedModel(""); } }}><ToggleButton value="current">当前工作区</ToggleButton><ToggleButton value="all">全部工作区</ToggleButton></ToggleButtonGroup>}
        <FormControl sx={{ minWidth: 150 }}><InputLabel id="validation-region-label">验证区域</InputLabel><Select labelId="validation-region-label" label="验证区域" value={validationRegion} onChange={(event) => { setValidationRegion(event.target.value); setSiteQuery(""); }}>{(regions.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key}>{item.name} ({item.site_count})</MenuItem>)}</Select></FormControl>
        <Autocomplete
          sx={{ minWidth: { xs: "100%", sm: 280 }, flex: 1 }}
          open={sitePickerOpen}
          onOpen={() => setSitePickerOpen(true)}
          onClose={() => setSitePickerOpen(false)}
          options={siteOptions}
          value={selectedSite}
          loading={siteListLoading}
          getOptionLabel={(site) => site.display_name || site.site_id}
          isOptionEqualToValue={(option, value) => option.site_id === value.site_id && option.region === value.region}
          onInputChange={(_, value, reason) => { if (reason === "input") setSiteQuery(value); }}
          onChange={(_, site) => { if (site) { selectionVersion.current += 1; setRequestedIdentity(""); random.reset(); onRouteResult(site.site_id, validationRegion, selectedModel); } }}
          renderInput={(params) => <TextField {...params} label="选择观测区域" placeholder="搜索或选择 Site" InputProps={{ ...params.InputProps, endAdornment: <>{siteListLoading ? <CircularProgress color="inherit" size={16} /> : null}{params.InputProps.endAdornment}</> }} />}
          disabled={!validationRegion}
          noOptionsText={siteListLoading ? "正在加载" : "没有可用观测区域"}
        />
        <Box sx={{ width: 180, display: "flex", alignItems: "center", gap: 1 }}><Typography variant="caption">阈值</Typography><Slider min={.05} max={.95} step={.05} value={threshold} onChange={(_, value) => { selectionVersion.current += 1; setRequestedIdentity(""); setThreshold(value as number); }} /><Typography variant="caption">{threshold.toFixed(2)}</Typography></Box>
        <Button variant="outlined" startIcon={<Dices size={16} />} disabled={!selectedModel || random.isPending || waitingForImagery || prediction.isFetching || models.isLoading} onClick={() => {
          const version = ++selectionVersion.current;
          setSitePickerOpen(false);
          setSiteQuery("");
          setRequestedIdentity("");
          random.mutate(undefined, { onSuccess: (value) => {
            if (version !== selectionVersion.current) return;
            const region = value.site.region || value.region;
            const siteKey = ["site", workspaceId, region, value.site_id];
            if (!queryClient.getQueryData(siteKey)) queryClient.setQueryData(siteKey, value.site);
            setRequestedIdentity(`${workspaceId}:${region}:${value.site_id}:${selectedModel}:${threshold}`);
            onRouteResult(value.site_id, region, selectedModel);
          } });
        }}>{random.isPending ? "正在选择区域" : waitingForImagery ? "加载五光谱影像中" : prediction.isFetching ? "推理中" : "随机验证一个区域"}</Button>
      </Box>
      {models.isLoading ? <LinearProgress sx={{ mt: 1 }} /> : models.isError ? <Alert severity="error" sx={{ mt: 1 }}>{models.error.message}</Alert> : <ModelDetail model={selected} />}
      {random.isError && <Alert severity="error" sx={{ mt: 1 }}>{random.error.message}</Alert>}
      {waitingForImagery && <Alert severity="info" sx={{ mt: 1 }}>正在优先加载本地影像瓦片，加载后自动开始推理。</Alert>}
      {imageryTimedOutIdentity === imageryIdentity && imageryReadyIdentity !== imageryIdentity && <Alert severity="warning" sx={{ mt: 1 }}>影像瓦片加载超过15秒，已继续推理；影像仍在后台加载。</Alert>}
      {prediction.isError && <Alert severity="error" sx={{ mt: 1 }}>{prediction.error.message}</Alert>}
    </Box>
    <Box sx={{ minHeight: 0, position: "relative" }}>
      {prediction.isFetching && <LinearProgress sx={{ position: "absolute", top: 0, left: 0, right: 0, zIndex: 2 }} />}
      {mapResult ? <ValidationWorkspace onImageryReady={handleImageryReady} slopeRequired={requiresSlope(selected)} workspaceId={workspaceId} key={`${mapResult.region}:${mapResult.site_id}`} result={mapResult} hasPrediction={Boolean(result)} threshold={threshold} predictionPending={prediction.isFetching} canPredict={Boolean(selectedModel && routeSiteId && !random.isPending)} onPredict={() => {
        random.reset();
        if (predictionRequested) void prediction.refetch();
        else setRequestedIdentity(predictionIdentity);
      }} onTrainingDataGenerated={onTrainingDataGenerated} /> : <Box sx={{ display: "grid", placeItems: "center", minHeight: 320 }}><Typography color="text.secondary">先选择观测区域查看地图，再点击“模型预测”运行模型</Typography></Box>}
    </Box>
  </Box>;
}
