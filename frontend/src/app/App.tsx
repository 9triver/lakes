import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Button, Chip, CircularProgress, FormControl, IconButton, InputLabel, List, ListItemButton, ListItemIcon, ListItemText, MenuItem, Select, Typography } from "@mui/material";
import { ArrowLeft, BrainCircuit, Database, Images, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useContextWater, useEsaLayer, useHydrolakesLayer, useJrcLayer, useOsmLayer, useSite, useSites, useLocalLabel, useTileMeta } from "../features/sites/api";
import { useRegions } from "../features/regions/api";
import { DEFAULT_SITE_LAYER_VISIBILITY, SiteMap, type BasemapType, type SiteLayerVisibility, type SiteMapHandle } from "../features/map/SiteMap";
import { SiteMapToolbar } from "../features/map/SiteMapToolbar";
import { ImageryPanel } from "../features/imagery/ImageryPanel";
import type { ImagerySelection } from "../features/imagery/ImageryPanel";
import { useSentinelTiles } from "../features/imagery/api";
import { SitePatchReviewPanel, type PatchGroup, type PatchOperation } from "../features/patches/SitePatchReviewPanel";
import { useBatchUpdateLogicalPatches, useLogicalPatchSourceMeta, useSiteLogicalPatches, useTrainingPatchSites } from "../features/patches/api";
import { TrainingCaptureButton, TrainingCaptureStatus, useTrainingCapture } from "../features/training-samples/TrainingCapture";
import { PatchReviewView } from "../features/patches/PatchReviewView";
import { TrainingSiteList } from "../features/patches/TrainingSiteList";
import { ModelWorkbench, type ModelStage } from "../features/models/ModelWorkbench";
import { ModelExperimentSidebar } from "../features/models/ModelExperimentSidebar";
import { useSiteModelPrediction, useValidationModels } from "../features/model-validation/api";
import { CurrentUserBar } from "../features/users/UserSelector";
import { UserRouteError } from "../features/users/UserRouteError";
import { AuthLoading, AuthRequired } from "../features/auth/AuthGate";
import { useAuthSession } from "../features/auth/api";
import { useWorkspace } from "../features/workspaces/api";
import type { TrainingPatch, WorkbenchUser } from "../api/types";
import { useWorkbenchStore } from "./store";

function Workbench({ user, logoutUrl }: { user: WorkbenchUser; logoutUrl: string }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [jrcThreshold, setJrcThreshold] = useState(75);
  const [basemap, setBasemap] = useState<BasemapType>("satellite");
  const [layerVisibility, setLayerVisibility] = useState<SiteLayerVisibility>(() => ({ ...DEFAULT_SITE_LAYER_VISIBILITY, prediction: false }));
  const [imagerySelection, setImagerySelection] = useState<ImagerySelection>({ assetId: "", tile: "", product: "", localLabelId: "", localLabel: null });
  const [patchReviewEnabled, setPatchReviewEnabled] = useState(false);
  const [patchGroupKey, setPatchGroupKey] = useState("");
  const [patchOperation, setPatchOperation] = useState<PatchOperation>("exclude");
  const [pendingPatchIds, setPendingPatchIds] = useState<Set<string>>(() => new Set());
  const [activePatchId, setActivePatchId] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const mapRef = useRef<SiteMapHandle>(null);
  const region = useWorkbenchStore((state) => state.region);
  const setRegion = useWorkbenchStore((state) => state.setRegion);
  const workspaceRoute = location.pathname.match(/^\/users\/[^/]+\/workspaces\/([^/]+)(\/regions\/.*)?$/);
  const workspaceId = workspaceRoute?.[1] || user.default_workspace_id;
  const workspacePath = workspaceRoute?.[2] || "";
  const activeWorkspaceId = workspaceId;
  const activeWorkspace = user.workspace;
  const workspacePrefix = `/users/${user.id}/workspaces/${activeWorkspaceId}`;
  const siteRoute = workspacePath.match(/^\/regions\/([^/]+)\/sites\/([^/]+)$/);
  const siteListRoute = workspacePath.match(/^\/regions\/([^/]+)\/sites$/);
  const trainingRoute = workspacePath.match(/^\/regions\/([^/]+)\/training-data(?:\/([^/]+))?$/);
  const modelRoute = workspacePath.match(/^\/regions\/([^/]+)\/models\/(train|validate)(?:\/([^/]+))?$/);
  const selectedRegion = siteRoute?.[1] || "";
  const selectedSiteId = siteRoute?.[2] || "";
  const trainingScope = trainingRoute?.[1] || region;
  const trainingSiteId = trainingRoute?.[2] || "";
  const isTraining = Boolean(trainingRoute);
  const isModel = Boolean(modelRoute);
  const modelStage: ModelStage = modelRoute?.[2] === "train" ? "train" : "validate";
  const modelSiteId = modelRoute?.[3] || "";
  const routeParams = new URLSearchParams(location.search);
  const routeModel = routeParams.get("model") || "";
  const modelRunId = routeParams.get("run") || "";
  const sourceSampleId = routeParams.get("source") || "";
  const modelSiteRegion = routeParams.get("site_region") || (modelRoute?.[1] === "all" ? "" : modelRoute?.[1] || "");
  const isSiteWorkspace = !isTraining && !isModel;
  const regions = useRegions();
  const trainingSites = useTrainingPatchSites(activeWorkspaceId, trainingScope, isTraining);
  const sites = useSites(activeWorkspaceId, region, "");
  const siteItems = sites.data?.pages.flatMap((page) => page.items) || [];
  const siteTotal = sites.data?.pages[0]?.total;
  const site = useSite(selectedRegion, selectedSiteId, activeWorkspaceId);
  const tileMeta = useTileMeta(selectedRegion, selectedSiteId);
  const sentinelTiles = useSentinelTiles(selectedRegion, selectedSiteId);
  const contextWater = useContextWater(selectedRegion, selectedSiteId);
  const osm = useOsmLayer(selectedRegion, selectedSiteId);
  const hydrolakes = useHydrolakesLayer(selectedRegion, selectedSiteId);
  const esa = useEsaLayer(selectedRegion, selectedSiteId);
  const jrc = useJrcLayer(selectedRegion, selectedSiteId, jrcThreshold);
  const localLabel = useLocalLabel(selectedRegion, selectedSiteId, imagerySelection.localLabelId);
  const validationModels = useValidationModels(activeWorkspaceId, selectedRegion, "current");
  const predictionModel = useMemo(() => {
    const items = validationModels.data?.items || [];
    return items.find((item) => !item.error && item.scope === selectedRegion)
      || items.find((item) => !item.error && item.scope === "all")
      || null;
  }, [selectedRegion, validationModels.data?.items]);
  const modelPrediction = useSiteModelPrediction(
    activeWorkspaceId,
    selectedRegion,
    selectedSiteId,
    predictionModel?.key || "",
    0.5,
    Boolean(selectedSiteId && predictionModel && layerVisibility.prediction),
  );
  const sitePatches = useSiteLogicalPatches(activeWorkspaceId, selectedRegion, selectedSiteId);
  const updatePatches = useBatchUpdateLogicalPatches(activeWorkspaceId, selectedRegion);
  const patchGroups = useMemo<PatchGroup[]>(() => {
    const groups = new Map<string, TrainingPatch[]>();
    for (const patch of sitePatches.data?.items || []) {
      const key = `${patch.sample_id}:${patch.image_index || 0}`;
      groups.set(key, [...(groups.get(key) || []), patch]);
    }
    return [...groups.entries()].map(([key, patches]) => ({ key, patches, label: `${patches[0]?.product_date || patches[0]?.product_name || patches[0]?.sample_id} · ${patches.length} 个` })).sort((a, b) => b.label.localeCompare(a.label));
  }, [sitePatches.data?.items]);
  const activePatchGroup = patchGroups.find((group) => group.key === patchGroupKey) || patchGroups[0];
  const visiblePatches = activePatchGroup?.patches || sitePatches.data?.items || [];
  const activePatch = visiblePatches.find((patch) => (patch.logical_patch_id || patch.patch_id) === activePatchId);
  const patchSource = useLogicalPatchSourceMeta(activeWorkspaceId, selectedRegion, selectedSiteId, visiblePatches[0]?.logical_patch_id || visiblePatches[0]?.patch_id || "", patchReviewEnabled);

  useEffect(() => {
    if (!region && regions.data) setRegion(siteListRoute?.[1] || siteRoute?.[1] || trainingRoute?.[1] || modelRoute?.[1] || regions.data.default);
  }, [siteListRoute, siteRoute, modelRoute, region, regions.data, setRegion, trainingRoute]);

  useEffect(() => {
    const workspaceScope = siteListRoute?.[1] || trainingRoute?.[1] || modelRoute?.[1];
    if (workspaceScope && region && workspaceScope !== region) setRegion(workspaceScope);
  }, [siteListRoute, modelRoute, region, setRegion, trainingRoute]);

  useEffect(() => {
    setImagerySelection({ assetId: "", tile: "", product: "", localLabelId: "", localLabel: null });
    setLayerVisibility((current) => ({ ...current, prediction: false }));
  }, [selectedRegion, selectedSiteId]);
  useEffect(() => {
    setPatchReviewEnabled(false);
    setPatchGroupKey("");
    setPendingPatchIds(new Set());
    setActivePatchId("");
  }, [selectedRegion, selectedSiteId]);
  useEffect(() => {
    if (!patchGroups.length) setPatchGroupKey("");
    else if (!patchGroups.some((group) => group.key === patchGroupKey)) setPatchGroupKey(patchGroups[0].key);
  }, [patchGroupKey, patchGroups]);
  const handleImagerySelection = useCallback((selection: ImagerySelection) => {
    setImagerySelection((current) => current.assetId === selection.assetId && current.tile === selection.tile && current.product === selection.product && current.localLabelId === selection.localLabelId ? current : selection);
  }, []);
  const trainingCapture = useTrainingCapture({
    workspaceId: activeWorkspaceId,
    region: selectedRegion,
    siteId: selectedSiteId,
    jrcThreshold,
    localLabel: imagerySelection.localLabel || undefined,
    imagery: imagerySelection,
    mapHandle: mapRef,
    onComplete: (sampleId) => navigate(`${workspacePrefix}/regions/${selectedRegion}/training-data/${encodeURIComponent(selectedSiteId)}?source=${encodeURIComponent(sampleId)}`),
  });
  const handleLayerVisibility = useCallback((layer: keyof SiteLayerVisibility, visible: boolean) => {
    setLayerVisibility((current) => ({ ...current, [layer]: visible }));
  }, []);
  const handlePatchClick = useCallback((patchId: string) => {
    setActivePatchId(patchId);
    const patch = visiblePatches.find((item) => (item.logical_patch_id || item.patch_id) === patchId);
    const selectable = patchOperation === "exclude" ? patch?.included : patch && !patch.included;
    if (!selectable) return;
    setPendingPatchIds((current) => {
      const next = new Set(current);
      if (next.has(patchId)) next.delete(patchId); else next.add(patchId);
      return next;
    });
  }, [patchOperation, visiblePatches]);
  const handleRegionChange = (nextRegion: string) => {
    setRegion(nextRegion);
    navigate(isTraining ? `${workspacePrefix}/regions/${nextRegion}/training-data` : isModel ? `${workspacePrefix}/regions/${nextRegion}/models/${modelStage}` : `${workspacePrefix}/regions/${nextRegion}/sites`);
  };
  const handleTrainingSiteSelect = (siteId: string) => {
    navigate(`${workspacePrefix}/regions/${trainingScope}/training-data${siteId ? `/${encodeURIComponent(siteId)}` : ""}`);
  };
  const handleModelRunSelect = (runId: string) => {
    navigate(runId ? `${workspacePrefix}/regions/${region}/models/train?run=${encodeURIComponent(runId)}` : `${workspacePrefix}/regions/${region}/models/train`);
  };

  useEffect(() => {
    if (workspaceId && !workspaceRoute?.[2] && regions.data) {
      navigate(`${workspacePrefix}/regions/${regions.data.default}/sites`, { replace: true });
    }
  }, [navigate, workspaceId, workspacePrefix, workspaceRoute, regions.data]);

  const sidebarHiddenOnMobile = selectedSiteId || isTraining || isModel;
  const sidebarWidth = sidebarCollapsed ? 76 : 360;
  const openSites = () => navigate(`${workspacePrefix}/regions/${region}/sites`);
  const openTraining = () => navigate(`${workspacePrefix}/regions/${region}/training-data`);
  const openModel = () => navigate(`${workspacePrefix}/regions/${region}/models/validate`);
  const navItems = [
    { label: "观测区域", icon: <Images size={18} />, selected: isSiteWorkspace, onClick: openSites },
    { label: "训练数据", icon: <Database size={18} />, selected: isTraining, onClick: openTraining },
    { label: "模型实验", icon: <BrainCircuit size={18} />, selected: isModel, onClick: openModel },
  ];
  const regionControl = <FormControl fullWidth size="small">
    <InputLabel id="sidebar-region-label">区域</InputLabel>
    <Select labelId="sidebar-region-label" label="区域" value={region} onChange={(event) => handleRegionChange(event.target.value)}>
      <MenuItem value="all">全部区域</MenuItem>
      {(regions.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key}>{item.name} ({item.site_count})</MenuItem>)}
    </Select>
  </FormControl>;

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: `${sidebarWidth}px minmax(0, 1fr)` }, height: "100dvh", minHeight: 0, overflow: "hidden", bgcolor: "background.default", transition: "grid-template-columns .2s ease" }}>
      <Box component="aside" sx={{ display: { xs: sidebarHiddenOnMobile ? "none" : "grid", md: "grid" }, gridTemplateRows: sidebarCollapsed ? "auto minmax(0, 1fr)" : "auto minmax(0, 1fr) auto", minWidth: 0, minHeight: 0, height: "100%", borderRight: 1, borderColor: "divider", bgcolor: "background.paper", overflow: "hidden" }}>
        {sidebarCollapsed ? <>
          <Box sx={{ p: 1, display: "grid", justifyItems: "center", gap: .5, borderBottom: 1, borderColor: "divider" }}>
            <Typography variant="h6" color="primary.main" aria-label="湖泊工作台">湖</Typography>
            <IconButton size="small" onClick={() => setSidebarCollapsed(false)} aria-label="展开侧栏" title="展开侧栏"><PanelLeftOpen size={18} /></IconButton>
          </Box>
          <List disablePadding sx={{ pt: 1 }}>
            {navItems.map((item) => <ListItemButton key={item.label} selected={item.selected} onClick={item.onClick} sx={{ minHeight: 46, justifyContent: "center", px: 1 }} aria-label={item.label} title={item.label}><ListItemIcon sx={{ minWidth: 0 }}>{item.icon}</ListItemIcon></ListItemButton>)}
          </List>
        </> : <>
          <Box sx={{ px: 2, pt: 1.5, pb: 1, display: "grid", gap: .75, borderBottom: 1, borderColor: "divider" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Typography variant="h6" noWrap sx={{ minWidth: 0, flex: 1 }}>湖泊工作台</Typography>
              <IconButton size="small" onClick={() => setSidebarCollapsed(true)} aria-label="收起侧栏" title="收起侧栏"><PanelLeftClose size={18} /></IconButton>
            </Box>
            <Box component="nav" aria-label="工作模式">
              <List disablePadding sx={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: .5 }}>
                {navItems.map((item) => <ListItemButton key={item.label} selected={item.selected} onClick={item.onClick} sx={{ minWidth: 0, minHeight: 42, justifyContent: "center", gap: .5, px: .75 }}><ListItemIcon sx={{ minWidth: 0, flexShrink: 0 }}>{item.icon}</ListItemIcon><ListItemText primary={item.label} slotProps={{ primary: { variant: "body2", noWrap: true, textAlign: "center" } }} sx={{ flex: "0 1 auto", minWidth: 0 }} /></ListItemButton>)}
              </List>
            </Box>
          </Box>
          {isSiteWorkspace ? <Box sx={{ minHeight: 0, overflow: "hidden", display: "grid", gridTemplateRows: "auto minmax(0, 1fr)" }}>
            <Box sx={{ px: 2, py: 1.25, display: "grid", gap: .75, borderBottom: 1, borderColor: "divider" }}>
              {regionControl}
              <Typography variant="caption" color="text.secondary">{siteTotal != null ? `共 ${siteTotal} 个，已显示 ${siteItems.length} 个` : "加载中"}</Typography>
            </Box>
            <Box sx={{ minHeight: 0, overflow: "auto" }}>
              {sites.isLoading ? <Box sx={{ p: 3, textAlign: "center" }}><CircularProgress size={24} /></Box> : sites.isError ? <Typography color="error" sx={{ p: 2 }}>{sites.error.message}</Typography> : <List disablePadding>
                {!siteItems.length && <Box sx={{ p: 3, textAlign: "center" }}><Typography variant="body2">没有符合条件的观测区域</Typography></Box>}
                {siteItems.map((site, index) => <ListItemButton key={`${site.region || region}:${site.site_id}:${index}`} divider selected={site.site_id === selectedSiteId} onClick={() => navigate(`${workspacePrefix}/regions/${site.region || region}/sites/${site.site_id}`)} sx={{ alignItems: "flex-start", py: 1.25 }}>
                  <ListItemText
                    primary={<Box sx={{ display: "flex", alignItems: "center", gap: .75 }}><Typography variant="body2" noWrap sx={{ flex: 1 }}>{site.display_name || site.site_id}</Typography>{Boolean(site.included_logical_patch_count) && <Chip size="small" color="success" variant="outlined" label={`Patch ${site.included_logical_patch_count}`} />}{region === "all" && <Chip size="small" label={site.region_name || site.region} />}</Box>}
                    secondary={<Box component="span" sx={{ display: "grid", gap: .25, mt: .25 }}><Typography component="span" variant="caption" color="text.secondary">覆盖 {site.coverage_area_km2.toFixed(2)} km² · {site.image_count ? `${site.image_count} 期本地影像` : "暂无本地影像"}</Typography><Typography component="span" variant="caption" color="text.secondary">最近影像 {site.last_acquisition_date || "暂无日期"} · {site.label_asset_count || 0} 个本地标注</Typography></Box>}
                  />
                </ListItemButton>)}
                {sites.hasNextPage && <Box sx={{ p: 1.5 }}><Button fullWidth variant="outlined" disabled={sites.isFetchingNextPage} onClick={() => sites.fetchNextPage()}>{sites.isFetchingNextPage ? "加载中" : "加载更多"}</Button></Box>}
              </List>}
            </Box>
          </Box> : isTraining ? <TrainingSiteList workspaceId={activeWorkspaceId} scope={trainingScope} selectedSiteId={trainingSiteId} regionControl={regionControl} onSelect={handleTrainingSiteSelect} /> : <ModelExperimentSidebar workspaceId={activeWorkspaceId} scope={region} regionControl={regionControl} showRegionControl={modelStage !== "train"} selectedRunId={modelRunId} onSelectRun={handleModelRunSelect} />}
          <Box data-testid="sidebar-user" sx={{ minHeight: 80, px: 2, py: 1.25, position: "relative", zIndex: 1, borderTop: 1, borderColor: "divider", bgcolor: "background.paper" }}><CurrentUserBar user={user} logoutUrl={logoutUrl} /></Box>
        </>}
      </Box>
      <Box component="main" sx={{ minWidth: 0, minHeight: 0, height: "100%", overflow: "hidden", display: { xs: selectedSiteId || isTraining || isModel ? "grid" : "none", md: "grid" }, gridTemplateRows: { xs: selectedSiteId ? "auto auto minmax(0, 1fr) auto" : isTraining ? "auto minmax(0, 1fr)" : isModel ? "auto minmax(0, 1fr)" : "1fr", md: selectedSiteId ? "auto minmax(0, 1fr) auto" : "1fr" }, color: "text.secondary" }}>
        <Box sx={{ display: { xs: "block", md: "none" }, px: 1.5, py: .75, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}><CurrentUserBar compact user={user} logoutUrl={logoutUrl} /></Box>
        {isModel ? <ModelWorkbench workspaceId={activeWorkspaceId} scope={region} regionOptions={regions.data?.items || []} onScopeChange={handleRegionChange} stage={modelStage} defaults={activeWorkspace?.training_defaults} selectedRunId={modelRunId} onRunSelect={handleModelRunSelect} routeSiteId={modelSiteId} routeSiteRegion={modelSiteRegion} routeModel={routeModel} onBack={() => navigate(`${workspacePrefix}/regions/${region}/sites`)} onStageChange={(stage) => navigate(`${workspacePrefix}/regions/${region}/models/${stage}`)} onRouteModel={(model) => navigate(`${workspacePrefix}/regions/${region}/models/validate?model=${encodeURIComponent(model)}`, { replace: true })} onRouteResult={(siteId, siteRegion, model) => navigate(`${workspacePrefix}/regions/${region}/models/validate/${encodeURIComponent(siteId)}?model=${encodeURIComponent(model)}${region === "all" ? `&site_region=${encodeURIComponent(siteRegion)}` : ""}`)} onTrainingDataGenerated={(siteRegion, sampleId, siteId) => navigate(`${workspacePrefix}/regions/${siteRegion}/training-data/${encodeURIComponent(siteId)}?source=${encodeURIComponent(sampleId)}`)} allowForeignModels={user.role === "admin"} /> : isTraining ? <>
          <Box sx={{ display: { xs: "flex", md: "none" }, alignItems: "center", gap: 1, px: 1.5, py: .75, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}>
            <IconButton size="small" onClick={() => navigate(`${workspacePrefix}/regions/${trainingScope}/sites`)} aria-label="返回观测区域"><ArrowLeft size={19} /></IconButton>
            <FormControl size="small" sx={{ flex: 1, minWidth: 0 }}><InputLabel id="mobile-training-site-label">训练区域</InputLabel><Select labelId="mobile-training-site-label" label="训练区域" value={(trainingSites.data || []).some((item) => item.siteId === trainingSiteId) ? trainingSiteId : ""} onChange={(event) => handleTrainingSiteSelect(event.target.value)}><MenuItem value="">全部训练数据</MenuItem>{(trainingSites.data || []).map((item) => <MenuItem key={item.key} value={item.siteId}>{item.displayName} ({item.patchCount})</MenuItem>)}</Select></FormControl>
          </Box>
          <PatchReviewView workspaceId={activeWorkspaceId} scope={trainingScope} siteId={trainingSiteId} sourceSampleId={sourceSampleId} onClearSource={() => navigate(`${workspacePrefix}/regions/${trainingScope}/training-data${trainingSiteId ? `/${encodeURIComponent(trainingSiteId)}` : ""}`, { replace: true })} />
        </> : !selectedSiteId ? <Box sx={{ display: "grid", placeItems: "center" }}><Typography>选择一个观测区域</Typography></Box> : site.isLoading ? <Box sx={{ display: "grid", placeItems: "center" }}><CircularProgress size={28} /></Box> : site.data ? <>
          <SiteMapToolbar
            leading={<IconButton sx={{ display: { md: "none" } }} onClick={() => navigate(`${workspacePrefix}/regions/${region}/sites`)} aria-label="返回观测区域列表"><ArrowLeft size={19} /></IconButton>}
            title={site.data.display_name || site.data.site_id}
            subtitle={`影像覆盖 ${site.data.coverage_area_km2.toFixed(2)} km² · ${site.data.image_count || 0} 期影像`}
            imageryControl={<ImageryPanel region={selectedRegion} siteId={selectedSiteId} compact onSelectionChange={handleImagerySelection} />}
            basemap={basemap}
            onBasemapChange={setBasemap}
            visibility={layerVisibility}
            onVisibilityChange={handleLayerVisibility}
            jrcThreshold={jrcThreshold}
            onJrcThresholdChange={setJrcThreshold}
            trainingAction={!patchReviewEnabled ? <TrainingCaptureButton controller={trainingCapture} /> : undefined}
            showPrediction={Boolean(predictionModel)}
            showPatches={visiblePatches.length > 0}
            patchReviewEnabled={patchReviewEnabled}
            onPatchReviewEnabledChange={(enabled) => { setPatchReviewEnabled(enabled); setPendingPatchIds(new Set()); setActivePatchId(""); }}
            canFitSite={Boolean(tileMeta.data?.site_bounds || site.data.bbox)}
            canFitTile={Boolean(tileMeta.data?.tile_bounds)}
            onFitSite={() => mapRef.current?.fitSite()}
            onFitTile={() => mapRef.current?.fitTile()}
          />
          <SiteMap
            ref={mapRef}
            site={site.data}
            basemap={basemap}
            visibility={layerVisibility}
            tileMeta={tileMeta.data}
            sentinelTiles={sentinelTiles.data}
            osm={osm.data}
            hydrolakes={hydrolakes.data}
            contextOsm={contextWater.data?.sources.osm}
            contextHydro={contextWater.data?.sources.hydrolakes}
            esa={esa.data}
            jrc={jrc.data}
            localLabel={localLabel.data}
            modelPrediction={modelPrediction.data?.prediction}
            patches={visiblePatches}
            patchReviewEnabled={patchReviewEnabled}
            activePatchId={activePatchId}
            pendingPatchIds={pendingPatchIds}
            onPatchClick={handlePatchClick}
            patchSourceMeta={patchSource.data}
          />
          <Box sx={{ maxHeight: "38vh", overflow: "auto" }}>
            {patchReviewEnabled ? <SitePatchReviewPanel groups={patchGroups} groupKey={activePatchGroup?.key || ""} onGroupChange={(value) => { setPatchGroupKey(value); setPendingPatchIds(new Set()); setActivePatchId(""); }} operation={patchOperation} onOperationChange={(value) => { setPatchOperation(value); setPendingPatchIds(new Set()); }} active={activePatch} pendingCount={pendingPatchIds.size} applying={updatePatches.isPending} onClear={() => setPendingPatchIds(new Set())} onApply={() => updatePatches.mutate({ operation: patchOperation, ids: [...pendingPatchIds] }, { onSuccess: () => { setPendingPatchIds(new Set()); setActivePatchId(""); } })} /> : <>
              <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                <TrainingCaptureStatus controller={trainingCapture} />
                {modelPrediction.isFetching && layerVisibility.prediction && <Typography variant="caption" color="text.secondary">模型预测加载中 · {predictionModel?.name || ""}</Typography>}
                {modelPrediction.isError && layerVisibility.prediction && <Typography variant="caption" color="error">模型预测失败：{modelPrediction.error.message}</Typography>}
                {predictionModel && layerVisibility.prediction && modelPrediction.data && <Typography variant="caption" color="text.secondary">模型 {predictionModel.name} · 阈值 0.50 · 水体像元 {((modelPrediction.data.stats.predicted_ratio || 0) * 100).toFixed(1)}%</Typography>}
              </Box>
            </>}
          </Box>
        </> : <Box sx={{ display: "grid", placeItems: "center" }}><Typography color="error">观测区域加载失败</Typography></Box>}
      </Box>
    </Box>
  );
}

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    if (!query.has("__cf_access_message")) return;
    query.delete("__cf_access_message");
    const search = query.toString();
    const cleanUrl = `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`;
    window.history.replaceState(window.history.state, "", cleanUrl);
  }, []);
  const route = location.pathname.match(/^\/users\/([^/]+)\/workspaces\/([^/]+)(\/regions\/.*)?$/);
  const userId = route?.[1] || "";
  const workspaceId = route?.[2] || "";
  const workspacePath = route?.[3] || "";
  const validWorkspacePath = !workspacePath || /^\/regions\/[^/]+\/(?:sites(?:\/[^/]+)?|training-data(?:\/[^/]+)?|models\/(?:train|validate)(?:\/[^/]+)?)$/.test(workspacePath);
  const session = useAuthSession();
  const workspace = useWorkspace(workspaceId);
  if (session.isLoading) return <AuthLoading />;
  if (session.isError) return <AuthRequired message={session.error.message} />;
  const authenticatedUser = session.data!.user;
  if (!userId && location.pathname === "/") {
    return <Navigate replace to={`/users/${encodeURIComponent(authenticatedUser.id)}/workspaces/${encodeURIComponent(authenticatedUser.default_workspace_id)}`} />;
  }
  if (!userId || !workspaceId || !validWorkspacePath) return <UserRouteError reason="route" onBack={() => navigate("/")} />;
  if (workspace.isLoading) return <AuthLoading />;
  if (workspace.isError || authenticatedUser.id !== userId || authenticatedUser.default_workspace_id !== workspaceId) return <UserRouteError reason="missing" onBack={() => navigate("/")} />;
  if (authenticatedUser.status === "archived" || workspace.data?.workspace.status === "archived") return <UserRouteError reason="archived" onBack={() => navigate("/")} />;
  return <Workbench user={{ ...authenticatedUser, workspace: workspace.data!.workspace }} logoutUrl={session.data!.logout_url} />;
}
