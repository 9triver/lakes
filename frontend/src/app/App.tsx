import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Button, Chip, CircularProgress, FormControl, IconButton, InputLabel, List, ListItemButton, ListItemIcon, ListItemText, MenuItem, Select, Tab, Tabs, TextField, Typography } from "@mui/material";
import { ArrowLeft, BrainCircuit, Database, Images, Search } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useContextWater, useEsaLayer, useHydrolakesLayer, useJrcLayer, useOsmLayer, useSite, useSites, useLocalLabel, useLocalLabels, useTileMeta } from "../features/sites/api";
import { useRegions } from "../features/regions/api";
import { SiteMap, type SiteMapHandle } from "../features/map/SiteMap";
import { ImageryPanel } from "../features/imagery/ImageryPanel";
import { useSentinelTiles } from "../features/imagery/api";
import { PatchReviewView } from "../features/patches/PatchReviewView";
import { SitePatchReviewPanel, type PatchGroup, type PatchOperation } from "../features/patches/SitePatchReviewPanel";
import { useBatchUpdateLogicalPatches, useLogicalPatchSourceMeta, useSiteLogicalPatches } from "../features/patches/api";
import { TrainingCapture } from "../features/training-samples/TrainingCapture";
import { TrainingSamplesView } from "../features/training-samples/TrainingSamplesView";
import { TrainingView } from "../features/training/TrainingView";
import { ModelValidationView } from "../features/model-validation/ModelValidationView";
import { CurrentProfileBar } from "../features/profiles/ProfileSelector";
import { ProfileHome, ProfileRouteError } from "../features/profiles/ProfileHome";
import { SourceConflictsView } from "../features/profiles/SourceConflictsView";
import { useProfile, useProfiles } from "../features/profiles/api";
import type { TrainingPatch, TrainingSample } from "../api/types";
import { useWorkbenchStore } from "./store";

function Workbench() {
  const location = useLocation();
  const navigate = useNavigate();
  const initialSearch = useMemo(() => new URLSearchParams(location.search), []);
  const [query, setQuery] = useState(() => initialSearch.get("q") || "");
  const [jrcThreshold, setJrcThreshold] = useState(75);
  const [selectedLocalLabel, setSelectedLocalLabel] = useState("");
  const [imagerySelection, setImagerySelection] = useState({ tile: "", product: "" });
  const [patchReviewEnabled, setPatchReviewEnabled] = useState(false);
  const [patchGroupKey, setPatchGroupKey] = useState("");
  const [patchOperation, setPatchOperation] = useState<PatchOperation>("exclude");
  const [pendingPatchIds, setPendingPatchIds] = useState<Set<string>>(() => new Set());
  const [activePatchId, setActivePatchId] = useState("");
  const mapRef = useRef<SiteMapHandle>(null);
  const region = useWorkbenchStore((state) => state.region);
  const setRegion = useWorkbenchStore((state) => state.setRegion);
  const profileRoute = location.pathname.match(/^\/profiles\/([^/]+)(\/regions\/.*)?$/);
  const profileId = profileRoute?.[1] || "";
  const workspacePath = profileRoute?.[2] || "";
  const profileList = useProfiles();
  const activeProfileId = profileId;
  const activeProfile = profileList.data?.items.find((item) => item.id === activeProfileId);
  const profilePrefix = `/profiles/${activeProfileId}`;
  const siteRoute = workspacePath.match(/^\/regions\/([^/]+)\/sites\/([^/]+)$/);
  const siteListRoute = workspacePath.match(/^\/regions\/([^/]+)\/sites$/);
  const trainingRoute = workspacePath.match(/^\/regions\/([^/]+)\/training\/(samples|patches|sources|train)$/);
  const modelRoute = workspacePath.match(/^\/regions\/([^/]+)\/model(?:\/([^/]+))?$/);
  const selectedRegion = siteRoute?.[1] || "";
  const selectedSiteId = siteRoute?.[2] || "";
  const trainingView = trainingRoute?.[2] || "";
  const trainingScope = trainingRoute?.[1] || region;
  const isTraining = Boolean(trainingRoute);
  const isModelValidation = Boolean(modelRoute);
  const modelSiteId = modelRoute?.[2] || "";
  const routeParams = new URLSearchParams(location.search);
  const routeModel = routeParams.get("model") || "";
  const modelSiteRegion = routeParams.get("site_region") || (modelRoute?.[1] === "all" ? "" : modelRoute?.[1] || "");
  const isSiteWorkspace = !isTraining && !isModelValidation;
  const regions = useRegions();
  const sites = useSites(activeProfileId, region, query);
  const siteItems = sites.data?.pages.flatMap((page) => page.items) || [];
  const siteTotal = sites.data?.pages[0]?.total;
  const site = useSite(selectedRegion, selectedSiteId, activeProfileId);
  const tileMeta = useTileMeta(selectedRegion, selectedSiteId);
  const sentinelTiles = useSentinelTiles(selectedRegion, selectedSiteId);
  const contextWater = useContextWater(selectedRegion, selectedSiteId);
  const osm = useOsmLayer(selectedRegion, selectedSiteId);
  const hydrolakes = useHydrolakesLayer(selectedRegion, selectedSiteId);
  const esa = useEsaLayer(selectedRegion, selectedSiteId);
  const jrc = useJrcLayer(selectedRegion, selectedSiteId, jrcThreshold);
  const localLabels = useLocalLabels(selectedRegion, selectedSiteId);
  const localLabel = useLocalLabel(selectedRegion, selectedSiteId, selectedLocalLabel);
  const logicalPatches = useSiteLogicalPatches(activeProfileId, selectedRegion, selectedSiteId);
  const updateLogicalPatches = useBatchUpdateLogicalPatches(activeProfileId, selectedRegion, selectedSiteId);
  const patchGroups = useMemo<PatchGroup[]>(() => {
    const groups = new Map<string, TrainingPatch[]>();
    for (const patch of logicalPatches.data?.items || []) {
      const key = `${patch.sample_id}:${patch.image_index || 0}`;
      groups.set(key, [...(groups.get(key) || []), patch]);
    }
    return [...groups.entries()].map(([key, patches]) => ({ key, patches, label: `${patches[0]?.product_date || patches[0]?.product_name || patches[0]?.sample_id} · ${patches.length} 个` })).sort((a, b) => b.label.localeCompare(a.label));
  }, [logicalPatches.data?.items]);
  const activePatchGroup = patchGroups.find((group) => group.key === patchGroupKey) || patchGroups[0];
  const visibleLogicalPatches = activePatchGroup?.patches || logicalPatches.data?.items || [];
  const activePatch = visibleLogicalPatches.find((patch) => (patch.logical_patch_id || patch.patch_id) === activePatchId);
  const patchSource = useLogicalPatchSourceMeta(activeProfileId, selectedRegion, selectedSiteId, visibleLogicalPatches[0]?.logical_patch_id || visibleLogicalPatches[0]?.patch_id || "", patchReviewEnabled);

  useEffect(() => {
    if (!region && regions.data) setRegion(siteListRoute?.[1] || siteRoute?.[1] || trainingRoute?.[1] || modelRoute?.[1] || regions.data.default);
  }, [siteListRoute, siteRoute, modelRoute, region, regions.data, setRegion, trainingRoute]);

  useEffect(() => {
    const workspaceScope = siteListRoute?.[1] || trainingRoute?.[1] || modelRoute?.[1];
    if (workspaceScope && region && workspaceScope !== region) setRegion(workspaceScope);
  }, [siteListRoute, modelRoute, region, setRegion, trainingRoute]);

  useEffect(() => {
    const items = localLabels.data || [];
    if (!items.some((item) => item.id === selectedLocalLabel)) setSelectedLocalLabel(items[0]?.id || "");
  }, [localLabels.data, selectedSiteId, selectedLocalLabel]);
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
  const siteSearch = useMemo(() => {
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    const value = params.toString();
    return value ? `?${value}` : "";
  }, [query]);

  useEffect(() => {
    if (!isSiteWorkspace || location.search === siteSearch) return;
    navigate({ pathname: location.pathname, search: siteSearch }, { replace: true });
  }, [isSiteWorkspace, siteSearch, location.pathname, location.search, navigate]);
  const handleImagerySelection = useCallback((selection: { tile: string; product: string }) => setImagerySelection((current) => current.tile === selection.tile && current.product === selection.product ? current : selection), []);
  const handleLogicalPatchClick = useCallback((patchId: string) => {
    setActivePatchId(patchId);
    const patch = visibleLogicalPatches.find((item) => (item.logical_patch_id || item.patch_id) === patchId);
    const selectable = patchOperation === "exclude" ? patch?.included : patch && !patch.included;
    if (!selectable) return;
    setPendingPatchIds((current) => {
      const next = new Set(current);
      if (next.has(patchId)) next.delete(patchId); else next.add(patchId);
      return next;
    });
  }, [patchOperation, visibleLogicalPatches]);
  const navigateToSite = useCallback((item: TrainingSample | TrainingPatch) => {
    navigate(`${profilePrefix}/regions/${item.region || region}/sites/${item.site_id}${siteSearch}`);
  }, [siteSearch, navigate, profilePrefix, region]);

  const handleRegionChange = (nextRegion: string) => {
    setRegion(nextRegion);
    navigate(isTraining ? `${profilePrefix}/regions/${nextRegion}/training/${trainingView || "samples"}` : isModelValidation ? `${profilePrefix}/regions/${nextRegion}/model` : `${profilePrefix}/regions/${nextRegion}/sites${siteSearch}`);
  };

  useEffect(() => {
    if (profileId && !profileRoute?.[2] && profileList.data && regions.data) {
      navigate(`${profilePrefix}/regions/${regions.data.default}/sites`, { replace: true });
    }
  }, [navigate, profileId, profileList.data, profilePrefix, profileRoute, regions.data]);

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "360px minmax(0, 1fr)" }, height: "100vh", bgcolor: "background.default" }}>
      <Box component="aside" sx={{ display: { xs: selectedSiteId || isTraining || isModelValidation ? "none" : "block", md: "block" }, borderRight: 1, borderColor: "divider", bgcolor: "background.paper", overflow: "auto" }}>
        <Box sx={{ p: 2, display: "grid", gap: 1.5, position: "sticky", top: 0, bgcolor: "background.paper", zIndex: 1 }}>
          <Typography variant="h6">Lakes Workbench</Typography>
          <CurrentProfileBar profile={activeProfile} onSwitch={() => navigate("/")} />
          <FormControl fullWidth>
            <InputLabel id="region-label">区域</InputLabel>
            <Select labelId="region-label" label="区域" value={region} onChange={(event) => handleRegionChange(event.target.value)}>
              <MenuItem value="all">全部区域</MenuItem>
              {(regions.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key}>{item.name} ({item.site_count})</MenuItem>)}
            </Select>
          </FormControl>
          <List disablePadding sx={{ mx: -1 }}>
            <ListItemButton selected={isSiteWorkspace} onClick={() => navigate(`${profilePrefix}/regions/${region}/sites${siteSearch}`)}><ListItemIcon sx={{ minWidth: 34 }}><Images size={18} /></ListItemIcon><ListItemText primary="观测区域" /></ListItemButton>
            <ListItemButton selected={isTraining} onClick={() => navigate(`${profilePrefix}/regions/${region}/training/samples`)}><ListItemIcon sx={{ minWidth: 34 }}><Database size={18} /></ListItemIcon><ListItemText primary="训练集" /></ListItemButton>
            <ListItemButton selected={isModelValidation} onClick={() => navigate(`${profilePrefix}/regions/${region}/model`)}><ListItemIcon sx={{ minWidth: 34 }}><BrainCircuit size={18} /></ListItemIcon><ListItemText primary="模型验证" /></ListItemButton>
          </List>
          {isSiteWorkspace && <>
            <TextField fullWidth placeholder="区域 ID / 名称提示" value={query} onChange={(event) => setQuery(event.target.value)} slotProps={{ input: { startAdornment: <Search size={17} /> } }} />
            <Typography variant="caption" color="text.secondary">{siteTotal != null ? `${siteTotal} 个观测区域，显示 ${siteItems.length} 个` : "加载中"}</Typography>
          </>}
        </Box>
        {isSiteWorkspace && (sites.isLoading ? <Box sx={{ p: 3, textAlign: "center" }}><CircularProgress size={24} /></Box> : sites.isError ? <Typography color="error" sx={{ p: 2 }}>{sites.error.message}</Typography> : (
          <List disablePadding>
            {!siteItems.length && <Box sx={{ p: 3, textAlign: "center" }}><Typography variant="body2">没有符合条件的观测区域</Typography></Box>}
            {siteItems.map((site, index) => (
              <ListItemButton key={`${site.region || region}:${site.site_id}:${index}`} divider selected={site.site_id === selectedSiteId} onClick={() => navigate(`${profilePrefix}/regions/${site.region || region}/sites/${site.site_id}${siteSearch}`)}>
                <ListItemText primary={<Box sx={{ display: "flex", alignItems: "center", gap: .75 }}><Typography variant="body2" noWrap sx={{ flex: 1 }}>{site.display_name || site.site_id}</Typography>{Boolean(site.included_logical_patch_count) && <Chip size="small" color="success" variant="outlined" label={`逻辑 ${site.included_logical_patch_count}`} />}{region === "all" && <Chip size="small" label={site.region_name || site.region} />}</Box>} secondary={`覆盖 ${site.coverage_area_km2.toFixed(2)} km² · ${(site.tiles || []).join(", ")}${site.has_tci ? " · 影像" : ""}`} />
              </ListItemButton>
            ))}
            {sites.hasNextPage && <Box sx={{ p: 1.5 }}><Button fullWidth variant="outlined" disabled={sites.isFetchingNextPage} onClick={() => sites.fetchNextPage()}>{sites.isFetchingNextPage ? "加载中" : "加载更多"}</Button></Box>}
          </List>
        ))}
      </Box>
      <Box component="main" sx={{ minWidth: 0, display: { xs: selectedSiteId || isTraining || isModelValidation ? "grid" : "none", md: "grid" }, gridTemplateRows: { xs: selectedSiteId ? "auto auto minmax(0, 1fr) auto" : isTraining ? "auto auto minmax(0, 1fr)" : isModelValidation ? "auto minmax(0, 1fr)" : "1fr", md: selectedSiteId ? "auto minmax(0, 1fr) auto" : isTraining ? "auto minmax(0, 1fr)" : "1fr" }, color: "text.secondary" }}>
        <Box sx={{ display: { xs: "block", md: "none" }, px: 1.5, py: .75, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}><CurrentProfileBar compact profile={activeProfile} onSwitch={() => navigate("/")} /></Box>
        {isModelValidation ? <ModelValidationView profileId={activeProfileId} scope={region} routeSiteId={modelSiteId} routeSiteRegion={modelSiteRegion} routeModel={routeModel} onBack={() => navigate(`${profilePrefix}/regions/${region}/sites${siteSearch}`)} onRouteModel={(model) => navigate(`${profilePrefix}/regions/${region}/model?model=${encodeURIComponent(model)}`, { replace: true })} onRouteResult={(siteId, siteRegion, model) => navigate(`${profilePrefix}/regions/${region}/model/${encodeURIComponent(siteId)}?model=${encodeURIComponent(model)}${region === "all" ? `&site_region=${encodeURIComponent(siteRegion)}` : ""}`)} /> : isTraining ? <>
          <Box sx={{ borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", alignItems: "center", px: { xs: 1, sm: 2 }, minWidth: 0, overflow: "hidden" }}>
            <IconButton sx={{ display: { md: "none" }, mr: .5 }} onClick={() => navigate(`${profilePrefix}/regions/${trainingScope}/sites`)}><ArrowLeft size={19} /></IconButton>
            <Tabs variant="scrollable" scrollButtons="auto" value={trainingView} onChange={(_, value) => navigate(`${profilePrefix}/regions/${trainingScope}/training/${value}`)} sx={{ minWidth: 0, flex: 1 }}>
              <Tab value="samples" label="样本" />
              <Tab value="patches" label="逻辑 Patch" />
              {(activeProfile?.conflict_count || trainingView === "sources") && <Tab value="sources" label={`来源冲突${activeProfile?.conflict_count ? ` (${activeProfile.conflict_count})` : ""}`} />}
              <Tab value="train" label="训练" />
            </Tabs>
            <Button sx={{ ml: "auto", display: { xs: "none", sm: "inline-flex" } }} startIcon={<Images size={16} />} onClick={() => navigate(`${profilePrefix}/regions/${trainingScope}/sites`)}>浏览区域</Button>
          </Box>
          {trainingView === "patches" ? <PatchReviewView profileId={activeProfileId} scope={trainingScope} onLocate={navigateToSite} /> : trainingView === "sources" ? <SourceConflictsView profileId={activeProfileId} /> : trainingView === "train" ? <TrainingView profileId={activeProfileId} scope={trainingScope} blocked={activeProfile?.status === "needs_resolution"} defaults={activeProfile?.training_defaults} /> : <TrainingSamplesView scope={trainingScope} onLocate={navigateToSite} />}
        </> : !selectedSiteId ? <Box sx={{ display: "grid", placeItems: "center" }}><Typography>选择一个观测区域</Typography></Box> : site.isLoading ? <Box sx={{ display: "grid", placeItems: "center" }}><CircularProgress size={28} /></Box> : site.data ? <>
          <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", alignItems: "center", gap: 1 }}>
            <IconButton sx={{ display: { md: "none" } }} onClick={() => navigate(`${profilePrefix}/regions/${region}/sites${siteSearch}`)}><ArrowLeft size={19} /></IconButton>
            <Box><Typography variant="subtitle1" color="text.primary">{site.data.display_name || site.data.site_id}</Typography><Typography variant="caption">影像覆盖 {site.data.coverage_area_km2.toFixed(2)} km² · {site.data.image_count || 0} 期影像</Typography></Box>
          </Box>
          <SiteMap
            ref={mapRef}
            site={site.data}
            tileMeta={tileMeta.data}
            sentinelTiles={sentinelTiles.data}
            osm={osm.data}
            hydrolakes={hydrolakes.data}
            contextOsm={contextWater.data?.sources.osm}
            contextHydro={contextWater.data?.sources.hydrolakes}
            esa={esa.data}
            jrc={jrc.data}
            localLabel={localLabel.data}
            localLabels={localLabels.data || []}
            selectedLocalLabel={selectedLocalLabel}
            onLocalLabelChange={setSelectedLocalLabel}
            jrcThreshold={jrcThreshold}
            onJrcThresholdChange={setJrcThreshold}
            logicalPatches={visibleLogicalPatches}
            patchReviewEnabled={patchReviewEnabled}
            onPatchReviewEnabledChange={(enabled) => { setPatchReviewEnabled(enabled); setPendingPatchIds(new Set()); setActivePatchId(""); }}
            activePatchId={activePatchId}
            pendingPatchIds={pendingPatchIds}
            onLogicalPatchClick={handleLogicalPatchClick}
            patchSourceMeta={patchSource.data}
          />
          <Box sx={{ maxHeight: "38vh", overflow: "auto" }}>
            {patchReviewEnabled ? <SitePatchReviewPanel groups={patchGroups} groupKey={activePatchGroup?.key || ""} onGroupChange={(value) => { setPatchGroupKey(value); setPendingPatchIds(new Set()); setActivePatchId(""); }} operation={patchOperation} onOperationChange={(value) => { setPatchOperation(value); setPendingPatchIds(new Set()); }} active={activePatch} pendingCount={pendingPatchIds.size} applying={updateLogicalPatches.isPending} onClear={() => setPendingPatchIds(new Set())} onApply={() => updateLogicalPatches.mutate({ operation: patchOperation, ids: [...pendingPatchIds] }, { onSuccess: () => { setPendingPatchIds(new Set()); setActivePatchId(""); } })} /> : <>
              <ImageryPanel region={selectedRegion} siteId={selectedSiteId} onSelectionChange={handleImagerySelection} />
              <TrainingCapture profileId={activeProfileId} region={selectedRegion} siteId={selectedSiteId} jrcThreshold={jrcThreshold} localLabel={(localLabels.data || []).find((item) => item.id === selectedLocalLabel)} imagery={imagerySelection} mapHandle={mapRef} />
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
  const profileMatch = location.pathname.match(/^\/profiles\/([^/]+)(\/regions\/.*)?$/);
  const profileId = profileMatch?.[1] || "";
  const workspacePath = profileMatch?.[2] || "";
  const validWorkspacePath = !workspacePath || /^\/regions\/[^/]+\/(?:sites(?:\/[^/]+)?|training\/(?:samples|patches|sources|train)|model(?:\/[^/]+)?)$/.test(workspacePath);
  const profile = useProfile(profileId);
  if (!profileId && location.pathname === "/") return <ProfileHome onSelect={(id) => navigate(`/profiles/${encodeURIComponent(id)}`)} />;
  if (!profileId) return <ProfileRouteError reason="route" onBack={() => navigate("/")} />;
  if (!validWorkspacePath) return <ProfileRouteError reason="route" onBack={() => navigate("/")} />;
  if (profile.isLoading) return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center" }}><CircularProgress /></Box>;
  if (profile.isError) return <ProfileRouteError reason="missing" onBack={() => navigate("/")} />;
  if (profile.data?.profile.status === "archived") return <ProfileRouteError reason="archived" onBack={() => navigate("/")} />;
  return <Workbench />;
}
