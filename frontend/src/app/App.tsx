import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Button, Chip, CircularProgress, FormControl, IconButton, InputLabel, List, ListItemButton, ListItemIcon, ListItemText, MenuItem, Select, Tab, Tabs, TextField, Typography } from "@mui/material";
import { ArrowLeft, BrainCircuit, Database, Images, Search } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useContextWater, useEsaLayer, useHydrolakesLayer, useJrcLayer, useOsmLayer, useSite, useSites, useLocalLabel, useLocalLabels, useTileMeta } from "../features/sites/api";
import { SiteFilterControls } from "../features/sites/SiteFilterControls";
import { useRegions } from "../features/regions/api";
import { SiteMap, type SiteMapHandle } from "../features/map/SiteMap";
import { ImageryPanel } from "../features/imagery/ImageryPanel";
import { useSentinelTiles } from "../features/imagery/api";
import { PatchReviewView } from "../features/patches/PatchReviewView";
import { TrainingCapture } from "../features/training-samples/TrainingCapture";
import { TrainingSamplesView } from "../features/training-samples/TrainingSamplesView";
import { TrainingView } from "../features/training/TrainingView";
import { ModelValidationView } from "../features/model-validation/ModelValidationView";
import type { SiteFilters, TrainingPatch, TrainingSample } from "../api/types";
import { useWorkbenchStore } from "./store";

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const initialSearch = useMemo(() => new URLSearchParams(location.search), []);
  const [query, setQuery] = useState(() => initialSearch.get("q") || "");
  const [filters, setFilters] = useState<SiteFilters>(() => ({
    area_bucket: initialSearch.get("area_bucket") || "",
    has_name: initialSearch.get("has_name") || "",
    has_tci: initialSearch.get("has_tci") || "",
    has_osm: initialSearch.get("has_osm") || "",
    has_hydrolakes: initialSearch.get("has_hydrolakes") || "",
    has_local_labels: initialSearch.get("has_local_labels") || "",
  }));
  const [jrcThreshold, setJrcThreshold] = useState(75);
  const [selectedLocalLabel, setSelectedLocalLabel] = useState("");
  const [imagerySelection, setImagerySelection] = useState({ tile: "", product: "" });
  const mapRef = useRef<SiteMapHandle>(null);
  const region = useWorkbenchStore((state) => state.region);
  const setRegion = useWorkbenchStore((state) => state.setRegion);
  const siteRoute = location.pathname.match(/^\/regions\/([^/]+)\/sites\/([^/]+)$/);
  const siteListRoute = location.pathname.match(/^\/regions\/([^/]+)\/sites$/);
  const trainingRoute = location.pathname.match(/^\/regions\/([^/]+)\/training\/(samples|patches|train)$/);
  const modelRoute = location.pathname.match(/^\/regions\/([^/]+)\/model(?:\/([^/]+))?$/);
  const selectedRegion = siteRoute?.[1] || "";
  const selectedSiteId = siteRoute?.[2] || "";
  const trainingView = trainingRoute?.[2] || "";
  const isTraining = Boolean(trainingRoute);
  const isModelValidation = Boolean(modelRoute);
  const modelSiteId = modelRoute?.[2] || "";
  const routeParams = new URLSearchParams(location.search);
  const routeModel = routeParams.get("model") || "";
  const modelSiteRegion = routeParams.get("site_region") || (modelRoute?.[1] === "all" ? "" : modelRoute?.[1] || "");
  const isSiteWorkspace = !isTraining && !isModelValidation;
  const regions = useRegions();
  const sites = useSites(region, query, filters);
  const siteItems = sites.data?.pages.flatMap((page) => page.items) || [];
  const siteTotal = sites.data?.pages[0]?.total;
  const site = useSite(selectedRegion, selectedSiteId);
  const tileMeta = useTileMeta(selectedRegion, selectedSiteId);
  const sentinelTiles = useSentinelTiles(selectedRegion, selectedSiteId);
  const contextWater = useContextWater(selectedRegion, selectedSiteId);
  const osm = useOsmLayer(selectedRegion, selectedSiteId);
  const hydrolakes = useHydrolakesLayer(selectedRegion, selectedSiteId);
  const esa = useEsaLayer(selectedRegion, selectedSiteId);
  const jrc = useJrcLayer(selectedRegion, selectedSiteId, jrcThreshold);
  const localLabels = useLocalLabels(selectedRegion, selectedSiteId);
  const localLabel = useLocalLabel(selectedRegion, selectedSiteId, selectedLocalLabel);

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
  const siteSearch = useMemo(() => {
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    Object.entries(filters).forEach(([key, value]) => { if (value) params.set(key, value); });
    const value = params.toString();
    return value ? `?${value}` : "";
  }, [filters, query]);

  useEffect(() => {
    if (region && location.pathname === "/") navigate(`/regions/${region}/sites${siteSearch}`, { replace: true });
  }, [siteSearch, location.pathname, navigate, region]);

  useEffect(() => {
    if (!isSiteWorkspace || location.search === siteSearch) return;
    navigate({ pathname: location.pathname, search: siteSearch }, { replace: true });
  }, [isSiteWorkspace, siteSearch, location.pathname, location.search, navigate]);
  const handleImagerySelection = useCallback((selection: { tile: string; product: string }) => setImagerySelection((current) => current.tile === selection.tile && current.product === selection.product ? current : selection), []);
  const navigateToSite = useCallback((item: TrainingSample | TrainingPatch) => {
    navigate(`/regions/${item.region || region}/sites/${item.site_id}${siteSearch}`);
  }, [siteSearch, navigate, region]);

  const handleRegionChange = (nextRegion: string) => {
    setRegion(nextRegion);
    navigate(isTraining ? `/regions/${nextRegion}/training/${trainingView || "samples"}` : isModelValidation ? `/regions/${nextRegion}/model` : `/regions/${nextRegion}/sites${siteSearch}`);
  };

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "360px minmax(0, 1fr)" }, height: "100vh", bgcolor: "background.default" }}>
      <Box component="aside" sx={{ display: { xs: selectedSiteId || isTraining || isModelValidation ? "none" : "block", md: "block" }, borderRight: 1, borderColor: "divider", bgcolor: "background.paper", overflow: "auto" }}>
        <Box sx={{ p: 2, display: "grid", gap: 1.5, position: "sticky", top: 0, bgcolor: "background.paper", zIndex: 1 }}>
          <Typography variant="h6">Lakes Workbench</Typography>
          <FormControl fullWidth>
            <InputLabel id="region-label">区域</InputLabel>
            <Select labelId="region-label" label="区域" value={region} onChange={(event) => handleRegionChange(event.target.value)}>
              <MenuItem value="all">全部区域</MenuItem>
              {(regions.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key}>{item.name} ({item.site_count})</MenuItem>)}
            </Select>
          </FormControl>
          <List disablePadding sx={{ mx: -1 }}>
            <ListItemButton selected={isSiteWorkspace} onClick={() => navigate(`/regions/${region}/sites${siteSearch}`)}><ListItemIcon sx={{ minWidth: 34 }}><Images size={18} /></ListItemIcon><ListItemText primary="观测区域" /></ListItemButton>
            <ListItemButton selected={isTraining} onClick={() => navigate(`/regions/${region}/training/samples`)}><ListItemIcon sx={{ minWidth: 34 }}><Database size={18} /></ListItemIcon><ListItemText primary="训练集" /></ListItemButton>
            <ListItemButton selected={isModelValidation} onClick={() => navigate(`/regions/${region}/model`)}><ListItemIcon sx={{ minWidth: 34 }}><BrainCircuit size={18} /></ListItemIcon><ListItemText primary="模型验证" /></ListItemButton>
          </List>
          {isSiteWorkspace && <>
            <TextField fullWidth placeholder="区域 ID / 名称提示" value={query} onChange={(event) => setQuery(event.target.value)} slotProps={{ input: { startAdornment: <Search size={17} /> } }} />
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}><SiteFilterControls value={filters} onChange={setFilters} /></Box>
            <Typography variant="caption" color="text.secondary">{siteTotal != null ? `${siteTotal} 个观测区域，显示 ${siteItems.length} 个` : "加载中"}</Typography>
          </>}
        </Box>
        {isSiteWorkspace && (sites.isLoading ? <Box sx={{ p: 3, textAlign: "center" }}><CircularProgress size={24} /></Box> : sites.isError ? <Typography color="error" sx={{ p: 2 }}>{sites.error.message}</Typography> : (
          <List disablePadding>
            {!siteItems.length && <Box sx={{ p: 3, textAlign: "center" }}><Typography variant="body2">没有符合条件的观测区域</Typography></Box>}
            {siteItems.map((site, index) => (
              <ListItemButton key={`${site.region || region}:${site.site_id}:${index}`} divider selected={site.site_id === selectedSiteId} onClick={() => navigate(`/regions/${site.region || region}/sites/${site.site_id}${siteSearch}`)}>
                <ListItemText primary={<Box sx={{ display: "flex", alignItems: "center", gap: .75 }}><Typography variant="body2" noWrap sx={{ flex: 1 }}>{site.display_name || site.site_id}</Typography>{region === "all" && <Chip size="small" label={site.region_name || site.region} />}</Box>} secondary={`覆盖 ${site.coverage_area_km2.toFixed(2)} km² · ${(site.tiles || []).join(", ")}${site.has_tci ? " · 影像" : ""}`} />
              </ListItemButton>
            ))}
            {sites.hasNextPage && <Box sx={{ p: 1.5 }}><Button fullWidth variant="outlined" disabled={sites.isFetchingNextPage} onClick={() => sites.fetchNextPage()}>{sites.isFetchingNextPage ? "加载中" : "加载更多"}</Button></Box>}
          </List>
        ))}
      </Box>
      <Box component="main" sx={{ minWidth: 0, display: { xs: selectedSiteId || isTraining || isModelValidation ? "grid" : "none", md: "grid" }, gridTemplateRows: selectedSiteId ? "auto minmax(0, 1fr) auto" : isTraining ? "auto minmax(0, 1fr)" : "1fr", color: "text.secondary" }}>
        {isModelValidation ? <ModelValidationView scope={region} routeSiteId={modelSiteId} routeSiteRegion={modelSiteRegion} routeModel={routeModel} onBack={() => navigate(`/regions/${region}/sites${siteSearch}`)} onRouteModel={(model) => navigate(`/regions/${region}/model?model=${encodeURIComponent(model)}`, { replace: true })} onRouteResult={(siteId, siteRegion, model) => navigate(`/regions/${region}/model/${encodeURIComponent(siteId)}?model=${encodeURIComponent(model)}${region === "all" ? `&site_region=${encodeURIComponent(siteRegion)}` : ""}`)} /> : isTraining ? <>
          <Box sx={{ borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", alignItems: "center", px: { xs: 1, sm: 2 } }}>
            <IconButton sx={{ display: { md: "none" }, mr: .5 }} onClick={() => navigate(`/regions/${region}/sites`)}><ArrowLeft size={19} /></IconButton>
            <Tabs value={trainingView} onChange={(_, value) => navigate(`/regions/${region}/training/${value}`)}>
              <Tab value="samples" label="样本" />
              <Tab value="patches" label="Patch 审核" />
              <Tab value="train" label="训练" />
            </Tabs>
            <Button sx={{ ml: "auto", display: { xs: "none", sm: "inline-flex" } }} startIcon={<Images size={16} />} onClick={() => navigate(`/regions/${region}/sites`)}>浏览区域</Button>
          </Box>
          {trainingView === "patches" ? <PatchReviewView scope={region} onLocate={navigateToSite} /> : trainingView === "train" ? <TrainingView scope={region} /> : <TrainingSamplesView scope={region} onLocate={navigateToSite} />}
        </> : !selectedSiteId ? <Box sx={{ display: "grid", placeItems: "center" }}><Typography>选择一个观测区域</Typography></Box> : site.isLoading ? <Box sx={{ display: "grid", placeItems: "center" }}><CircularProgress size={28} /></Box> : site.data ? <>
          <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", alignItems: "center", gap: 1 }}>
            <IconButton sx={{ display: { md: "none" } }} onClick={() => navigate(`/regions/${region}/sites${siteSearch}`)}><ArrowLeft size={19} /></IconButton>
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
          />
          <Box sx={{ maxHeight: "38vh", overflow: "auto" }}>
            <ImageryPanel region={selectedRegion} siteId={selectedSiteId} onSelectionChange={handleImagerySelection} />
            <TrainingCapture region={selectedRegion} siteId={selectedSiteId} jrcThreshold={jrcThreshold} localLabel={(localLabels.data || []).find((item) => item.id === selectedLocalLabel)} imagery={imagerySelection} mapHandle={mapRef} />
          </Box>
        </> : <Box sx={{ display: "grid", placeItems: "center" }}><Typography color="error">观测区域加载失败</Typography></Box>}
      </Box>
    </Box>
  );
}
