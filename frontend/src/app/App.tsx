import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Button, Chip, CircularProgress, FormControl, IconButton, InputLabel, List, ListItemButton, ListItemIcon, ListItemText, MenuItem, Select, Tab, Tabs, TextField, Typography } from "@mui/material";
import { ArrowLeft, BrainCircuit, Database, Images, Search } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useContextWater, useEsaLayer, useJrcLayer, useSite, useSites, useLocalLabel, useLocalLabels, useTileMeta } from "../features/sites/api";
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
  const lakeRoute = location.pathname.match(/^\/regions\/([^/]+)\/(?:sites|lakes)\/([^/]+)$/);
  const lakeListRoute = location.pathname.match(/^\/regions\/([^/]+)\/(?:sites|lakes)$/);
  const trainingRoute = location.pathname.match(/^\/regions\/([^/]+)\/training\/(samples|patches|train)$/);
  const modelRoute = location.pathname.match(/^\/regions\/([^/]+)\/model(?:\/([^/]+))?$/);
  const selectedRegion = lakeRoute?.[1] || "";
  const selectedLakeId = lakeRoute?.[2] || "";
  const trainingView = trainingRoute?.[2] || "";
  const isTraining = Boolean(trainingRoute);
  const isModelValidation = Boolean(modelRoute);
  const modelLakeId = modelRoute?.[2] || "";
  const routeParams = new URLSearchParams(location.search);
  const routeModel = routeParams.get("model") || "";
  const modelLakeRegion = routeParams.get("site_region") || routeParams.get("lake_region") || (modelRoute?.[1] === "all" ? "" : modelRoute?.[1] || "");
  const isLakeWorkspace = !isTraining && !isModelValidation;
  const regions = useRegions();
  const lakes = useSites(region, query, filters);
  const lakeItems = lakes.data?.pages.flatMap((page) => page.items) || [];
  const lakeTotal = lakes.data?.pages[0]?.total;
  const lake = useSite(selectedRegion, selectedLakeId);
  const tileMeta = useTileMeta(selectedRegion, selectedLakeId);
  const sentinelTiles = useSentinelTiles(selectedRegion, selectedLakeId);
  const contextWater = useContextWater(selectedRegion, selectedLakeId);
  const esa = useEsaLayer(selectedRegion, selectedLakeId);
  const jrc = useJrcLayer(selectedRegion, selectedLakeId, jrcThreshold);
  const localLabels = useLocalLabels(selectedRegion, selectedLakeId);
  const localLabel = useLocalLabel(selectedRegion, selectedLakeId, selectedLocalLabel);

  useEffect(() => {
    if (!region && regions.data) setRegion(lakeListRoute?.[1] || lakeRoute?.[1] || trainingRoute?.[1] || modelRoute?.[1] || regions.data.default);
  }, [lakeListRoute, lakeRoute, modelRoute, region, regions.data, setRegion, trainingRoute]);

  useEffect(() => {
    const workspaceScope = lakeListRoute?.[1] || trainingRoute?.[1] || modelRoute?.[1];
    if (workspaceScope && region && workspaceScope !== region) setRegion(workspaceScope);
  }, [lakeListRoute, modelRoute, region, setRegion, trainingRoute]);

  useEffect(() => {
    const items = localLabels.data || [];
    if (!items.some((item) => item.id === selectedLocalLabel)) setSelectedLocalLabel(items[0]?.id || "");
  }, [localLabels.data, selectedLakeId, selectedLocalLabel]);
  const lakeSearch = useMemo(() => {
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    Object.entries(filters).forEach(([key, value]) => { if (value) params.set(key, value); });
    const value = params.toString();
    return value ? `?${value}` : "";
  }, [filters, query]);

  useEffect(() => {
    if (region && location.pathname === "/") navigate(`/regions/${region}/sites${lakeSearch}`, { replace: true });
  }, [lakeSearch, location.pathname, navigate, region]);

  useEffect(() => {
    if (location.pathname.includes("/lakes")) {
      navigate({ pathname: location.pathname.replace("/lakes", "/sites"), search: location.search }, { replace: true });
    }
  }, [location.pathname, location.search, navigate]);

  useEffect(() => {
    if (!isLakeWorkspace || location.search === lakeSearch) return;
    navigate({ pathname: location.pathname, search: lakeSearch }, { replace: true });
  }, [isLakeWorkspace, lakeSearch, location.pathname, location.search, navigate]);
  const handleImagerySelection = useCallback((selection: { tile: string; product: string }) => setImagerySelection((current) => current.tile === selection.tile && current.product === selection.product ? current : selection), []);
  const navigateToLake = useCallback((item: TrainingSample | TrainingPatch) => {
    navigate(`/regions/${item.region || region}/sites/${item.site_id || item.lake_id}${lakeSearch}`);
  }, [lakeSearch, navigate, region]);

  const handleRegionChange = (nextRegion: string) => {
    setRegion(nextRegion);
    navigate(isTraining ? `/regions/${nextRegion}/training/${trainingView || "samples"}` : isModelValidation ? `/regions/${nextRegion}/model` : `/regions/${nextRegion}/sites${lakeSearch}`);
  };

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "360px minmax(0, 1fr)" }, height: "100vh", bgcolor: "background.default" }}>
      <Box component="aside" sx={{ display: { xs: selectedLakeId || isTraining || isModelValidation ? "none" : "block", md: "block" }, borderRight: 1, borderColor: "divider", bgcolor: "background.paper", overflow: "auto" }}>
        <Box sx={{ p: 2, display: "grid", gap: 1.5, position: "sticky", top: 0, bgcolor: "background.paper", zIndex: 1 }}>
          <Typography variant="h6">Lakes Workbench</Typography>
          <FormControl fullWidth>
            <InputLabel id="region-label">区域</InputLabel>
            <Select labelId="region-label" label="区域" value={region} onChange={(event) => handleRegionChange(event.target.value)}>
              <MenuItem value="all">全部区域</MenuItem>
              {(regions.data?.items || []).map((item) => <MenuItem key={item.key} value={item.key}>{item.name} ({item.site_count ?? item.lake_count ?? 0})</MenuItem>)}
            </Select>
          </FormControl>
          <List disablePadding sx={{ mx: -1 }}>
            <ListItemButton selected={isLakeWorkspace} onClick={() => navigate(`/regions/${region}/sites${lakeSearch}`)}><ListItemIcon sx={{ minWidth: 34 }}><Images size={18} /></ListItemIcon><ListItemText primary="观测区域" /></ListItemButton>
            <ListItemButton selected={isTraining} onClick={() => navigate(`/regions/${region}/training/samples`)}><ListItemIcon sx={{ minWidth: 34 }}><Database size={18} /></ListItemIcon><ListItemText primary="训练集" /></ListItemButton>
            <ListItemButton selected={isModelValidation} onClick={() => navigate(`/regions/${region}/model`)}><ListItemIcon sx={{ minWidth: 34 }}><BrainCircuit size={18} /></ListItemIcon><ListItemText primary="模型验证" /></ListItemButton>
          </List>
          {isLakeWorkspace && <>
            <TextField fullWidth placeholder="区域 ID / 名称提示" value={query} onChange={(event) => setQuery(event.target.value)} slotProps={{ input: { startAdornment: <Search size={17} /> } }} />
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}><SiteFilterControls value={filters} onChange={setFilters} /></Box>
            <Typography variant="caption" color="text.secondary">{lakeTotal != null ? `${lakeTotal} 个观测区域，显示 ${lakeItems.length} 个` : "加载中"}</Typography>
          </>}
        </Box>
        {isLakeWorkspace && (lakes.isLoading ? <Box sx={{ p: 3, textAlign: "center" }}><CircularProgress size={24} /></Box> : lakes.isError ? <Typography color="error" sx={{ p: 2 }}>{lakes.error.message}</Typography> : (
          <List disablePadding>
            {!lakeItems.length && <Box sx={{ p: 3, textAlign: "center" }}><Typography variant="body2">没有符合条件的观测区域</Typography></Box>}
            {lakeItems.map((lake, index) => (
              <ListItemButton key={`${lake.region || region}:${lake.site_id || lake.object_id}:${index}`} divider selected={(lake.site_id || lake.object_id) === selectedLakeId} onClick={() => navigate(`/regions/${lake.region || region}/sites/${lake.site_id || lake.object_id}${lakeSearch}`)}>
                <ListItemText primary={<Box sx={{ display: "flex", alignItems: "center", gap: .75 }}><Typography variant="body2" noWrap sx={{ flex: 1 }}>{lake.display_name || lake.name || lake.site_id || lake.object_id}</Typography>{region === "all" && <Chip size="small" label={lake.region_name || lake.region} />}</Box>} secondary={`覆盖 ${(lake.coverage_area_km2 ?? lake.area_km2).toFixed(2)} km² · ${(lake.tiles || []).join(", ")}${lake.has_tci ? " · 影像" : ""}`} />
              </ListItemButton>
            ))}
            {lakes.hasNextPage && <Box sx={{ p: 1.5 }}><Button fullWidth variant="outlined" disabled={lakes.isFetchingNextPage} onClick={() => lakes.fetchNextPage()}>{lakes.isFetchingNextPage ? "加载中" : "加载更多"}</Button></Box>}
          </List>
        ))}
      </Box>
      <Box component="main" sx={{ minWidth: 0, display: { xs: selectedLakeId || isTraining || isModelValidation ? "grid" : "none", md: "grid" }, gridTemplateRows: selectedLakeId ? "auto minmax(0, 1fr) auto" : isTraining ? "auto minmax(0, 1fr)" : "1fr", color: "text.secondary" }}>
        {isModelValidation ? <ModelValidationView scope={region} routeLakeId={modelLakeId} routeLakeRegion={modelLakeRegion} routeModel={routeModel} onBack={() => navigate(`/regions/${region}/sites${lakeSearch}`)} onRouteModel={(model) => navigate(`/regions/${region}/model?model=${encodeURIComponent(model)}`, { replace: true })} onRouteResult={(siteId, siteRegion, model) => navigate(`/regions/${region}/model/${encodeURIComponent(siteId)}?model=${encodeURIComponent(model)}${region === "all" ? `&site_region=${encodeURIComponent(siteRegion)}` : ""}`)} /> : isTraining ? <>
          <Box sx={{ borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", alignItems: "center", px: { xs: 1, sm: 2 } }}>
            <IconButton sx={{ display: { md: "none" }, mr: .5 }} onClick={() => navigate(`/regions/${region}/sites`)}><ArrowLeft size={19} /></IconButton>
            <Tabs value={trainingView} onChange={(_, value) => navigate(`/regions/${region}/training/${value}`)}>
              <Tab value="samples" label="样本" />
              <Tab value="patches" label="Patch 审核" />
              <Tab value="train" label="训练" />
            </Tabs>
            <Button sx={{ ml: "auto", display: { xs: "none", sm: "inline-flex" } }} startIcon={<Images size={16} />} onClick={() => navigate(`/regions/${region}/sites`)}>浏览区域</Button>
          </Box>
          {trainingView === "patches" ? <PatchReviewView scope={region} onLocate={navigateToLake} /> : trainingView === "train" ? <TrainingView scope={region} /> : <TrainingSamplesView scope={region} onLocate={navigateToLake} />}
        </> : !selectedLakeId ? <Box sx={{ display: "grid", placeItems: "center" }}><Typography>选择一个观测区域</Typography></Box> : lake.isLoading ? <Box sx={{ display: "grid", placeItems: "center" }}><CircularProgress size={28} /></Box> : lake.data ? <>
          <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", alignItems: "center", gap: 1 }}>
            <IconButton sx={{ display: { md: "none" } }} onClick={() => navigate(`/regions/${region}/sites${lakeSearch}`)}><ArrowLeft size={19} /></IconButton>
            <Box><Typography variant="subtitle1" color="text.primary">{lake.data.display_name || lake.data.name || lake.data.site_id || lake.data.object_id}</Typography><Typography variant="caption">影像覆盖 {(lake.data.coverage_area_km2 ?? lake.data.area_km2).toFixed(2)} km² · {lake.data.image_count || 0} 期影像</Typography></Box>
          </Box>
          <SiteMap
            ref={mapRef}
            lake={lake.data}
            tileMeta={tileMeta.data}
            sentinelTiles={sentinelTiles.data}
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
            <ImageryPanel region={selectedRegion} lakeId={selectedLakeId} onSelectionChange={handleImagerySelection} />
            <TrainingCapture region={selectedRegion} lakeId={selectedLakeId} jrcThreshold={jrcThreshold} localLabel={(localLabels.data || []).find((item) => item.id === selectedLocalLabel)} imagery={imagerySelection} mapHandle={mapRef} />
          </Box>
        </> : <Box sx={{ display: "grid", placeItems: "center" }}><Typography color="error">观测区域加载失败</Typography></Box>}
      </Box>
    </Box>
  );
}
