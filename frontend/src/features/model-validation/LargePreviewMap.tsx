import { useEffect, useRef, useState, type ReactNode } from "react";
import { Alert, Box, Button, Checkbox, CircularProgress, FormControlLabel, MenuItem, TextField } from "@mui/material";
import Map from "ol/Map";
import View from "ol/View";
import { transformExtent } from "ol/proj";
import { defaults as controls } from "ol/control/defaults";
import { defaults as interactions } from "ol/interaction/defaults";
import ImageLayer from "ol/layer/Image";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import GeoJSON from "ol/format/GeoJSON";
import { Fill, Stroke, Style } from "ol/style";
import ImageStatic from "ol/source/ImageStatic";
import OSM from "ol/source/OSM";
import XYZ from "ol/source/XYZ";
import { createBasemapTileLoadFunction } from "../map/basemapCache";

interface Props {
  width: number; height: number; base: string; overlayKey?: string; overlay?: string | null; overlayColor?: string; ignored?: string | null;
  peekOriginal?: boolean; imageryStyle: string; onImageryStyleChange: (style: string) => void;
  imageTileUrl: string; previewBase: boolean; osm: boolean; bounds?: number[]; nativeBounds?: number[]; crs: string; siteBounds?: number[];
  navigation?: ReactNode; siteGeometry?: object; zoom: number; onZoom: (zoom: number) => void;
}

export function LargePreviewMap({ peekOriginal = false, imageryStyle, onImageryStyleChange, imageTileUrl, previewBase, base, overlayKey, overlay, overlayColor, ignored, osm, bounds, nativeBounds, crs, siteBounds, siteGeometry, navigation, zoom, onZoom }: Props) {
  const target = useRef<HTMLDivElement>(null);
  const map = useRef<Map | null>(null);
  const imageLayers = useRef<ImageLayer<ImageStatic>[]>([]);
  const baseLayers = useRef<TileLayer[]>([]);
  const rasterLayer = useRef<TileLayer<XYZ> | null>(null);
  const fittedResolution = useRef(1);
  const zoomRef = useRef(zoom); zoomRef.current = zoom;
  const callback = useRef(onZoom); callback.current = onZoom;
  const [basemap, setBasemap] = useState("satellite");
  const [showRaster, setShowRaster] = useState(true);
  const [tileError, setTileError] = useState(false);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [overlayError, setOverlayError] = useState(false);
  const extentKey = JSON.stringify([bounds, nativeBounds, siteBounds, siteGeometry]);
  const fit = (site: boolean) => {
    const extent = site ? siteBounds || bounds : bounds;
    if (!extent || !map.current) return;
    const instance = map.current; instance.updateSize();
    instance.getView().fit(transformExtent(extent, "EPSG:4326", "EPSG:3857"), { size: instance.getSize(), padding: [30, 30, 30, 30], maxZoom: 16 });
    fittedResolution.current = instance.getView().getResolution() || 1;
    callback.current(1);
  };
  useEffect(() => {
    if (!target.current || !bounds || !nativeBounds) return;
    const images = [0, 1, 2].map(() => new ImageLayer<ImageStatic>());
    const load = createBasemapTileLoadFunction();
    const osmSource = new OSM({ url: "https://tile.openstreetmap.de/{z}/{x}/{y}.png", crossOrigin: "anonymous", tileLoadFunction: load });
    const satelliteSource = new XYZ({ url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attributions: "Tiles © Esri", crossOrigin: "anonymous", maxZoom: 19, tileLoadFunction: load });
    osmSource.on("tileloaderror", () => setTileError(true)); satelliteSource.on("tileloaderror", () => setTileError(true));
    const bases = [new TileLayer({ source: osmSource }), new TileLayer({ source: satelliteSource })];
    const raster = new TileLayer({ source: new XYZ({ url: imageTileUrl, tileSize: 256, minZoom: 5, maxZoom: 16 }), visible: false });
    raster.getSource()?.on("tileloaderror", () => setTileError(true));
    const boundary = new VectorLayer({ source: new VectorSource({ features: siteGeometry ? new GeoJSON().readFeatures({ type: "Feature", geometry: siteGeometry, properties: {} }, { dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" }) : [] }), style: new Style({ stroke: new Stroke({ color: "#ffca28", width: 2 }), fill: new Fill({ color: "rgba(255,202,40,0.02)" }) }) });
    const instance = new Map({ target: target.current, view: new View({ projection: "EPSG:3857", center: [0, 0], zoom: 5, minZoom: 4, maxZoom: 17, enableRotation: false }), interactions: interactions({ doubleClickZoom: false, keyboard: false, pinchZoom: false, shiftDragZoom: false }), layers: [...bases, images[0], raster, ...images.slice(1), boundary], controls: controls({ zoom: false, rotate: false, attributionOptions: { collapsible: false } }) });
    map.current = instance; imageLayers.current = images; baseLayers.current = bases; rasterLayer.current = raster;
    fit(true);
    instance.on("moveend", () => {
      const ratio = fittedResolution.current / (instance.getView().getResolution() || 1);
      if (Math.abs(ratio - zoomRef.current) > .001) { zoomRef.current = ratio; callback.current(ratio); }
    });
    const observer = new ResizeObserver(() => instance.updateSize()); observer.observe(target.current);
    return () => { observer.disconnect(); instance.dispose(); map.current = null; rasterLayer.current = null; };
  }, [extentKey, crs]);
  useEffect(() => {
    const layer = rasterLayer.current;
    if (!layer || layer.getSource()?.getUrls()?.[0] === imageTileUrl) return;
    const previous = layer.getSource();
    const source = new XYZ({ url: imageTileUrl, tileSize: 256, minZoom: 5, maxZoom: 16 });
    source.on("tileloaderror", () => setTileError(true));
    layer.setSource(source); previous?.dispose();
  }, [imageTileUrl, extentKey, crs]);
  useEffect(() => { if (osm) setBasemap("osm"); }, [osm]);
  useEffect(() => {
    setTileError(false);
    baseLayers.current[0]?.setVisible(basemap === "osm"); baseLayers.current[1]?.setVisible(basemap === "satellite");
    rasterLayer.current?.setVisible((showRaster || peekOriginal) && !osm && !previewBase);
    imageLayers.current.forEach((layer, index) => layer.setVisible((showRaster || peekOriginal) && !osm && (!peekOriginal || index === 0)));
    map.current?.renderSync();
  }, [basemap, showRaster, osm, previewBase, peekOriginal, extentKey, imageTileUrl]);
  useEffect(() => {
    ([[base, 0], [ignored, 2]] as const).forEach(([url, index]) => {
      imageLayers.current[index]?.setSource(url && nativeBounds ? new ImageStatic({
        url, projection: crs, imageExtent: nativeBounds, interpolate: index === 0,
      }) : null);
    });
  }, [base, ignored, crs, extentKey]);
  useEffect(() => {
    const instance = map.current;
    const previous = imageLayers.current[1];
    if (!instance || !previous) return;
    let cancelled = false;
    let input: HTMLImageElement | undefined;
    let source: ImageStatic | undefined;
    // Replace the result layer itself to discard the renderer's previous image.
    const layer = new ImageLayer<ImageStatic>({ visible: previous.getVisible() });
    const position = instance.getLayers().getArray().indexOf(previous);
    imageLayers.current[1] = layer;
    instance.getLayers().setAt(position, layer);
    previous.setSource(null); previous.dispose();
    instance.renderSync();
    setOverlayError(false); setOverlayLoading(Boolean(overlay));
    const failed = () => {
      if (cancelled) return;
      layer.setSource(null); instance.render();
      setOverlayLoading(false); setOverlayError(true);
    };
    const attach = (url: string) => {
      if (cancelled || !nativeBounds) return;
      source = new ImageStatic({ url, projection: crs, imageExtent: nativeBounds, interpolate: false });
      source.once("imageloadend", () => {
        if (cancelled) return;
        setOverlayLoading(false); layer.changed(); instance.render();
      });
      source.once("imageloaderror", failed);
      layer.setSource(source); layer.changed(); instance.render();
    };
    if (overlay && nativeBounds) {
      if (overlayColor) {
        // Colorize before attaching the source, so its standard loader can await decoding.
        input = new Image();
        input.onload = () => {
          if (cancelled || !input) return;
          try {
            const canvas = document.createElement("canvas"); canvas.width = input.naturalWidth; canvas.height = input.naturalHeight;
            const context = canvas.getContext("2d");
            if (!context) throw new Error("Canvas is unavailable");
            context.drawImage(input, 0, 0); context.globalCompositeOperation = "source-in";
            context.fillStyle = overlayColor; context.fillRect(0, 0, canvas.width, canvas.height);
            attach(canvas.toDataURL());
          } catch { failed(); }
        };
        input.onerror = failed; input.src = overlay;
      } else attach(overlay);
    }
    return () => {
      cancelled = true;
      if (input) { input.onload = null; input.onerror = null; }
      source?.dispose();
    };
  }, [overlayKey, overlay, overlayColor, crs, extentKey]);
  useEffect(() => {
    const view = map.current?.getView();
    if (view && Math.abs((view.getResolution() || 1) - fittedResolution.current / zoom) > .000001) view.setResolution(fittedResolution.current / zoom);
  }, [zoom]);
  return <Box sx={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column" }}>
    <Box sx={{ p: 1, bgcolor: "background.paper", display: "flex", alignItems: "center", gap: 1, flexShrink: 0 }}>
      <TextField size="small" select label="底图" value={basemap} onChange={(event) => setBasemap(event.target.value)} sx={{ minWidth: 130 }}><MenuItem value="satellite">卫星图</MenuItem><MenuItem value="osm">OSM 地图</MenuItem><MenuItem value="none">无底图</MenuItem></TextField>
      <TextField size="small" select label="影像底图" value={imageryStyle} onChange={(event) => onImageryStyleChange(event.target.value)} sx={{ minWidth: 170 }}><MenuItem value="rgb">RGB</MenuItem><MenuItem value="nir">NIR-R-G</MenuItem><MenuItem value="swir">SWIR-NIR-R</MenuItem></TextField>
      <FormControlLabel control={<Checkbox checked={showRaster && !osm} disabled={osm} onChange={(_, value) => setShowRaster(value)} />} label="显示当前影像/结果" />
      <Button onClick={() => fit(true)}>定位整个 Site</Button><Button onClick={() => fit(false)}>定位当前影像</Button>
    </Box>
    <Box sx={{ position: "relative", flex: 1, minHeight: 0 }}>
      <Box ref={target} data-testid="comparison-preview-map" aria-label="当前 Site 地理地图" sx={{ position: "absolute", inset: 0 }} />
      {navigation && <Box sx={{ position: "absolute", top: 12, left: 12, zIndex: 5, display: "flex", gap: 1 }}>{navigation}</Box>}
    </Box>
    {overlayLoading && showRaster && !osm && !peekOriginal && <Box role="status" sx={{ position: "absolute", bottom: 36, left: 8, px: 1, py: .5, bgcolor: "background.paper", display: "flex", alignItems: "center", gap: 1, fontSize: 13 }}><CircularProgress size={16} />正在更新结果图层</Box>}
    {overlayError && <Alert severity="error" sx={{ position: "absolute", bottom: 36, left: 8 }}>结果图层加载失败，请切换模型重试。</Alert>}
    {(!bounds || !nativeBounds || tileError) && <Alert severity="warning" sx={{ position: "absolute", bottom: 30, left: 8 }}>{!bounds || !nativeBounds ? "缺少当前影像的地理范围，请刷新页面" : "部分底图瓦片未加载，可切换底图或检查网络。"}</Alert>}
  </Box>;
}
