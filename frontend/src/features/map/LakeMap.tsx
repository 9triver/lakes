import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Box, Checkbox, FormControl, FormControlLabel, IconButton, MenuItem, Select, Slider, Tooltip, Typography } from "@mui/material";
import { Focus, Grid2X2 } from "lucide-react";
import Map from "ol/Map";
import View from "ol/View";
import GeoJSON from "ol/format/GeoJSON";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import XYZ from "ol/source/XYZ";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style, Text as TextStyle } from "ol/style";
import { toLonLat, transformExtent } from "ol/proj";
import type { FeatureCollection, GeoJsonLayer, LakeDetail, LocalLabelItem, SentinelTile, TileMeta } from "../../api/types";

interface LakeMapProps {
  lake: LakeDetail;
  tileMeta?: TileMeta;
  sentinelTiles?: SentinelTile[];
  contextOsm?: FeatureCollection;
  contextHydro?: FeatureCollection;
  esa?: GeoJsonLayer | null;
  jrc?: GeoJsonLayer | null;
  localLabel?: FeatureCollection;
  localLabels: LocalLabelItem[];
  selectedLocalLabel: string;
  onLocalLabelChange: (labelId: string) => void;
  jrcThreshold: number;
  onJrcThresholdChange: (threshold: number) => void;
  modelPrediction?: FeatureCollection;
}

export interface LakeMapHandle {
  captureView: () => {
    imagery_visible: boolean;
    visible_layers: Record<string, boolean>;
    map: { center: number[]; zoom: number; extent: number[] };
  } | null;
}

function vectorStyle(stroke: string, fill: string) {
  return new Style({ stroke: new Stroke({ color: stroke, width: 2 }), fill: new Fill({ color: fill }) });
}

export const LakeMap = forwardRef<LakeMapHandle, LakeMapProps>(function LakeMap({ lake, tileMeta, sentinelTiles, contextOsm, contextHydro, esa, jrc, localLabel, localLabels, selectedLocalLabel, onLocalLabelChange, jrcThreshold, onJrcThresholdChange, modelPrediction }, ref) {
  const targetRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const imageLayerRef = useRef(new TileLayer({ visible: true }));
  const osmSourceRef = useRef(new VectorSource());
  const tileSourceRef = useRef(new VectorSource());
  const hydroSourceRef = useRef(new VectorSource());
  const contextOsmSourceRef = useRef(new VectorSource());
  const contextHydroSourceRef = useRef(new VectorSource());
  const esaSourceRef = useRef(new VectorSource());
  const jrcSourceRef = useRef(new VectorSource());
  const localSourceRef = useRef(new VectorSource());
  const predictionSourceRef = useRef(new VectorSource());
  const osmLayerRef = useRef(new VectorLayer({ source: osmSourceRef.current, style: vectorStyle("#00a6ff", "rgba(0,166,255,.20)") }));
  const hydroLayerRef = useRef(new VectorLayer({ source: hydroSourceRef.current, style: vectorStyle("#d6a900", "rgba(255,212,71,.18)") }));
  const contextOsmLayerRef = useRef(new VectorLayer({ source: contextOsmSourceRef.current, style: vectorStyle("#0088cc", "rgba(0,136,204,.06)") }));
  const contextHydroLayerRef = useRef(new VectorLayer({ source: contextHydroSourceRef.current, style: vectorStyle("#967800", "rgba(255,212,71,.06)") }));
  const esaLayerRef = useRef(new VectorLayer({ source: esaSourceRef.current, style: vectorStyle("#e53696", "rgba(229,54,150,.28)") }));
  const jrcLayerRef = useRef(new VectorLayer({ source: jrcSourceRef.current, style: vectorStyle("#0b9c64", "rgba(11,156,100,.24)") }));
  const localLayerRef = useRef(new VectorLayer({ source: localSourceRef.current, style: vectorStyle("#ffffff", "rgba(0,0,0,.08)") }));
  const predictionLayerRef = useRef(new VectorLayer({ source: predictionSourceRef.current, style: vectorStyle("#ff3b30", "rgba(255,59,48,.32)") }));
  const tileLayerRef = useRef(new VectorLayer({ source: tileSourceRef.current, style: (feature) => new Style({ stroke: new Stroke({ color: "rgba(247,125,35,.95)", width: 2 }), fill: new Fill({ color: "rgba(247,125,35,.04)" }), text: new TextStyle({ text: String(feature.get("tile") || ""), font: "600 13px system-ui", fill: new Fill({ color: "#743900" }), stroke: new Stroke({ color: "rgba(255,255,255,.86)", width: 4 }), overflow: true }) }) }));
  const [visibility, setVisibility] = useState({ image: true, tile: true, osm: true, hydro: true, context: true, esa: true, jrc: true, local: true, prediction: true });

  useImperativeHandle(ref, () => ({
    captureView: () => {
      const map = mapRef.current;
      if (!map) return null;
      const view = map.getView();
      const size = map.getSize();
      if (!size) return null;
      return {
        imagery_visible: visibility.image,
        visible_layers: {
          tile_grid: visibility.tile,
          osm: visibility.osm,
          hydrolakes: visibility.hydro,
          context_osm: visibility.context,
          context_hydrolakes: visibility.context,
          esa: visibility.esa,
          jrc: visibility.jrc,
          local_label: visibility.local,
          model_prediction: false,
        },
        map: {
          center: toLonLat(view.getCenter() || [0, 0]),
          zoom: view.getZoom() || 0,
          extent: transformExtent(view.calculateExtent(size), "EPSG:3857", "EPSG:4326"),
        },
      };
    },
  }), [visibility]);

  useEffect(() => {
    if (!targetRef.current || mapRef.current) return;
    mapRef.current = new Map({
      target: targetRef.current,
      layers: [imageLayerRef.current, tileLayerRef.current, contextOsmLayerRef.current, contextHydroLayerRef.current, osmLayerRef.current, hydroLayerRef.current, esaLayerRef.current, jrcLayerRef.current, localLayerRef.current, predictionLayerRef.current],
      view: new View({ center: [0, 0], zoom: 6, minZoom: 4, maxZoom: 17 }),
    });
    return () => { mapRef.current?.setTarget(undefined); mapRef.current = null; };
  }, []);

  useEffect(() => {
    imageLayerRef.current.setVisible(visibility.image);
    tileLayerRef.current.setVisible(visibility.tile);
    osmLayerRef.current.setVisible(visibility.osm);
    hydroLayerRef.current.setVisible(visibility.hydro);
    contextOsmLayerRef.current.setVisible(visibility.context);
    contextHydroLayerRef.current.setVisible(visibility.context);
    esaLayerRef.current.setVisible(visibility.esa);
    jrcLayerRef.current.setVisible(visibility.jrc);
    localLayerRef.current.setVisible(visibility.local);
    predictionLayerRef.current.setVisible(visibility.prediction);
  }, [visibility]);

  useEffect(() => {
    const format = new GeoJSON({ dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" });
    tileSourceRef.current.clear();
    for (const tile of sentinelTiles || []) {
      if (!tile.geometry) continue;
      tileSourceRef.current.addFeatures(format.readFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: tile.geometry, properties: { tile: tile.tile, aoi_coverage_ratio: tile.aoi_coverage_ratio } }] }));
    }
  }, [sentinelTiles]);

  useEffect(() => {
    const format = new GeoJSON({ dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" });
    osmSourceRef.current.clear();
    hydroSourceRef.current.clear();
    if (lake.layers?.osm?.geometry) osmSourceRef.current.addFeatures(format.readFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: lake.layers.osm.geometry, properties: lake.layers.osm.properties || {} }] }));
    if (lake.layers?.hydrolakes?.geometry) hydroSourceRef.current.addFeatures(format.readFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: lake.layers.hydrolakes.geometry, properties: lake.layers.hydrolakes.properties || {} }] }));
    const bounds = tileMeta?.lake_bounds || lake.bbox;
    mapRef.current?.getView().fit(transformExtent(bounds, "EPSG:4326", "EPSG:3857"), { padding: [40, 40, 40, 40], maxZoom: 14 });
  }, [lake, tileMeta]);

  useEffect(() => {
    const format = new GeoJSON({ dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" });
    const setCollection = (source: VectorSource, collection?: FeatureCollection) => {
      source.clear();
      if (collection?.features?.length) source.addFeatures(format.readFeatures(collection));
    };
    const setLayer = (source: VectorSource, layer?: GeoJsonLayer | null) => {
      source.clear();
      if (layer?.geometry) source.addFeatures(format.readFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: layer.geometry, properties: layer.properties || {} }] }));
    };
    setCollection(contextOsmSourceRef.current, contextOsm);
    setCollection(contextHydroSourceRef.current, contextHydro);
    setLayer(esaSourceRef.current, esa);
    setLayer(jrcSourceRef.current, jrc);
    setCollection(localSourceRef.current, localLabel);
    setCollection(predictionSourceRef.current, modelPrediction);
  }, [contextOsm, contextHydro, esa, jrc, localLabel, modelPrediction]);

  useEffect(() => {
    imageLayerRef.current.setSource(tileMeta ? new XYZ({ url: `${tileMeta.tile_url}?v=${Date.now()}`, tileSize: 256, minZoom: 5, maxZoom: 16 }) : null);
  }, [tileMeta]);

  return (
    <Box sx={{ position: "relative", height: "100%", minHeight: 360, bgcolor: "#101010" }}>
      <Box sx={{ position: "absolute", zIndex: 2, top: 10, left: 10, right: 10, bgcolor: "rgba(255,255,255,.94)", border: 1, borderColor: "divider", px: 1, borderRadius: 1, display: "flex", flexWrap: "wrap", alignItems: "center", gap: .5 }}>
        <FormControlLabel control={<Checkbox size="small" checked={visibility.image} onChange={(_, checked) => setVisibility((value) => ({ ...value, image: checked }))} />} label="影像" />
        <FormControlLabel control={<Checkbox size="small" checked={visibility.tile} onChange={(_, checked) => setVisibility((value) => ({ ...value, tile: checked }))} />} label="Tile" />
        <FormControlLabel control={<Checkbox size="small" checked={visibility.osm} onChange={(_, checked) => setVisibility((value) => ({ ...value, osm: checked }))} />} label="OSM" />
        <FormControlLabel control={<Checkbox size="small" checked={visibility.hydro} onChange={(_, checked) => setVisibility((value) => ({ ...value, hydro: checked }))} />} label="HydroLAKES" />
        <FormControlLabel control={<Checkbox size="small" checked={visibility.context} onChange={(_, checked) => setVisibility((value) => ({ ...value, context: checked }))} />} label="其他" />
        <FormControlLabel control={<Checkbox size="small" checked={visibility.esa} onChange={(_, checked) => setVisibility((value) => ({ ...value, esa: checked }))} />} label="ESA" />
        <FormControlLabel control={<Checkbox size="small" checked={visibility.jrc} onChange={(_, checked) => setVisibility((value) => ({ ...value, jrc: checked }))} />} label="JRC" />
        <Box sx={{ width: 130, display: "flex", alignItems: "center", gap: 1 }}><Slider size="small" min={1} max={100} value={jrcThreshold} onChangeCommitted={(_, value) => onJrcThresholdChange(value as number)} /><Typography variant="caption">{jrcThreshold}%</Typography></Box>
        <FormControlLabel control={<Checkbox size="small" checked={visibility.local} onChange={(_, checked) => setVisibility((value) => ({ ...value, local: checked }))} />} label="本地标注" />
        <FormControl size="small" sx={{ minWidth: 210 }}>
          <Select value={selectedLocalLabel} displayEmpty onChange={(event) => onLocalLabelChange(event.target.value)}>
            <MenuItem value="">无本地标注</MenuItem>
            {localLabels.map((item) => <MenuItem key={item.id} value={item.id}>{item.date ? `${item.date} ${item.name}` : item.name}</MenuItem>)}
          </Select>
        </FormControl>
        {modelPrediction && <FormControlLabel control={<Checkbox size="small" checked={visibility.prediction} onChange={(_, checked) => setVisibility((value) => ({ ...value, prediction: checked }))} />} label="模型预测" />}
        <Box sx={{ ml: "auto", display: "flex", alignItems: "center" }}>
          <Tooltip title="定位水体"><span><IconButton size="small" disabled={!tileMeta?.lake_bounds} onClick={() => tileMeta?.lake_bounds && mapRef.current?.getView().fit(transformExtent(tileMeta.lake_bounds, "EPSG:4326", "EPSG:3857"), { padding: [40, 40, 40, 40], maxZoom: 14 })} aria-label="定位水体"><Focus size={18} /></IconButton></span></Tooltip>
          <Tooltip title="定位 Tile"><span><IconButton size="small" disabled={!tileMeta?.tile_bounds} onClick={() => tileMeta?.tile_bounds && mapRef.current?.getView().fit(transformExtent(tileMeta.tile_bounds, "EPSG:4326", "EPSG:3857"), { padding: [40, 40, 40, 40], maxZoom: 14 })} aria-label="定位 Tile"><Grid2X2 size={18} /></IconButton></span></Tooltip>
        </Box>
      </Box>
      <Box ref={targetRef} sx={{ position: "absolute", inset: 0 }} />
      {!tileMeta && <Typography sx={{ position: "absolute", bottom: 12, left: 12, color: "white", bgcolor: "rgba(0,0,0,.65)", px: 1 }}>当前水体没有可用影像</Typography>}
    </Box>
  );
});
