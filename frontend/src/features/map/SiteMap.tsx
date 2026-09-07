import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { Box, Typography } from "@mui/material";
import Map from "ol/Map";
import View from "ol/View";
import GeoJSON from "ol/format/GeoJSON";
import { defaults as defaultInteractions } from "ol/interaction/defaults";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import OSM from "ol/source/OSM";
import XYZ from "ol/source/XYZ";
import VectorSource from "ol/source/Vector";
import { Fill, Stroke, Style, Text as TextStyle } from "ol/style";
import { toLonLat, transformExtent } from "ol/proj";
import type { FeatureCollection, GeoJsonLayer, SiteDetail, SentinelTile, TileMeta, TrainingPatch } from "../../api/types";
import { createBasemapTileLoadFunction } from "./basemapCache";

export type BasemapType = "osm" | "satellite" | "none";

export interface SiteLayerVisibility {
  image: boolean;
  tile: boolean;
  osm: boolean;
  hydro: boolean;
  context: boolean;
  esa: boolean;
  jrc: boolean;
  local: boolean;
  spectralWater: boolean;
  spectralOsmConsensus: boolean;
  osmSpectralConsensus: boolean;
  prediction: boolean;
}

export const DEFAULT_SITE_LAYER_VISIBILITY: SiteLayerVisibility = {
  image: true,
  tile: false,
  osm: false,
  hydro: false,
  context: false,
  esa: false,
  jrc: false,
  local: false,
  spectralWater: false,
  spectralOsmConsensus: false,
  osmSpectralConsensus: false,
  prediction: true,
};

interface SiteMapProps {
  site: SiteDetail;
  basemap: BasemapType;
  visibility: SiteLayerVisibility;
  tileMeta?: TileMeta;
  sentinelTiles?: SentinelTile[];
  osm?: GeoJsonLayer | null;
  hydrolakes?: GeoJsonLayer | null;
  contextOsm?: FeatureCollection;
  contextHydro?: FeatureCollection;
  esa?: GeoJsonLayer | null;
  jrc?: GeoJsonLayer | null;
  localLabel?: FeatureCollection | null;
  spectralWater?: FeatureCollection | null;
  spectralOsmConsensus?: FeatureCollection | null;
  osmSpectralConsensus?: FeatureCollection | null;
  modelPrediction?: FeatureCollection;
  patches?: TrainingPatch[];
  patchReviewEnabled?: boolean;
  activePatchId?: string;
  pendingPatchIds?: Set<string>;
  onPatchClick?: (patchId: string) => void;
  patchSourceMeta?: TileMeta;
}

export interface SiteMapHandle {
  captureView: () => {
    imagery_visible: boolean;
    visible_layers: Record<string, boolean>;
    map: { center: number[]; zoom: number; extent: number[] };
  } | null;
  fitSite: () => void;
  fitTile: () => void;
}

function vectorStyle(stroke: string, fill: string) {
  return new Style({ stroke: new Stroke({ color: stroke, width: 2 }), fill: new Fill({ color: fill }) });
}

export const SiteMap = forwardRef<SiteMapHandle, SiteMapProps>(function SiteMap({ site, basemap, visibility, tileMeta, sentinelTiles, osm, hydrolakes, contextOsm, contextHydro, esa, jrc, localLabel, spectralWater, spectralOsmConsensus, osmSpectralConsensus, modelPrediction, patches = [], patchReviewEnabled = false, activePatchId = "", pendingPatchIds = new Set(), onPatchClick, patchSourceMeta }, ref) {
  const targetRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const basemapTileLoadFunction = createBasemapTileLoadFunction();
  const osmBasemapLayerRef = useRef(new TileLayer({ source: new OSM({ crossOrigin: "anonymous", tileLoadFunction: basemapTileLoadFunction }), visible: true }));
  const satelliteBasemapLayerRef = useRef(new TileLayer({
    source: new XYZ({
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attributions: "Tiles © Esri",
      crossOrigin: "anonymous",
      maxZoom: 19,
      tileLoadFunction: basemapTileLoadFunction,
    }),
    visible: false,
  }));
  const imageLayerRef = useRef(new TileLayer({ visible: true }));
  const osmSourceRef = useRef(new VectorSource());
  const tileSourceRef = useRef(new VectorSource());
  const hydroSourceRef = useRef(new VectorSource());
  const contextOsmSourceRef = useRef(new VectorSource());
  const contextHydroSourceRef = useRef(new VectorSource());
  const esaSourceRef = useRef(new VectorSource());
  const jrcSourceRef = useRef(new VectorSource());
  const localSourceRef = useRef(new VectorSource());
  const spectralWaterSourceRef = useRef(new VectorSource());
  const spectralOsmConsensusSourceRef = useRef(new VectorSource());
  const osmSpectralConsensusSourceRef = useRef(new VectorSource());
  const predictionSourceRef = useRef(new VectorSource());
  const patchSourceRef = useRef(new VectorSource());
  const osmLayerRef = useRef(new VectorLayer({ source: osmSourceRef.current, style: vectorStyle("#00a6ff", "rgba(0,166,255,.20)") }));
  const hydroLayerRef = useRef(new VectorLayer({ source: hydroSourceRef.current, style: vectorStyle("#d6a900", "rgba(255,212,71,.18)") }));
  const contextOsmLayerRef = useRef(new VectorLayer({ source: contextOsmSourceRef.current, style: vectorStyle("#0088cc", "rgba(0,136,204,.06)") }));
  const contextHydroLayerRef = useRef(new VectorLayer({ source: contextHydroSourceRef.current, style: vectorStyle("#967800", "rgba(255,212,71,.06)") }));
  const esaLayerRef = useRef(new VectorLayer({ source: esaSourceRef.current, style: vectorStyle("#e53696", "rgba(229,54,150,.28)") }));
  const jrcLayerRef = useRef(new VectorLayer({ source: jrcSourceRef.current, style: vectorStyle("#0b9c64", "rgba(11,156,100,.24)") }));
  const localLayerRef = useRef(new VectorLayer({ source: localSourceRef.current, style: vectorStyle("#ffffff", "rgba(0,0,0,.08)") }));
  const spectralWaterLayerRef = useRef(new VectorLayer({ source: spectralWaterSourceRef.current, style: vectorStyle("#ff8a00", "rgba(255,138,0,.25)") }));
  const spectralOsmConsensusLayerRef = useRef(new VectorLayer({ source: spectralOsmConsensusSourceRef.current, style: vectorStyle("#00a896", "rgba(0,168,150,.30)") }));
  const osmSpectralConsensusLayerRef = useRef(new VectorLayer({ source: osmSpectralConsensusSourceRef.current, style: vectorStyle("#7b61ff", "rgba(123,97,255,.28)") }));
  const predictionLayerRef = useRef(new VectorLayer({ source: predictionSourceRef.current, style: vectorStyle("#ff3b30", "rgba(255,59,48,.32)") }));
  const patchLayerRef = useRef(new VectorLayer({ source: patchSourceRef.current, style: (feature) => {
    const included = Boolean(feature.get("included"));
    const pending = Boolean(feature.get("pending"));
    const active = Boolean(feature.get("active"));
    const color = pending ? "#ffb000" : included ? "#00a676" : "#d64545";
    return new Style({ stroke: new Stroke({ color, width: active || pending ? 4 : 2 }), fill: new Fill({ color: included ? "rgba(0,166,118,.10)" : "rgba(214,69,69,.16)" }) });
  } }));
  const tileLayerRef = useRef(new VectorLayer({ source: tileSourceRef.current, style: (feature) => new Style({ stroke: new Stroke({ color: "rgba(247,125,35,.95)", width: 2 }), fill: new Fill({ color: "rgba(247,125,35,.04)" }), text: new TextStyle({ text: String(feature.get("tile") || ""), font: "600 13px system-ui", fill: new Fill({ color: "#743900" }), stroke: new Stroke({ color: "rgba(255,255,255,.86)", width: 4 }), overflow: true }) }) }));

  const fitBounds = (bounds?: number[]) => {
    if (bounds) mapRef.current?.getView().fit(transformExtent(bounds, "EPSG:4326", "EPSG:3857"), { padding: [40, 40, 40, 40], maxZoom: 14 });
  };

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
          basemap_osm: basemap === "osm",
          basemap_satellite: basemap === "satellite",
          tile_grid: visibility.tile,
          osm: visibility.osm,
          hydrolakes: visibility.hydro,
          context_osm: visibility.context,
          context_hydrolakes: visibility.context,
          esa: visibility.esa,
          jrc: visibility.jrc,
          local_label: visibility.local,
          spectral_water: visibility.spectralWater && Boolean(spectralWater),
          spectral_osm_consensus: visibility.spectralOsmConsensus && Boolean(spectralOsmConsensus),
          osm_spectral_consensus: visibility.osmSpectralConsensus && Boolean(osmSpectralConsensus),
          model_prediction: visibility.prediction && Boolean(modelPrediction),
        },
        map: {
          center: toLonLat(view.getCenter() || [0, 0]),
          zoom: view.getZoom() || 0,
          extent: transformExtent(view.calculateExtent(size), "EPSG:3857", "EPSG:4326"),
        },
      };
    },
    fitSite: () => fitBounds(tileMeta?.site_bounds || site.bbox),
    fitTile: () => fitBounds(tileMeta?.tile_bounds),
  }), [basemap, modelPrediction, osmSpectralConsensus, site.bbox, spectralOsmConsensus, spectralWater, tileMeta, visibility]);

  useEffect(() => {
    if (!targetRef.current || mapRef.current) return;
    mapRef.current = new Map({
      target: targetRef.current,
      interactions: defaultInteractions({ doubleClickZoom: false, keyboard: false, pinchZoom: false, shiftDragZoom: false }),
      layers: [osmBasemapLayerRef.current, satelliteBasemapLayerRef.current, imageLayerRef.current, tileLayerRef.current, contextOsmLayerRef.current, contextHydroLayerRef.current, osmLayerRef.current, hydroLayerRef.current, esaLayerRef.current, jrcLayerRef.current, spectralWaterLayerRef.current, spectralOsmConsensusLayerRef.current, osmSpectralConsensusLayerRef.current, predictionLayerRef.current, localLayerRef.current, patchLayerRef.current],
      view: new View({ center: [0, 0], zoom: 6, minZoom: 4, maxZoom: 17 }),
    });
    return () => { mapRef.current?.setTarget(undefined); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !onPatchClick) return;
    const handleClick = (event: { pixel: number[]; originalEvent?: Event }) => {
      if (!patchReviewEnabled) return;
      if (event.originalEvent instanceof MouseEvent && event.originalEvent.detail > 1) return;
      const feature = map.forEachFeatureAtPixel(event.pixel, (candidate) => candidate, { layerFilter: (layer) => layer === patchLayerRef.current });
      const patchId = feature?.get("logical_patch_id");
      if (patchId) onPatchClick(String(patchId));
    };
    map.on("click", handleClick);
    return () => { map.un("click", handleClick); };
  }, [onPatchClick, patchReviewEnabled]);

  useEffect(() => {
    osmBasemapLayerRef.current.setVisible(basemap === "osm");
    satelliteBasemapLayerRef.current.setVisible(basemap === "satellite");
    imageLayerRef.current.setVisible(visibility.image);
    tileLayerRef.current.setVisible(visibility.tile);
    osmLayerRef.current.setVisible(visibility.osm);
    hydroLayerRef.current.setVisible(visibility.hydro);
    contextOsmLayerRef.current.setVisible(visibility.context);
    contextHydroLayerRef.current.setVisible(visibility.context);
    esaLayerRef.current.setVisible(visibility.esa);
    jrcLayerRef.current.setVisible(visibility.jrc);
    localLayerRef.current.setVisible(visibility.local);
    spectralWaterLayerRef.current.setVisible(visibility.spectralWater);
    spectralOsmConsensusLayerRef.current.setVisible(visibility.spectralOsmConsensus);
    osmSpectralConsensusLayerRef.current.setVisible(visibility.osmSpectralConsensus);
    predictionLayerRef.current.setVisible(visibility.prediction);
  }, [basemap, visibility]);

  useEffect(() => {
    const format = new GeoJSON({ dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" });
    tileSourceRef.current.clear();
    for (const tile of sentinelTiles || []) {
      if (!tile.geometry) continue;
      tileSourceRef.current.addFeatures(format.readFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: tile.geometry, properties: { tile: tile.tile, aoi_coverage_ratio: tile.aoi_coverage_ratio } }] }));
    }
  }, [sentinelTiles]);

  useEffect(() => {
    const bounds = tileMeta?.site_bounds || site.bbox;
    mapRef.current?.getView().fit(transformExtent(bounds, "EPSG:4326", "EPSG:3857"), { padding: [40, 40, 40, 40], maxZoom: 12 });
  }, [site, tileMeta]);

  useEffect(() => {
    const format = new GeoJSON({ dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" });
    const setCollection = (
      source: VectorSource,
      collection?: FeatureCollection | null,
      filter?: (feature: FeatureCollection["features"][number]) => boolean,
    ) => {
      source.clear();
      if (collection?.features?.length) {
        const features = filter ? collection.features.filter(filter) : collection.features;
        if (features.length) source.addFeatures(format.readFeatures({ ...collection, features }));
      }
    };
    const setLayer = (source: VectorSource, layer?: GeoJsonLayer | null) => {
      source.clear();
      if (layer?.geometry) source.addFeatures(format.readFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: layer.geometry, properties: layer.properties || {} }] }));
    };
    setLayer(osmSourceRef.current, osm);
    setLayer(hydroSourceRef.current, hydrolakes);
    setCollection(contextOsmSourceRef.current, contextOsm);
    setCollection(contextHydroSourceRef.current, contextHydro);
    setLayer(esaSourceRef.current, esa);
    setLayer(jrcSourceRef.current, jrc);
    setCollection(localSourceRef.current, localLabel);
    setCollection(
      spectralWaterSourceRef.current,
      spectralWater,
      (feature) => feature.properties?.label_class !== "ignore" && Number(feature.properties?.label_value) !== 255,
    );
    setCollection(
      spectralOsmConsensusSourceRef.current,
      spectralOsmConsensus,
      (feature) => feature.properties?.label_class !== "ignore" && Number(feature.properties?.label_value) !== 255,
    );
    setCollection(
      osmSpectralConsensusSourceRef.current,
      osmSpectralConsensus,
      (feature) => feature.properties?.label_class !== "ignore" && Number(feature.properties?.label_value) !== 255,
    );
    setCollection(predictionSourceRef.current, modelPrediction);
  }, [osm, hydrolakes, contextOsm, contextHydro, esa, jrc, localLabel, osmSpectralConsensus, spectralOsmConsensus, spectralWater, modelPrediction]);

  useEffect(() => {
    const format = new GeoJSON({ dataProjection: "EPSG:4326", featureProjection: "EPSG:3857" });
    patchSourceRef.current.clear();
    if (!patchReviewEnabled) return;
    const features = patches.filter((patch) => patch.geometry).map((patch) => ({ type: "Feature", geometry: patch.geometry, properties: { logical_patch_id: patch.logical_patch_id || patch.patch_id, included: patch.included, pending: pendingPatchIds.has(patch.logical_patch_id || patch.patch_id), active: (patch.logical_patch_id || patch.patch_id) === activePatchId } }));
    patchSourceRef.current.addFeatures(format.readFeatures({ type: "FeatureCollection", features }));
  }, [activePatchId, patches, patchReviewEnabled, pendingPatchIds]);

  useEffect(() => {
    const sourceMeta = patchReviewEnabled && patchSourceMeta ? patchSourceMeta : tileMeta;
    imageLayerRef.current.setSource(sourceMeta ? new XYZ({ url: `${sourceMeta.tile_url}${sourceMeta.tile_url.includes("?") ? "&" : "?"}v=${Date.now()}`, tileSize: 256, minZoom: 5, maxZoom: 16 }) : null);
  }, [patchReviewEnabled, patchSourceMeta, tileMeta]);

  return (
    <Box data-testid="site-map" sx={{ position: "relative", height: "100%", minHeight: 360, bgcolor: "#101010" }}>
      <Box ref={targetRef} sx={{ position: "absolute", inset: 0 }} />
      {!tileMeta && <Typography sx={{ position: "absolute", bottom: 12, left: 12, color: "white", bgcolor: "rgba(0,0,0,.65)", px: 1 }}>当前观测区域没有可用影像</Typography>}
    </Box>
  );
});
