function polygonStyle(ol, stroke, fill, lineDash = undefined, width = 2) {
  return new ol.style.Style({
    stroke: new ol.style.Stroke({ color: stroke, width, lineDash }),
    fill: new ol.style.Fill({ color: fill }),
  });
}

function tileGridStyle(ol, feature) {
  const tile = feature.get("tile") || "";
  return new ol.style.Style({
    stroke: new ol.style.Stroke({ color: "rgba(247, 125, 35, 0.95)", width: 2 }),
    fill: new ol.style.Fill({ color: "rgba(247, 125, 35, 0.04)" }),
    text: new ol.style.Text({
      text: tile,
      font: "600 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      fill: new ol.style.Fill({ color: "#743900" }),
      stroke: new ol.style.Stroke({ color: "rgba(255, 255, 255, 0.86)", width: 4 }),
      overflow: true,
    }),
  });
}

export function createMapController({
  target,
  wrap,
  canShow = () => true,
  isLayerVisible = () => true,
  ol = globalThis.ol,
}) {
  if (!ol) throw new Error("OpenLayers is not available");

  const rasterLayer = new ol.layer.Tile({ visible: true });
  const vectorSources = {
    tileGrid: new ol.source.Vector(),
    osm: new ol.source.Vector(),
    hydrolakes: new ol.source.Vector(),
    contextOsm: new ol.source.Vector(),
    contextHydrolakes: new ol.source.Vector(),
    esa: new ol.source.Vector(),
    jrc: new ol.source.Vector(),
    localLabel: new ol.source.Vector(),
    modelPrediction: new ol.source.Vector(),
  };
  const vectorLayers = {
    tileGrid: new ol.layer.Vector({ source: vectorSources.tileGrid, style: (feature) => tileGridStyle(ol, feature) }),
    osm: new ol.layer.Vector({ source: vectorSources.osm, style: polygonStyle(ol, "#00a6ff", "rgba(0, 166, 255, 0.20)") }),
    hydrolakes: new ol.layer.Vector({ source: vectorSources.hydrolakes, style: polygonStyle(ol, "#ffd447", "rgba(255, 212, 71, 0.18)") }),
    contextOsm: new ol.layer.Vector({ source: vectorSources.contextOsm, style: polygonStyle(ol, "#0088cc", "rgba(0, 136, 204, 0.06)", [6, 5], 1.4) }),
    contextHydrolakes: new ol.layer.Vector({ source: vectorSources.contextHydrolakes, style: polygonStyle(ol, "#b28b00", "rgba(255, 212, 71, 0.06)", [6, 5], 1.4) }),
    esa: new ol.layer.Vector({ source: vectorSources.esa, style: polygonStyle(ol, "#ff4fb3", "rgba(255, 79, 179, 0.30)") }),
    jrc: new ol.layer.Vector({ source: vectorSources.jrc, style: polygonStyle(ol, "#1ab878", "rgba(44, 214, 137, 0.24)") }),
    localLabel: new ol.layer.Vector({ source: vectorSources.localLabel, style: polygonStyle(ol, "#ffffff", "rgba(0, 0, 0, 0.08)", [8, 4], 2.5) }),
    modelPrediction: new ol.layer.Vector({ source: vectorSources.modelPrediction, style: polygonStyle(ol, "#f03a47", "rgba(240, 58, 71, 0.24)", undefined, 2.6) }),
  };
  const map = new ol.Map({
    target,
    layers: [
      rasterLayer,
      vectorLayers.tileGrid,
      vectorLayers.contextOsm,
      vectorLayers.contextHydrolakes,
      vectorLayers.osm,
      vectorLayers.hydrolakes,
      vectorLayers.esa,
      vectorLayers.jrc,
      vectorLayers.localLabel,
      vectorLayers.modelPrediction,
    ],
    view: new ol.View({
      center: ol.proj.fromLonLat([112.5, 28.8]),
      zoom: 8,
      minZoom: 5,
      maxZoom: 16,
    }),
  });
  const geojson = new ol.format.GeoJSON({
    dataProjection: "EPSG:4326",
    featureProjection: "EPSG:3857",
  });

  function addLayerGeometry(layerName, layer) {
    const source = vectorSources[layerName];
    source.clear();
    if (!layer?.geometry) return;
    source.addFeature(geojson.readFeature({
      type: "Feature",
      geometry: layer.geometry,
      properties: layer.properties || {},
    }));
    vectorLayers[layerName].setVisible(isLayerVisible(layerName));
  }

  function addFeatureCollection(layerName, collection) {
    const source = vectorSources[layerName];
    source.clear();
    const features = collection?.features || [];
    if (features.length) {
      source.addFeatures(geojson.readFeatures({ type: "FeatureCollection", features }));
    }
    vectorLayers[layerName].setVisible(isLayerVisible(layerName));
  }

  function ensureMapVisible() {
    if (!canShow()) return false;
    wrap.hidden = false;
    target.hidden = false;
    const rect = target.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      map.updateSize();
      return true;
    }
    requestAnimationFrame(() => map.updateSize());
    return false;
  }

  function clearVectorLayers(keep = new Set()) {
    for (const [name, source] of Object.entries(vectorSources)) {
      if (!keep.has(name)) source.clear();
    }
  }

  function fitToBounds(bounds) {
    if (!bounds || bounds.length !== 4) return;
    ensureMapVisible();
    const extent = ol.proj.transformExtent(bounds, "EPSG:4326", "EPSG:3857");
    map.getView().fit(extent, { padding: [36, 36, 36, 36], duration: 180, maxZoom: 14 });
    requestAnimationFrame(() => {
      map.updateSize();
      map.getView().fit(extent, { padding: [36, 36, 36, 36], duration: 0, maxZoom: 14 });
    });
  }

  return {
    map,
    rasterLayer,
    vectorSources,
    vectorLayers,
    geojson,
    addLayerGeometry,
    addFeatureCollection,
    ensureMapVisible,
    clearVectorLayers,
    fitToBounds,
  };
}
