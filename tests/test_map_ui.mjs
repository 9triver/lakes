import assert from "node:assert/strict";

import { createMapController } from "../src/lake_workbench/static/map-ui.js";

class VectorSource {
  constructor() {
    this.features = [];
  }
  clear() {
    this.features = [];
  }
  addFeature(feature) {
    this.features.push(feature);
  }
  addFeatures(features) {
    this.features.push(...features);
  }
}

class Layer {
  constructor(options = {}) {
    this.options = options;
    this.visible = options.visible ?? true;
  }
  setVisible(value) {
    this.visible = value;
  }
  getSource() {
    return this.options.source;
  }
}

class MapMock {
  constructor(options) {
    this.options = options;
    this.view = options.view;
  }
  getView() {
    return this.view;
  }
  updateSize() {}
}

class ViewMock {
  fit() {}
}

class GeoJSONMock {
  readFeature(feature) {
    return feature;
  }
  readFeatures(collection) {
    return collection.features;
  }
}

const ol = {
  layer: { Tile: Layer, Vector: Layer },
  source: { Vector: VectorSource },
  style: {
    Style: class { constructor(options) { this.options = options; } },
    Stroke: class { constructor(options) { this.options = options; } },
    Fill: class { constructor(options) { this.options = options; } },
    Text: class { constructor(options) { this.options = options; } },
  },
  Map: MapMock,
  View: ViewMock,
  format: { GeoJSON: GeoJSONMock },
  proj: {
    fromLonLat: (value) => value,
    transformExtent: (value) => value,
  },
};

const target = { hidden: false, getBoundingClientRect: () => ({ width: 800, height: 600 }) };
const wrap = { hidden: false };
const controller = createMapController({
  target,
  wrap,
  ol,
  isLayerVisible: (name) => name !== "osm",
});

controller.addLayerGeometry("osm", {
  geometry: { type: "Polygon", coordinates: [] },
  properties: { source: "OSM" },
});
assert.equal(controller.vectorSources.osm.features.length, 1);
assert.equal(controller.vectorLayers.osm.visible, false);

controller.addFeatureCollection("jrc", {
  features: [{ type: "Feature", geometry: { type: "Polygon", coordinates: [] }, properties: {} }],
});
assert.equal(controller.vectorSources.jrc.features.length, 1);
controller.clearVectorLayers(new Set(["jrc"]));
assert.equal(controller.vectorSources.osm.features.length, 0);
assert.equal(controller.vectorSources.jrc.features.length, 1);

console.log("map UI tests passed");
