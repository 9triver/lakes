export interface RegionSummary {
  key: string;
  name: string;
  ready: boolean;
  site_count: number;
}

export interface RegionsResponse {
  default: string;
  items: RegionSummary[];
}

export interface SiteSummary {
  site_id: string;
  display_name?: string;
  water_type?: string;
  coverage_area_km2: number;
  tiles?: string[];
  image_count?: number;
  label_asset_count?: number;
  label_feature_count?: number;
  has_tci?: boolean;
  region?: string;
  region_name?: string;
  polygon_quality?: string;
  metadata_quality?: string;
  best_tci_date?: string;
}

export interface SitesResponse {
  total: number;
  items: SiteSummary[];
  facets?: Record<string, Record<string, number>>;
}

export interface SiteFilters {
  area_bucket: string;
  has_name: string;
  has_tci: string;
  has_osm: string;
  has_hydrolakes: string;
  has_local_labels: string;
}

export interface GeoJsonLayer {
  geometry?: Record<string, unknown> | null;
  properties?: Record<string, unknown>;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Array<{ type: "Feature"; geometry: Record<string, unknown>; properties?: Record<string, unknown> }>;
}

export interface SiteDetail extends SiteSummary {
  site_id: string;
  bbox: [number, number, number, number];
  geometry: Record<string, unknown>;
}

export interface TileMeta {
  site_bounds?: [number, number, number, number];
  tile_bounds: [number, number, number, number];
  tile_url: string;
  tiles: string[];
  dates: string[];
  products: string[];
}

export interface ContextWaterResponse {
  min_area_km2: number;
  sources: { osm: FeatureCollection; hydrolakes: FeatureCollection };
}

export interface LocalLabelItem {
  id: string;
  name: string;
  date?: string;
  path?: string;
}

export interface ImageryProduct {
  product: string;
  date?: string;
  active?: boolean;
  asset_label?: string;
  source?: string;
  valid_ratio?: number;
}

export interface ImageryResponse {
  site_id: string;
  tiles: Array<{ tile: string; products: ImageryProduct[] }>;
}

export interface SentinelTile {
  tile: string;
  downloaded: boolean;
  geometry?: Record<string, unknown> | null;
  aoi_coverage_ratio?: number;
  date?: string | null;
  product?: string | null;
}

export interface SentinelProduct {
  product_id: string;
  name: string;
  tile: string;
  date?: string;
  cloud_cover?: number;
  site_coverage_ratio?: number;
  aoi_coverage_ratio?: number;
  downloaded?: boolean;
}

export interface TrainingSample {
  sample_id: string;
  site_id: string;
  site_display_name?: string;
  site_name?: string;
  region?: string;
  region_name?: string;
  status?: string;
  split?: string;
  notes?: string;
  label_source?: string;
  product_date?: string;
  tile_count?: number;
  imagery_asset_labels?: string[];
}

export interface TrainingPatch {
  patch_id: string;
  sample_id: string;
  site_id: string;
  site_display_name?: string;
  site_name?: string;
  region?: string;
  included: boolean;
  preview_exists?: boolean;
  preview_url?: string;
  water_pixels?: number;
  water_ratio_valid?: number;
  valid_ratio?: number;
  ignore_pixels?: number;
  product_name?: string;
  image_path?: string;
}
