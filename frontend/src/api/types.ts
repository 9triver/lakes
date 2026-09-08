export interface RegionSummary {
  key: string;
  name: string;
  ready: boolean;
  site_count: number;
}

export interface TrainingWorkspace {
  id: string;
  name: string;
  status: "active" | "archived";
  selected_patch_count: number;
  site_count: number;
  default: boolean;
  training_defaults?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface WorkbenchUser {
  id: string;
  name: string;
  status: "active" | "archived";
  role?: "user" | "admin";
  email?: string;
  auth_provider?: string;
  default_workspace_id: string;
  workspace: TrainingWorkspace;
  default: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface AuthSession {
  authenticated: true;
  mode: "development" | "cloudflare";
  identity: { provider: string; subject: string; email: string; name: string };
  user: WorkbenchUser;
  logout_url: string;
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
  first_acquisition_date?: string;
  last_acquisition_date?: string;
  has_tci?: boolean;
  region?: string;
  region_name?: string;
  polygon_quality?: string;
  metadata_quality?: string;
  best_tci_date?: string;
  included_logical_patch_count?: number;
}

export interface SitesResponse {
  total: number;
  items: SiteSummary[];
  facets?: Record<string, Record<string, number>>;
}

export interface GeoJsonLayer {
  geometry?: Record<string, unknown> | null;
  properties?: Record<string, unknown>;
}

export type GeoJsonGeometry = Record<string, unknown>;

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Array<{ type: "Feature"; geometry: Record<string, unknown>; properties?: Record<string, unknown> }>;
  properties?: Record<string, unknown>;
}

export type GeneratedLabelSource = "spectral_water" | "spectral_osm_intersection" | "spectral_osm_consensus" | "osm_spectral_consensus";

export interface GeneratedLabelResult {
  label_id: string;
  source: GeneratedLabelSource;
  label: FeatureCollection;
  stats: {
    pixels: number;
    valid_pixels: number;
    water_pixels: number;
    background_pixels: number;
    ignore_pixels: number;
    water_ratio: number;
    confident_ratio: number;
    polygon_count: number;
    ignore_polygon_count: number;
    mean_water_score: number;
    max_water_score: number;
  };
  details: Record<string, unknown>;
}

export type GeneratedLabelResults = Partial<Record<GeneratedLabelSource, GeneratedLabelResult>>;

export interface SiteDetail extends SiteSummary {
  site_id: string;
  bbox: [number, number, number, number];
  geometry: Record<string, unknown>;
}

export interface TileMeta {
  bounds?: [number, number, number, number];
  site_bounds?: [number, number, number, number];
  tile_bounds?: [number, number, number, number];
  tile_url: string;
  tiles?: string[];
  dates?: string[];
  products?: string[];
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
  asset_id?: string;
  product: string;
  tile?: string;
  date?: string;
  active?: boolean;
  asset_label?: string;
  source?: string;
  valid_ratio?: number;
  label_id?: string;
  label?: LocalLabelItem | null;
}

export interface ImageryResponse {
  site_id: string;
  selection_mode?: "site_imagery" | "tile_fallback";
  assets?: ImageryProduct[];
  tiles?: Array<{ tile: string; products: ImageryProduct[] }>;
}

export interface SentinelTile {
  tile: string;
  downloaded: boolean;
  geometry?: Record<string, unknown> | null;
  aoi_coverage_ratio?: number;
  date?: string | null;
  product?: string | null;
}

export interface TrainingPatch {
  patch_id: string;
  sample_id: string;
  site_id: string;
  site_display_name?: string;
  site_name?: string;
  region?: string;
  region_name?: string;
  included: boolean;
  preview_exists?: boolean;
  overlay_available?: boolean;
  preview_url?: string;
  water_pixels?: number;
  water_ratio_valid?: number;
  valid_ratio?: number;
  ignore_pixels?: number;
  product_name?: string;
  image_path?: string;
  product_date?: string;
  label_source?: string;
  label_sources?: string;
  logical_patch_id?: string;
  image_index?: number | string;
  geometry?: GeoJsonGeometry | null;
  workspace_id?: string;
  logical_size?: number | string;
  window_width?: number | string;
  window_height?: number | string;
  image_fingerprint?: string;
  label_fingerprint?: string;
  contributed?: boolean;
  contribution_scopes?: string[];
}

export interface GlobalDataset {
  dataset_id: string;
  scope: string;
  total: number;
  manifest?: string;
  latest_version?: string;
  items: TrainingPatch[];
}
