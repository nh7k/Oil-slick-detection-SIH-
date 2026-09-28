import type { Feature, FeatureCollection, Geometry, MultiPolygon, Polygon } from 'geojson';

export type ClassId = 1 | 2 | 3;
export type BBox = [number, number, number, number];

export interface Health {
  status: string;
  model_loaded: boolean;
  model_version: string | null;
  db_detections: number;
}

export interface ModelInfo {
  loaded: boolean;
  model_version: string | null;
  architecture: string | null;
  classes: Record<string, string>;
  thresholds: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
  trained_on: string | null;
}

export interface DetectionProps {
  id: number;
  cls: ClassId;
  cls_name: string;
  confidence: number;
  confidence_mean: number | null;
  confidence_max: number | null;
  area_km2: number | null;
  length_km: number | null;
  perimeter_km: number | null;
  polsby_popper: number | null;
  centroid_lon: number | null;
  centroid_lat: number | null;
  bbox: BBox | null;
  scene_id: string | null;
  acquired_at: string | null;
  model_version: string | null;
  run_id: string | number | null;
}

export type DetectionGeometry = Polygon | MultiPolygon;
export type DetectionFeature = Feature<DetectionGeometry, DetectionProps>;
export type DetectionCollection = FeatureCollection<DetectionGeometry, DetectionProps>;

export interface Scene {
  scene_id: string;
  acquired_at: string | null;
  bounds: BBox;
  /** Lon/lat corners of the served PNGs (reprojected to Web Mercator server-side). Use these for image overlays. */
  image_bounds: BBox | null;
  n_detections: number;
  orbit_direction: string | null;
  processed_at: string | null;
}

export interface SearchSceneItem {
  scene_id: string;
  acquired_at: string | null;
  bounds: BBox;
  orbit_direction: string | null;
}

export type JobStatusValue = 'queued' | 'running' | 'done' | 'error';

export interface JobStatus {
  job_id: string;
  status: JobStatusValue;
  message: string | null;
  scene_id: string | null;
  n_detections: number | null;
}

export interface PredictGeotiffResponse {
  scene_id: string;
  detections: DetectionCollection;
}

export interface ReferenceProps {
  id?: number | string;
  machine_confidence?: number | null;
  cls?: number | string | null;
  slick_timestamp?: string | null;
  area?: number | null;
  s1_scene_id?: string | null;
  [key: string]: unknown;
}

export type ReferenceCollection = FeatureCollection<Geometry, ReferenceProps>;

export interface DetectionQuery {
  bbox: BBox;
  start: string;
  end: string;
  minConfidence: number;
  classes: ClassId[];
  minArea: number | null;
  maxArea: number | null;
  sceneId?: string | null;
  limit?: number;
}
