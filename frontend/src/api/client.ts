import type {
  BBox,
  DetectionCollection,
  DetectionFeature,
  DetectionQuery,
  Health,
  JobStatus,
  ModelInfo,
  PredictGeotiffResponse,
  ReferenceCollection,
  Scene,
  SearchSceneItem,
} from './types';

/**
 * API base. Empty string (default) means relative URLs (`/api/...`) which the
 * Vite dev server / nginx proxy to the backend. VITE_API_URL may override it
 * with an absolute origin such as `https://api.example.org`.
 */
const RAW_BASE: string = (import.meta.env.VITE_API_URL as string | undefined) ?? '';
export const API_BASE = RAW_BASE.trim().replace(/\/+$/, '');

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

/** Absolute URL (needed by MapLibre image sources, which resolve URLs off the page). */
export function absoluteApiUrl(path: string): string {
  return new URL(apiUrl(path), window.location.href).href;
}

export class ApiError extends Error {
  readonly status: number;
  readonly path: string;
  constructor(message: string, status: number, path: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.path = path;
  }
  /** True when the backend could not be reached at all (network error / proxy error). */
  get unreachable(): boolean {
    return this.status === 0 || this.status === 502 || this.status === 503 || this.status === 504;
  }
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError';
}

function describeDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg, type}]
    return detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : JSON.stringify(d)))
      .join('; ');
  }
  return JSON.stringify(detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, headers: { Accept: 'application/json', ...(init?.headers ?? {}) } });
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError('Cannot reach the API server. Is the backend running?', 0, path);
  }
  const ct = res.headers.get('content-type') ?? '';
  const isJson = ct.includes('json');
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`.trim();
    if (isJson) {
      try {
        const body = (await res.json()) as { detail?: unknown };
        if (body && body.detail !== undefined) msg = `${res.status}: ${describeDetail(body.detail)}`;
      } catch {
        /* keep status text */
      }
    } else if (res.status === 500 && !isJson) {
      // Vite's dev proxy answers 500 with an empty body when the backend is down.
      throw new ApiError('Cannot reach the API server (proxy error). Is the backend running?', 502, path);
    }
    throw new ApiError(msg, res.status, path);
  }
  if (!isJson) {
    throw new ApiError(`Unexpected non-JSON response from ${path}`, res.status, path);
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | null | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === '') continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : '';
}

const fmtCoord = (n: number) => Number(n.toFixed(5)).toString();
export const bboxParam = (b: BBox) => b.map(fmtCoord).join(',');

export interface Place {
  name: string;
  type: string;
  bbox: BBox;
}

export interface SourceCandidate {
  source_type: string | null;
  mmsi_or_structure_id: string | null;
  score: number | null;
  rank: number | null;
  cerulean_slick_id: number | null;
  cerulean_source_url: string | null;
  vessel_lookup_url: string | null;
}
export interface SourcesResponse {
  detection_id: number;
  matched_cerulean_slicks: { slick_id: number; iou: number; slick_url: string | null }[];
  sources: SourceCandidate[];
  note: string | null;
}

export const api = {
  geocode: (q: string, signal?: AbortSignal) =>
    request<{ results: Place[] }>(`/api/geocode?q=${encodeURIComponent(q)}`, { signal }),
  sources: (id: number | string, signal?: AbortSignal) =>
    request<SourcesResponse>(`/api/detections/${encodeURIComponent(String(id))}/sources`, { signal }),

  health: (signal?: AbortSignal) => request<Health>('/api/health', { signal }),

  modelInfo: (signal?: AbortSignal) => request<ModelInfo>('/api/model-info', { signal }),

  detections: (q: DetectionQuery, signal?: AbortSignal) =>
    request<DetectionCollection>(
      `/api/detections${qs({
        bbox: bboxParam(q.bbox),
        start: q.start,
        end: q.end,
        min_confidence: q.minConfidence,
        classes: q.classes.join(','),
        min_area_km2: q.minArea,
        max_area_km2: q.maxArea,
        scene_id: q.sceneId ?? null,
        limit: q.limit ?? 2000,
      })}`,
      { signal },
    ),

  detection: (id: number | string, signal?: AbortSignal) =>
    request<DetectionFeature>(`/api/detections/${encodeURIComponent(String(id))}`, { signal }),

  scenes: (p: { bbox?: BBox; start?: string; end?: string }, signal?: AbortSignal) =>
    request<{ scenes: Scene[] }>(
      `/api/scenes${qs({ bbox: p.bbox ? bboxParam(p.bbox) : null, start: p.start, end: p.end })}`,
      { signal },
    ),

  sceneOverlay: (sceneId: string, signal?: AbortSignal) =>
    request<{ bounds: BBox; image_bounds: BBox }>(`/api/scenes/${encodeURIComponent(sceneId)}/overlay`, { signal }),
  quicklookUrl: (sceneId: string) => absoluteApiUrl(`/api/scenes/${encodeURIComponent(sceneId)}/quicklook.png`),
  probUrl: (sceneId: string) => absoluteApiUrl(`/api/scenes/${encodeURIComponent(sceneId)}/prob.png`),

  processScene: (sceneId: string) =>
    request<{ job_id: string }>('/api/process-scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: sceneId }),
    }),

  job: (jobId: string, signal?: AbortSignal) =>
    request<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal }),

  predictGeotiff: (file: File, signal?: AbortSignal) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return request<PredictGeotiffResponse>('/api/predict-geotiff', { method: 'POST', body: fd, signal });
  },

  searchScenes: (p: { bbox: BBox; start: string; end: string; limit?: number }, signal?: AbortSignal) =>
    request<{ items: SearchSceneItem[] }>(
      `/api/search-scenes${qs({ bbox: bboxParam(p.bbox), start: p.start, end: p.end, limit: p.limit ?? 20 })}`,
      { signal },
    ),

  referenceSlicks: (
    p: { bbox: BBox; start: string; end: string; minConfidence: number; limit?: number },
    signal?: AbortSignal,
  ) =>
    request<ReferenceCollection>(
      `/api/reference/cerulean${qs({
        bbox: bboxParam(p.bbox),
        start: p.start,
        end: p.end,
        min_confidence: p.minConfidence,
        limit: p.limit ?? 500,
      })}`,
      { signal },
    ),
};

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
