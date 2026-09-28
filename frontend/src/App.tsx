import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useMatch, useNavigate } from 'react-router-dom';
import { api, ApiError, errorMessage, isAbortError } from './api/client';
import type {
  BBox,
  ClassId,
  DetectionCollection,
  DetectionFeature,
  Health,
  PredictGeotiffResponse,
  ReferenceCollection,
} from './api/types';
import { DetailPanel } from './components/DetailPanel';
import { DetectionList } from './components/DetectionList';
import { FilterPanel } from './components/FilterPanel';
import { Header } from './components/Header';
import { MapView, type ImageOverlay, type MapHandle } from './components/MapView';
import { ProcessPanel, type TrackedJob } from './components/ProcessPanel';
import { UploadDialog } from './components/UploadDialog';
import { Banners, type Banner } from './components/Banners';
import { Legend } from './components/Legend';
import { useDebounced } from './hooks/useDebounced';
import { parseView, useUrlState, type MapView as MapViewState } from './hooks/useUrlState';
import { addDays, DEFAULT_CLASS_NAMES, fmtNum } from './lib/format';
import { emptyCollection, featuresBBox, isValidBBox } from './lib/geo';

const DETECTION_LIMIT = 2000;
const QUICKLOOK_MAX = 12;
const HEALTH_POLL_MS = 30_000;
const JOB_POLL_MS = 2_000;

type Tab = 'filters' | 'list' | 'process';

function isFeatureCollection(x: unknown): x is DetectionCollection {
  return !!x && typeof x === 'object' && (x as { type?: unknown }).type === 'FeatureCollection' &&
    Array.isArray((x as { features?: unknown }).features);
}

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const slickMatch = useMatch('/slicks/:id');
  const routeId = slickMatch?.params.id ?? null;

  const { filters, setFilters, setView, searchParams } = useUrlState();
  const [initialView] = useState<MapViewState>(() => parseView(searchParams));
  const mapRef = useRef<MapHandle>(null);

  // ---------- layout ----------
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [tab, setTab] = useState<Tab>('filters');
  const [uploadOpen, setUploadOpen] = useState(false);

  // ---------- health / model ----------
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [healthTick, setHealthTick] = useState(0);
  const [classNames, setClassNames] = useState<Record<ClassId, string>>(DEFAULT_CLASS_NAMES);

  useEffect(() => {
    const ctrl = new AbortController();
    const load = () =>
      api
        .health(ctrl.signal)
        .then((h) => {
          setHealth(h);
          setHealthError(null);
        })
        .catch((e) => {
          if (isAbortError(e)) return;
          setHealth(null);
          setHealthError(errorMessage(e));
        });
    load();
    const t = window.setInterval(load, HEALTH_POLL_MS);
    return () => {
      ctrl.abort();
      window.clearInterval(t);
    };
  }, [healthTick]);

  const modelLoaded = health?.model_loaded ?? false;
  useEffect(() => {
    const ctrl = new AbortController();
    api
      .modelInfo(ctrl.signal)
      .then((info) => {
        if (!info?.classes) return;
        const next = { ...DEFAULT_CLASS_NAMES };
        for (const [k, v] of Object.entries(info.classes)) {
          const n = Number(k);
          // The API returns short codes (INFRA, NATURAL, VESSEL); keep the readable UI names for those.
          if ((n === 1 || n === 2 || n === 3) && typeof v === 'string' && v && v !== v.toUpperCase()) next[n] = v;
        }
        setClassNames(next);
      })
      .catch(() => {
        /* names fall back to the contract defaults; status is shown by the header badge */
      });
    return () => ctrl.abort();
  }, [modelLoaded]);

  // ---------- map view ----------
  const [viewBBox, setViewBBox] = useState<BBox | null>(null);
  const onViewChange = useCallback(
    (bbox: BBox, view: MapViewState) => {
      setViewBBox(bbox);
      setView(view);
    },
    [setView],
  );

  // ---------- detections ----------
  const [refreshTick, setRefreshTick] = useState(0);
  const queryKey = useMemo(
    () => (viewBBox ? JSON.stringify({ viewBBox, filters, refreshTick }) : null),
    [viewBBox, filters, refreshTick],
  );
  const debouncedKey = useDebounced(queryKey, 350);

  const [detections, setDetections] = useState<DetectionCollection>(() => emptyCollection());
  const [detLoading, setDetLoading] = useState(false);
  const [detError, setDetError] = useState<string | null>(null);
  const [detLoaded, setDetLoaded] = useState(false);

  useEffect(() => {
    if (!debouncedKey || !viewBBox) return;
    if (filters.classes.length === 0) {
      setDetections(emptyCollection());
      setDetError(null);
      setDetLoaded(true);
      return;
    }
    const ctrl = new AbortController();
    setDetLoading(true);
    api
      .detections(
        {
          bbox: viewBBox,
          start: filters.start,
          end: filters.end,
          minConfidence: filters.minConfidence,
          classes: filters.classes,
          minArea: filters.minArea,
          maxArea: filters.maxArea,
          limit: DETECTION_LIMIT,
        },
        ctrl.signal,
      )
      .then((fc) => {
        if (!isFeatureCollection(fc)) throw new Error('Malformed /api/detections response (expected a FeatureCollection)');
        setDetections(fc);
        setDetError(null);
        setDetLoaded(true);
      })
      .catch((e) => {
        if (isAbortError(e)) return;
        setDetError(errorMessage(e));
        setDetLoaded(true);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setDetLoading(false);
      });
    return () => ctrl.abort();
    // debouncedKey captures viewBBox + filters + refreshTick
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  // ---------- reference (global slick archive) ----------
  const [reference, setReference] = useState<ReferenceCollection | null>(null);
  const [refStatus, setRefStatus] = useState<string | null>(null);
  const [refError, setRefError] = useState<string | null>(null);
  const refKey = useDebounced(
    null, // replaced by global vector tiles (referenceTilesUrl): fast at every zoom
    500,
  );
  useEffect(() => {
    if (!filters.showReference) {
      setReference(null);
      setRefStatus(null);
      setRefError(null);
      return;
    }
    setRefStatus('Global slick archive. Click a blue-grey slick to see nearby ships.');
    if (!refKey || !viewBBox) return;
    const ctrl = new AbortController();
    setRefStatus('Loading archive slicks…');
    api
      .referenceSlicks(
        { bbox: viewBBox, start: filters.start, end: filters.end, minConfidence: filters.minConfidence, limit: 500 },
        ctrl.signal,
      )
      .then((fc) => {
        if (!isFeatureCollection(fc)) throw new Error('Malformed reference response');
        setReference(fc as unknown as ReferenceCollection);
        setRefError(null);
        const n = fc.features.length;
        setRefStatus(
          n === 0
            ? 'No archive slicks for this view and date range.'
            : `${n}${n >= 500 ? '+' : ''} archive slick${n === 1 ? '' : 's'} in view.`,
        );
      })
      .catch((e) => {
        if (isAbortError(e)) return;
        setReference(null);
        setRefStatus(null);
        setRefError(`Slick archive unavailable: ${errorMessage(e)}`);
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refKey, filters.showReference]);

  // ---------- scene bounds cache + overlays ----------
  const sceneBounds = useRef(new Map<string, BBox>());
  const [manualOverlays, setManualOverlays] = useState<ImageOverlay[]>([]);
  const [autoOverlays, setAutoOverlays] = useState<ImageOverlay[]>([]);
  const [qlStatus, setQlStatus] = useState<string | null>(null);
  const [qlError, setQlError] = useState<string | null>(null);

  /** PNG overlays are Web-Mercator images: place them by image_bounds, not the lat/lon scene bounds. */
  const overlayBBox = (s: { bounds: unknown; image_bounds?: unknown }): BBox | null =>
    isValidBBox(s.image_bounds) ? s.image_bounds : isValidBBox(s.bounds) ? s.bounds : null;
  const cacheScenes = (scenes: { scene_id: string; bounds: unknown; image_bounds?: unknown }[]) => {
    for (const s of scenes) {
      const b = s && typeof s.scene_id === 'string' ? overlayBBox(s) : null;
      if (b) sceneBounds.current.set(s.scene_id, b);
    }
  };

  const qlKey = useDebounced(
    filters.showQuicklooks && viewBBox ? JSON.stringify([viewBBox, filters.start, filters.end, refreshTick]) : null,
    500,
  );
  useEffect(() => {
    if (!filters.showQuicklooks) {
      setAutoOverlays([]);
      setQlStatus(null);
      setQlError(null);
      return;
    }
    if (!qlKey || !viewBBox) return;
    const ctrl = new AbortController();
    setQlStatus('Loading processed scenes…');
    api
      .scenes({ bbox: viewBBox, start: filters.start, end: filters.end }, ctrl.signal)
      .then((res) => {
        const scenes = Array.isArray(res?.scenes) ? res.scenes : [];
        cacheScenes(scenes);
        const valid = scenes
          .filter((s) => isValidBBox(s.bounds))
          .sort((a, b) => (b.acquired_at ?? '').localeCompare(a.acquired_at ?? ''));
        const picked = valid.slice(0, QUICKLOOK_MAX);
        setAutoOverlays(
          picked.map((s) => ({ key: `sar:${s.scene_id}`, kind: 'sar', sceneId: s.scene_id, url: api.quicklookUrl(s.scene_id), bounds: overlayBBox(s) ?? s.bounds })),
        );
        setQlError(null);
        setQlStatus(
          valid.length === 0
            ? 'No processed scenes in this view and date range.'
            : valid.length > QUICKLOOK_MAX
              ? `Showing the ${QUICKLOOK_MAX} most recent of ${valid.length} processed scenes in view.`
              : `Showing ${valid.length} processed scene${valid.length === 1 ? '' : 's'} in view.`,
        );
      })
      .catch((e) => {
        if (isAbortError(e)) return;
        setAutoOverlays([]);
        setQlStatus(null);
        setQlError(`Scene quicklooks unavailable: ${errorMessage(e)}`);
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qlKey, filters.showQuicklooks]);

  const overlays = useMemo(() => {
    const seen = new Set<string>();
    const out: ImageOverlay[] = [];
    for (const o of [...autoOverlays, ...manualOverlays]) {
      if (seen.has(o.key)) continue;
      seen.add(o.key);
      out.push(o);
    }
    return out;
  }, [autoOverlays, manualOverlays]);

  /** Scene bounds are not part of the detection payload; look them up via /api/scenes. */
  const resolveSceneBounds = useCallback(async (sceneId: string, hint: DetectionFeature | null): Promise<BBox | null> => {
    const cached = sceneBounds.current.get(sceneId);
    if (cached) return cached;
    try {
      const ov = await api.sceneOverlay(sceneId);
      if (isValidBBox(ov?.image_bounds)) {
        sceneBounds.current.set(sceneId, ov.image_bounds);
        return ov.image_bounds;
      }
    } catch {
      /* fall through to the /api/scenes lookup */
    }
    const p = hint?.properties;
    const day = p?.acquired_at ? p.acquired_at.slice(0, 10) : null;
    const hintBBox = p?.bbox && isValidBBox(p.bbox) ? p.bbox : hint ? featuresBBox([hint]) : null;
    const res = await api.scenes({
      bbox: hintBBox ?? undefined,
      start: day ? addDays(day, -1) : undefined,
      end: day ? addDays(day, 1) : undefined,
    });
    cacheScenes(Array.isArray(res?.scenes) ? res.scenes : []);
    return sceneBounds.current.get(sceneId) ?? null;
  }, []);

  // ---------- selection ----------
  const [selected, setSelected] = useState<DetectionFeature | null>(null);
  const [selLoading, setSelLoading] = useState(false);
  const [selError, setSelError] = useState<string | null>(null);
  const flownFor = useRef<string | null>(null);

  const [upload, setUpload] = useState<{ sceneId: string; fc: DetectionCollection } | null>(null);
  const [selectedUploadIndex, setSelectedUploadIndex] = useState<number | null>(null);

  const selectDetection = useCallback(
    (id: number) => {
      setSelectedUploadIndex(null);
      navigate({ pathname: `/slicks/${id}`, search: window.location.search });
    },
    [navigate],
  );

  const clearSelection = useCallback(() => {
    setSelectedUploadIndex(null);
    if (routeId !== null) navigate({ pathname: '/', search: window.location.search });
  }, [navigate, routeId]);

  useEffect(() => {
    if (routeId === null) {
      setSelected(null);
      setSelError(null);
      setSelLoading(false);
      flownFor.current = null;
      return;
    }
    if (!/^\d+$/.test(routeId)) {
      setSelected(null);
      setSelError(`"${routeId}" is not a valid detection id.`);
      return;
    }
    const ctrl = new AbortController();
    setSelLoading(true);
    setSelError(null);
    // Show the already-loaded feature immediately if we have it.
    const local = detections.features.find((f) => Number(f.properties?.id ?? f.id) === Number(routeId));
    if (local) setSelected(local);
    api
      .detection(routeId, ctrl.signal)
      .then((f) => {
        if (!f || f.type !== 'Feature') throw new Error('Malformed /api/detections/{id} response');
        setSelected(f);
        if (flownFor.current !== routeId) {
          flownFor.current = routeId;
          const b = f.properties?.bbox && isValidBBox(f.properties.bbox) ? f.properties.bbox : featuresBBox([f]);
          if (b) mapRef.current?.fitBBox(b, { maxZoom: 11, padding: 120 });
        }
      })
      .catch((e) => {
        if (isAbortError(e)) return;
        setSelected(null);
        setSelError(
          e instanceof ApiError && e.status === 404 ? `Detection #${routeId} was not found.` : `Could not load detection #${routeId}: ${errorMessage(e)}`,
        );
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setSelLoading(false);
      });
    return () => ctrl.abort();
    // detections intentionally omitted: only refetch when the route id changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeId]);

  const uploadSelected: DetectionFeature | null =
    selectedUploadIndex !== null && upload ? upload.fc.features[selectedUploadIndex] ?? null : null;
  const detailFeature = uploadSelected ?? (routeId !== null ? selected : null);
  const detailOpen = uploadSelected !== null || routeId !== null;
  const selectedId = routeId !== null && /^\d+$/.test(routeId) ? Number(routeId) : null;

  // ---------- overlays for selected detection ----------
  const [overlayBusy, setOverlayBusy] = useState<'sar' | 'prob' | null>(null);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  useEffect(() => setOverlayError(null), [detailFeature]);
  const detailScene = detailFeature?.properties?.scene_id ?? (uploadSelected ? upload?.sceneId ?? null : null);
  const sarActive = !!detailScene && overlays.some((o) => o.key === `sar:${detailScene}`);
  const probActive = !!detailScene && overlays.some((o) => o.key === `prob:${detailScene}`);

  const toggleOverlay = async (kind: 'sar' | 'prob') => {
    if (!detailScene) return;
    const key = `${kind}:${detailScene}`;
    if (manualOverlays.some((o) => o.key === key)) {
      setManualOverlays((ov) => ov.filter((o) => o.key !== key));
      return;
    }
    if (kind === 'sar' && autoOverlays.some((o) => o.key === key)) {
      // Shown by the global quicklook toggle.
      setOverlayError('This scene’s SAR image is shown by the “Scene SAR quicklooks” layer toggle.');
      return;
    }
    setOverlayBusy(kind);
    setOverlayError(null);
    try {
      const bounds = await resolveSceneBounds(detailScene, detailFeature);
      if (!bounds) {
        setOverlayError(`Scene bounds for ${detailScene} are not available from /api/scenes.`);
        return;
      }
      const url = kind === 'sar' ? api.quicklookUrl(detailScene) : api.probUrl(detailScene);
      // Verify the image exists so we can report an honest error instead of a silent blank.
      const probe = await fetch(url, { method: 'GET', cache: 'force-cache' });
      if (!probe.ok) {
        setOverlayError(`${kind === 'sar' ? 'SAR quicklook' : 'Probability heatmap'} not available (${probe.status}).`);
        return;
      }
      setManualOverlays((ov) => [...ov, { key, kind, sceneId: detailScene, url, bounds }]);
    } catch (e) {
      setOverlayError(errorMessage(e));
    } finally {
      setOverlayBusy(null);
    }
  };

  const zoomToDetail = () => {
    if (!detailFeature) return;
    const b = detailFeature.properties?.bbox && isValidBBox(detailFeature.properties.bbox) ? detailFeature.properties.bbox : featuresBBox([detailFeature]);
    if (b) mapRef.current?.fitBBox(b, { maxZoom: 12, padding: 120 });
  };

  // ---------- upload ----------
  const onUploadResult = (res: PredictGeotiffResponse) => {
    setUpload({ sceneId: res.scene_id, fc: res.detections });
    setSelectedUploadIndex(null);
    if (routeId !== null) navigate({ pathname: '/', search: window.location.search });
    const b = featuresBBox(res.detections.features);
    if (b) mapRef.current?.fitBBox(b, { maxZoom: 11, padding: 100 });
    setUploadNotice(
      res.detections.features.length === 0
        ? `Model ran on ${res.scene_id}: no oil slicks detected.`
        : `Model found ${res.detections.features.length} slick${res.detections.features.length === 1 ? '' : 's'} in ${res.scene_id} (temporary layer).`,
    );
    setHealthTick((t) => t + 1);
    setRefreshTick((t) => t + 1);
  };
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);

  // ---------- processing jobs ----------
  const [jobs, setJobs] = useState<TrackedJob[]>([]);
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const runScene = useCallback(async (sceneId: string) => {
    const { job_id } = await api.processScene(sceneId);
    setJobs((js) => [{ jobId: job_id, sceneId, status: null, error: null }, ...js]);
  }, []);

  const hasActiveJobs = jobs.some((j) => !j.status || j.status.status === 'queued' || j.status.status === 'running');
  useEffect(() => {
    if (!hasActiveJobs) return;
    let cancelled = false;
    const tick = async () => {
      const active = jobsRef.current.filter((j) => !j.status || j.status.status === 'queued' || j.status.status === 'running');
      const results = await Promise.all(
        active.map(async (j) => {
          try {
            return { jobId: j.jobId, status: await api.job(j.jobId), error: null as string | null };
          } catch (e) {
            return { jobId: j.jobId, status: null, error: errorMessage(e) };
          }
        }),
      );
      if (cancelled) return;
      const finished = results.some(
        (r) => r.status?.status === 'done' && jobsRef.current.find((j) => j.jobId === r.jobId)?.status?.status !== 'done',
      );
      setJobs((js) =>
        js.map((j) => {
          const r = results.find((x) => x.jobId === j.jobId);
          if (!r) return j;
          return r.status ? { ...j, status: r.status, error: null } : { ...j, error: r.error };
        }),
      );
      if (finished) {
        setRefreshTick((t) => t + 1);
        setHealthTick((t) => t + 1);
      }
    };
    const t = window.setInterval(tick, JOB_POLL_MS);
    void tick();
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [hasActiveJobs]);

  // Stop polling jobs that 404 repeatedly: mark as error.
  useEffect(() => {
    setJobs((js) => {
      let changed = false;
      const next = js.map((j) => {
        if (!j.status && j.error && /^404/.test(j.error)) {
          changed = true;
          return { ...j, status: { job_id: j.jobId, status: 'error' as const, message: 'Job not found on server', scene_id: j.sceneId, n_detections: null } };
        }
        return j;
      });
      return changed ? next : js;
    });
  }, [jobs]);

  const [footprint, setFootprint] = useState<BBox | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  // ---------- banners ----------
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const banners: Banner[] = [];
  if (healthError) banners.push({ id: `health:${healthError}`, level: 'error', text: `API unavailable — ${healthError}` });
  if (detError && !healthError) banners.push({ id: `det:${detError}`, level: 'error', text: `Could not load detections — ${detError}` });
  if (refError && !healthError) banners.push({ id: `ref:${refError}`, level: 'warn', text: refError });
  if (qlError && !healthError) banners.push({ id: `ql:${qlError}`, level: 'warn', text: qlError });
  if (health && !health.model_loaded)
    banners.push({
      id: 'nomodel',
      level: 'warn',
      text: 'No model is loaded on the server — existing detections can be browsed, but new scenes cannot be processed.',
    });
  if (uploadNotice) banners.push({ id: `up:${uploadNotice}`, level: 'info', text: uploadNotice });
  const visibleBanners = banners.filter((b) => !dismissed.has(b.id));

  const features = detections.features;
  const truncated = features.length >= DETECTION_LIMIT;
  const showEmpty = detLoaded && !detLoading && !detError && !healthError && features.length === 0;
  const dbEmpty = health?.db_detections === 0;

  // Close sidebar on mobile when selecting from the list.
  const selectFromList = (id: number) => {
    selectDetection(id);
    if (window.innerWidth <= 760) setSidebarOpen(false);
  };

  useEffect(() => {
    // Keep document title informative.
    document.title = routeId ? `Slick #${routeId} · Ocean Police` : 'Ocean Police';
  }, [routeId]);

  return (
    <div className={`app ${sidebarOpen ? 'sidebar-open' : ''} ${detailOpen ? 'detail-open' : ''}`}>
      <Header
        health={health}
        healthError={healthError}
        onUploadClick={() => setUploadOpen(true)}
        onToggleSidebar={() => setSidebarOpen((o) => !o)}
        onPlace={(b) => mapRef.current?.fitBBox(b, { maxZoom: 9, padding: 40 })}
      />
      <div className="main">
        <aside className="sidebar" aria-label="Filters and detections">
          <div className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === 'filters'} className={tab === 'filters' ? 'active' : ''} onClick={() => setTab('filters')}>
              Filters
            </button>
            <button role="tab" aria-selected={tab === 'list'} className={tab === 'list' ? 'active' : ''} onClick={() => setTab('list')}>
              Detections <span className="count">{truncated ? `${DETECTION_LIMIT}+` : features.length}</span>
            </button>
            <button role="tab" aria-selected={tab === 'process'} className={tab === 'process' ? 'active' : ''} onClick={() => setTab('process')}>
              Process
              {hasActiveJobs && <span className="pulse" aria-label="job running" />}
            </button>
          </div>
          <div className="tab-body">
            {tab === 'filters' && (
              <FilterPanel
                filters={filters}
                classNames={classNames}
                onChange={setFilters}
                referenceStatus={filters.showReference ? refStatus : null}
                quicklookStatus={filters.showQuicklooks ? qlStatus : null}
              />
            )}
            {tab === 'list' && (
              <DetectionList
                features={features}
                classNames={classNames}
                selectedId={selectedId}
                loading={detLoading}
                truncated={truncated}
                onSelect={selectFromList}
                onHover={setHoveredId}
              />
            )}
            {tab === 'process' && (
              <ProcessPanel
                viewBBox={viewBBox}
                start={filters.start}
                end={filters.end}
                jobs={jobs}
                onRun={runScene}
                onHoverScene={setFootprint}
                onZoomScene={(b) => mapRef.current?.fitBBox(b, { maxZoom: 9, padding: 40 })}
                onDismissJob={(id) => setJobs((js) => js.filter((j) => j.jobId !== id))}
              />
            )}
          </div>
          {upload && (
            <div className="upload-strip">
              <span className="small">
                Upload layer: {upload.fc.features.length} polygon{upload.fc.features.length === 1 ? '' : 's'}
              </span>
              <button
                className="mini-btn"
                onClick={() => {
                  setUpload(null);
                  setSelectedUploadIndex(null);
                }}
              >
                Clear
              </button>
            </div>
          )}
        </aside>

        <div className="map-wrap">
          {detLoading && <div className="loading-bar" aria-label="Loading detections" />}
          <Banners banners={visibleBanners} onDismiss={(id) => setDismissed((s) => new Set(s).add(id))} />
          <MapView
            ref={mapRef}
            initialView={initialView}
            detections={detections}
            uploadDetections={upload?.fc ?? null}
            reference={filters.showReference ? reference : null}
            referenceTilesUrl={
              filters.showReference
                ? `${window.location.origin}/api/reference/tiles/{z}/{x}/{y}.pbf?start=${filters.start}&end=${filters.end}&min_confidence=${filters.minConfidence}`
                : null
            }
            overlays={overlays}
            footprint={footprint}
            selectedId={selectedId}
            selectedUploadIndex={selectedUploadIndex}
            hoveredId={hoveredId}
            onViewChange={onViewChange}
            onSelectDetection={selectDetection}
            onSelectUpload={(i) => {
              if (routeId !== null) navigate({ pathname: '/', search: window.location.search });
              setSelectedUploadIndex(i);
            }}
            onClearSelection={clearSelection}
          />
          {showEmpty && (
            <div className="empty-state" role="status">
              {dbEmpty ? (
                <>
                  <strong>No detections yet — the model has not processed any scenes.</strong>
                  <span>
                    Use <em>Process</em> to run the model on a Sentinel-1 scene, or <em>Upload GeoTIFF</em>.
                  </span>
                </>
              ) : filters.classes.length === 0 ? (
                <strong>No source classes selected.</strong>
              ) : (
                <>
                  <strong>No detections in this view yet.</strong>
                  {filters.showReference && <span>Blue-grey shapes are from the slick archive.</span>}
                  {health && <span>{fmtNum(health.db_detections, 0)} detections exist in the database overall — try zooming out or widening the date range.</span>}
                </>
              )}
            </div>
          )}
          <Legend
            classNames={classNames}
            count={features.length}
            truncated={truncated}
            showReference={filters.showReference}
            hasUpload={!!upload}
            overlays={overlays.length}
            onClearOverlays={manualOverlays.length ? () => setManualOverlays([]) : undefined}
          />
        </div>

        {detailOpen && (
          <DetailPanel
            requestedId={routeId}
            feature={detailFeature}
            isUpload={uploadSelected !== null}
            loading={selLoading && !detailFeature}
            error={uploadSelected ? null : selError}
            classNames={classNames}
            sarActive={sarActive}
            probActive={probActive}
            overlayBusy={overlayBusy}
            overlayError={overlayError}
            onToggleSar={() => void toggleOverlay('sar')}
            onToggleProb={() => void toggleOverlay('prob')}
            onZoom={zoomToDetail}
            onClose={() => {
              setSelectedUploadIndex(null);
              if (routeId !== null) navigate({ pathname: '/', search: location.search });
            }}
          />
        )}
      </div>
      <footer className="app-footer">
        <span className="footer-brand">Ocean Police</span>
      </footer>
      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} onResult={onUploadResult} />
    </div>
  );
}
