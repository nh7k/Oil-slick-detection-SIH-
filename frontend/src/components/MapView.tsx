import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import maplibregl, {
  type GeoJSONSource,
  type ImageSource,
  type LngLatBoundsLike,
  type Map as MLMap,
  type MapGeoJSONFeature,
  type StyleSpecification,
} from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { FeatureCollection, Geometry } from 'geojson';
import type { BBox, DetectionCollection, ReferenceCollection } from '../api/types';
import { CLASS_COLORS, fmtDateTime, fmtNum, fmtPct } from '../lib/format';
import { bboxToImageCorners, bboxToPolygon, clampBBox, emptyCollection, featureCollection } from '../lib/geo';
import type { MapView as MapViewState } from '../hooks/useUrlState';

export interface ImageOverlay {
  key: string; // unique, e.g. "sar:SCENE"
  kind: 'sar' | 'prob';
  sceneId: string;
  url: string;
  bounds: BBox;
}

export interface MapHandle {
  fitBBox: (b: BBox, opts?: { maxZoom?: number; padding?: number }) => void;
}

interface Props {
  initialView: MapViewState;
  detections: DetectionCollection;
  uploadDetections: DetectionCollection | null;
  reference: ReferenceCollection | null;
  /** Mapbox-vector-tile URL template for the global slick archive, or null when hidden. */
  referenceTilesUrl: string | null;
  overlays: ImageOverlay[];
  footprint: BBox | null;
  selectedId: number | null;
  /** Index into uploadDetections.features (source uses generateId). */
  selectedUploadIndex: number | null;
  hoveredId: number | null;
  onViewChange: (bbox: BBox, view: MapViewState) => void;
  onSelectDetection: (id: number) => void;
  onSelectUpload: (index: number) => void;
  onClearSelection: () => void;
}

const OSM_ATTRIBUTION = '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';

const STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      maxzoom: 19,
      attribution: OSM_ATTRIBUTION,
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#14181d' } },
    {
      id: 'osm',
      type: 'raster',
      source: 'osm',
      // Calm dark basemap from standard OSM tiles: fully desaturate, then invert brightness
      // (min > max) so the light sea becomes dark grey and land a slightly lighter grey.
      paint: {
        'raster-saturation': -0.75,
        'raster-hue-rotate': 180, // inverted OSM water turns a quiet slate blue; land stays neutral grey
        'raster-brightness-min': 0.9,
        'raster-brightness-max': 0.12,
        'raster-contrast': -0.15,
      },
    },
  ],
};

const classColorExpr = [
  'match',
  ['to-number', ['get', 'cls']],
  1,
  CLASS_COLORS[1],
  2,
  CLASS_COLORS[2],
  3,
  CLASS_COLORS[3],
  '#8a939d',
] as unknown as maplibregl.ExpressionSpecification;

const FIRST_DATA_LAYER = 'footprint-fill';

/** One point per feature (bbox centre), for the zoomed-out dot layers. */
function centroidPoints(fc: { features?: { geometry?: unknown; properties?: unknown }[] } | null | undefined) {
  const out: { type: 'Feature'; geometry: { type: 'Point'; coordinates: [number, number] }; properties: unknown }[] = [];
  for (const f of fc?.features ?? []) {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    const walk = (c: unknown): void => {
      if (Array.isArray(c) && typeof c[0] === 'number' && typeof c[1] === 'number') {
        minx = Math.min(minx, c[0]); maxx = Math.max(maxx, c[0]);
        miny = Math.min(miny, c[1]); maxy = Math.max(maxy, c[1]);
      } else if (Array.isArray(c)) c.forEach(walk);
    };
    walk((f.geometry as { coordinates?: unknown } | undefined)?.coordinates);
    if (Number.isFinite(minx)) {
      out.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [(minx + maxx) / 2, (miny + maxy) / 2] }, properties: f.properties ?? {} });
    }
  }
  return { type: 'FeatureCollection', features: out } as unknown as GeoJSON.FeatureCollection;
}

function addLayers(map: MLMap) {
  const empty = emptyCollection();
  map.addSource('footprint', { type: 'geojson', data: empty });
  map.addSource('reference', { type: 'geojson', data: empty });
  map.addSource('detections', { type: 'geojson', data: empty, promoteId: 'id' });
  map.addSource('upload', { type: 'geojson', data: empty, generateId: true });
  map.addSource('det-points', { type: 'geojson', data: empty });
  map.addSource('ref-points', { type: 'geojson', data: empty });

  map.addLayer({
    id: 'footprint-fill',
    type: 'fill',
    source: 'footprint',
    paint: { 'fill-color': '#5fa8b8', 'fill-opacity': 0.06 },
  });
  map.addLayer({
    id: 'footprint-line',
    type: 'line',
    source: 'footprint',
    paint: { 'line-color': '#5fa8b8', 'line-width': 1.2, 'line-dasharray': [4, 3] },
  });

  map.addLayer({
    id: 'reference-fill',
    type: 'fill',
    source: 'reference',
    paint: { 'fill-color': '#8fb3cc', 'fill-opacity': 0.12 },
  });
  map.addLayer({
    id: 'reference-line',
    type: 'line',
    source: 'reference',
    paint: { 'line-color': '#8fb3cc', 'line-width': 1.4, 'line-dasharray': [2, 2] },
  });

  map.addLayer({
    id: 'det-fill',
    type: 'fill',
    source: 'detections',
    paint: {
      'fill-color': classColorExpr,
      'fill-opacity': ['case', ['boolean', ['feature-state', 'hover'], false], 0.7, 0.4],
    },
  });
  map.addLayer({
    id: 'det-line',
    type: 'line',
    source: 'detections',
    paint: {
      'line-color': classColorExpr,
      'line-width': ['case', ['boolean', ['feature-state', 'hover'], false], 2.2, 1.2],
    },
  });
  map.addLayer({
    id: 'det-selected',
    type: 'line',
    source: 'detections',
    filter: ['==', ['to-number', ['get', 'id']], -1],
    paint: { 'line-color': '#e6e9ec', 'line-width': 2.5 },
  });

  // Zoomed out, slick polygons are sub-pixel: show our detections as clear dots instead.
  map.addLayer({
    id: 'det-dots',
    type: 'circle',
    source: 'det-points',
    maxzoom: 6,
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 2, 3.5, 6, 5.5],
      'circle-color': classColorExpr,
      'circle-stroke-color': '#14181d',
      'circle-stroke-width': 1.5,
    },
  });

  map.addLayer({
    id: 'upload-fill',
    type: 'fill',
    source: 'upload',
    paint: { 'fill-color': classColorExpr, 'fill-opacity': 0.55 },
  });
  map.addLayer({
    id: 'upload-line',
    type: 'line',
    source: 'upload',
    paint: { 'line-color': '#d9c77a', 'line-width': 1.8, 'line-dasharray': [1, 1.5] },
  });
  map.addLayer({
    id: 'upload-selected',
    type: 'line',
    source: 'upload',
    filter: ['==', ['id'], -1],
    paint: { 'line-color': '#e6e9ec', 'line-width': 2.5 },
  });
}

function textRow(parent: HTMLElement, label: string, value: string) {
  const row = document.createElement('div');
  row.className = 'popup-row';
  const l = document.createElement('span');
  l.textContent = label;
  const v = document.createElement('span');
  v.textContent = value;
  row.append(l, v);
  parent.appendChild(row);
}

function referencePopupContent(f: MapGeoJSONFeature): HTMLElement {
  const p = f.properties as Record<string, unknown>;
  const el = document.createElement('div');
  el.className = 'ref-popup';
  const h = document.createElement('div');
  h.className = 'popup-title';
  h.textContent = 'Archive slick';
  el.appendChild(h);
  const num = (v: unknown) => (typeof v === 'number' ? v : v == null ? null : Number(v));
  textRow(el, 'Archive ID', String(p.id ?? '—'));
  textRow(el, 'Machine confidence', fmtPct(num(p.machine_confidence)));
  textRow(el, 'Class', String(p.cls ?? '—'));
  textRow(el, 'Timestamp', fmtDateTime(typeof p.slick_timestamp === 'string' ? p.slick_timestamp : null));
  const area = num(p.area);
  textRow(el, 'Area', area == null ? '—' : `${fmtNum(area / 1e6, 3)} km²`);
  textRow(el, 'S1 scene', String(p.s1_scene_id ?? '—'));
  const note = document.createElement('div');
  note.className = 'popup-note';
  note.textContent = 'From the global slick archive, not detected by our model.';
  el.appendChild(note);
  if (p.id != null) {
    const a = document.createElement('a');
    a.className = 'popup-link';
    a.href = `https://cerulean.skytruth.org/slicks/${encodeURIComponent(String(p.id))}`;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = 'Source record ↗';
    el.appendChild(a);
  }
  return el;
}

/** Popup for a global-archive slick: details + candidate ships, loaded on demand. */
function referenceTilePopup(p: Record<string, unknown>): HTMLElement {
  const el = document.createElement('div');
  el.className = 'ref-popup';
  const h = document.createElement('div');
  h.className = 'popup-title';
  h.textContent = `Archive slick #${String(p.id ?? '')}`;
  el.appendChild(h);
  const num = (v: unknown) => (typeof v === 'number' ? v : v == null ? null : Number(v));
  textRow(el, 'Machine confidence', fmtPct(num(p.machine_confidence)));
  textRow(el, 'Detected', fmtDateTime(typeof p.slick_timestamp === 'string' ? p.slick_timestamp : null));
  const area = num(p.area);
  textRow(el, 'Area', area == null ? '—' : `${fmtNum(area / 1e6, 2)} km²`);
  const ships = document.createElement('div');
  ships.className = 'popup-ships';
  ships.textContent = 'Loading ships near this slick…';
  el.appendChild(ships);
  type Src = { source_type: string; mmsi_or_structure_id: string; score: number | null; vessel_lookup_url: string | null };
  fetch(`/api/reference/slick/${encodeURIComponent(String(p.id))}`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((d: { slick: Record<string, unknown>; sources: Src[] }) => {
      ships.textContent = '';
      const t = document.createElement('div');
      t.className = 'popup-subtitle';
      t.textContent = d.sources.length ? 'Possible sources (AIS match, best first)' : 'No candidate ship or platform matched.';
      ships.appendChild(t);
      for (const s of d.sources.slice(0, 5)) {
        const row = document.createElement('div');
        row.className = 'popup-row';
        const l = document.createElement('span');
        l.textContent = s.source_type === 'VESSEL' ? 'Ship MMSI' : s.source_type;
        const v = document.createElement(s.vessel_lookup_url ? 'a' : 'span');
        v.textContent = `${s.mmsi_or_structure_id} (score ${s.score == null ? '—' : s.score.toFixed(2)})`;
        if (s.vessel_lookup_url && v instanceof HTMLAnchorElement) {
          v.href = s.vessel_lookup_url;
          v.target = '_blank';
          v.rel = 'noopener noreferrer';
        }
        row.append(l, v);
        ships.appendChild(row);
      }
      if (typeof d.slick.s1_scene_id === 'string') textRow(ships, 'S1 scene', d.slick.s1_scene_id);
    })
    .catch((e) => {
      ships.textContent = `Ship lookup failed: ${e instanceof Error ? e.message : String(e)}`;
    });
  const note = document.createElement('div');
  note.className = 'popup-note';
  note.textContent = 'From the global slick archive (not our model). A candidate ship is not proof.';
  el.appendChild(note);
  const a = document.createElement('a');
  a.className = 'popup-link';
  a.href = `https://cerulean.skytruth.org/slicks/${encodeURIComponent(String(p.id))}`;
  a.target = '_blank';
  a.rel = 'noopener noreferrer';
  a.textContent = 'Source record ↗';
  el.appendChild(a);
  return el;
}

/** Popup for one of OUR detections: summary + candidate ships from /api/detections/{id}/sources. */
function detectionPopup(p: Record<string, unknown>): HTMLElement {
  const el = document.createElement('div');
  el.className = 'ref-popup';
  const h = document.createElement('div');
  h.className = 'popup-title';
  const niceClass: Record<string, string> = { INFRA: 'Infrastructure', NATURAL: 'Natural seep', VESSEL: 'Vessel' };
  const cls = String(p.cls_name ?? '');
  h.textContent = `${niceClass[cls] ?? (cls || 'Slick')} #${String(p.id ?? '')} · our model`;
  el.appendChild(h);
  const num = (v: unknown) => (typeof v === 'number' ? v : v == null ? null : Number(v));
  textRow(el, 'Confidence', fmtPct(num(p.confidence)));
  textRow(el, 'Detected', fmtDateTime(typeof p.acquired_at === 'string' ? p.acquired_at : null));
  const area = num(p.area_km2);
  textRow(el, 'Area', area == null ? '—' : `${fmtNum(area, 2)} km²`);
  const ships = document.createElement('div');
  ships.className = 'popup-ships';
  ships.textContent = 'Looking up ships near this slick…';
  el.appendChild(ships);
  type Src = { source_type: string | null; mmsi_or_structure_id: string | null; score: number | null; vessel_lookup_url: string | null };
  fetch(`/api/detections/${encodeURIComponent(String(p.id))}/sources`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((d: { sources: Src[]; note: string | null }) => {
      ships.textContent = '';
      const t = document.createElement('div');
      t.className = 'popup-subtitle';
      t.textContent = d.sources.length ? 'Possible sources (AIS match, best first)' : 'Possible sources';
      ships.appendChild(t);
      if (!d.sources.length) {
        const n = document.createElement('div');
        n.className = 'popup-note';
        n.textContent = d.note ?? 'No candidate ship found.';
        ships.appendChild(n);
      }
      for (const s of d.sources.slice(0, 5)) {
        const row = document.createElement('div');
        row.className = 'popup-row';
        const l = document.createElement('span');
        l.textContent = s.source_type === 'VESSEL' ? 'Ship MMSI' : String(s.source_type ?? 'Source');
        const v = document.createElement(s.vessel_lookup_url ? 'a' : 'span');
        v.textContent = `${s.mmsi_or_structure_id ?? '—'} (score ${s.score == null ? '—' : s.score.toFixed(2)})`;
        if (s.vessel_lookup_url && v instanceof HTMLAnchorElement) {
          v.href = s.vessel_lookup_url;
          v.target = '_blank';
          v.rel = 'noopener noreferrer';
        }
        row.append(l, v);
        ships.appendChild(row);
      }
    })
    .catch((e) => {
      ships.textContent = `Ship lookup failed: ${e instanceof Error ? e.message : String(e)}`;
    });
  const note = document.createElement('div');
  note.className = 'popup-note';
  note.textContent = 'Full details are in the panel on the right. A candidate ship is not proof.';
  el.appendChild(note);
  return el;
}

const REF_TILE_LAYERS = ['ref-tile-fill', 'ref-tile-line'];

export const MapView = forwardRef<MapHandle, Props>(function MapView(props, ref) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const overlayKeys = useRef<Set<string>>(new Set());
  const hoverRef = useRef<number | null>(null);
  const listHoverRef = useRef<number | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);

  // Keep latest callbacks without re-binding map listeners.
  const cb = useRef(props);
  cb.current = props;

  useImperativeHandle(ref, () => ({
    fitBBox: (b, opts) => {
      const map = mapRef.current;
      if (!map) return;
      const [minx, miny, maxx, maxy] = b;
      const bounds: LngLatBoundsLike = [
        [minx, miny],
        [maxx, maxy],
      ];
      const el = map.getContainer();
      // Keep padding proportional so small (mobile / squeezed) map areas still fit the target.
      const pad = Math.max(10, Math.min(opts?.padding ?? 80, el.clientWidth / 5, el.clientHeight / 5));
      // On phones the detail panel is a bottom sheet over the map; keep the target above it.
      const sheet = window.innerWidth <= 760 ? document.querySelector<HTMLElement>('.detail-panel') : null;
      const sheetH = sheet ? Math.min(sheet.offsetHeight, el.clientHeight * 0.6) : 0;
      map.fitBounds(bounds, {
        padding: { top: pad, left: pad, right: pad, bottom: pad + sheetH },
        maxZoom: opts?.maxZoom ?? 11,
        duration: 1200,
      });
    },
  }));

  // Init map once.
  useEffect(() => {
    if (!containerRef.current) return;
    const { lat, lon, zoom } = cb.current.initialView;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE,
      center: [lon, lat],
      zoom,
      attributionControl: false,
      maxPitch: 0,
      dragRotate: false,
      renderWorldCopies: true,
    });
    map.touchZoomRotate.disableRotation();
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');
    // Keep credits behind the (i) button until the user opens them.
    map.once('load', () => {
      containerRef.current?.querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show');
      containerRef.current?.querySelector('.maplibregl-ctrl-attrib')?.removeAttribute('open');
    });
    mapRef.current = map;

    const emitView = () => {
      const b = map.getBounds();
      const bbox = clampBBox([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
      const c = map.getCenter().wrap();
      cb.current.onViewChange(bbox, { lat: c.lat, lon: c.lng, zoom: map.getZoom() });
    };

    let styleReady = false;
    map.on('load', () => {
      addLayers(map);
      styleReady = true;
      setReady(true);
      emitView();
    });
    // Note: don't gate on map.isStyleLoaded() — it is false while tiles are still loading.
    map.on('moveend', () => {
      if (styleReady) emitView();
    });

    const setHover = (id: number | null) => {
      if (hoverRef.current !== null && hoverRef.current !== listHoverRef.current) {
        map.setFeatureState({ source: 'detections', id: hoverRef.current }, { hover: false });
      }
      hoverRef.current = id;
      if (id !== null) map.setFeatureState({ source: 'detections', id }, { hover: true });
    };

    map.on('mousemove', 'det-fill', (e) => {
      const f = e.features?.[0];
      const id = f?.id != null ? Number(f.id) : null;
      if (id !== hoverRef.current) setHover(id);
      map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', 'det-fill', () => {
      setHover(null);
      map.getCanvas().style.cursor = '';
    });
    for (const layer of ['upload-fill', 'reference-fill', 'det-dots', ...REF_TILE_LAYERS]) {
      map.on('mouseenter', layer, () => (map.getCanvas().style.cursor = 'pointer'));
      map.on('mouseleave', layer, () => (map.getCanvas().style.cursor = ''));
    }

    map.on('click', (e) => {
      const layers = ['upload-fill', 'det-fill', 'det-dots', 'reference-fill', ...REF_TILE_LAYERS].filter((l) => map.getLayer(l));
      const hits = map.queryRenderedFeatures(e.point, { layers });
      popupRef.current?.remove();
      const upload = hits.find((h) => h.layer.id === 'upload-fill');
      if (upload) {
        if (upload.id != null) cb.current.onSelectUpload(Number(upload.id));
        return;
      }
      const showDetPopup = (props: Record<string, unknown>) => {
        popupRef.current = new maplibregl.Popup({ maxWidth: '320px', className: 'oil-popup' })
          .setLngLat(e.lngLat)
          .setDOMContent(detectionPopup(props))
          .addTo(map);
      };
      const det = hits.find((h) => h.layer.id === 'det-fill');
      if (det && det.id != null) {
        cb.current.onSelectDetection(Number(det.id));
        showDetPopup({ ...(det.properties as Record<string, unknown>), id: det.id });
        return;
      }
      // Zoomed-out dot for one of our detections: its properties carry the detection id.
      const dot = hits.find((h) => h.layer.id === 'det-dots');
      const dotId = dot ? Number((dot.properties as Record<string, unknown>)?.id) : NaN;
      if (Number.isFinite(dotId)) {
        cb.current.onSelectDetection(dotId);
        showDetPopup(dot!.properties as Record<string, unknown>);
        return;
      }
      const tileHit = hits.find((h) => REF_TILE_LAYERS.includes(h.layer.id));
      if (tileHit) {
        popupRef.current = new maplibregl.Popup({ maxWidth: '320px', className: 'oil-popup' })
          .setLngLat(e.lngLat)
          .setDOMContent(referenceTilePopup(tileHit.properties as Record<string, unknown>))
          .addTo(map);
        return;
      }
      const refHit = hits.find((h) => h.layer.id === 'reference-fill');
      if (refHit) {
        popupRef.current = new maplibregl.Popup({ maxWidth: '300px', className: 'oil-popup' })
          .setLngLat(e.lngLat)
          .setDOMContent(referencePopupContent(refHit))
          .addTo(map);
        return;
      }
      cb.current.onClearSelection();
    });

    const ro = new ResizeObserver(() => map.resize());
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
      overlayKeys.current = new Set();
      setReady(false);
    };
  }, []);

  // Data bindings.
  useEffect(() => {
    if (!ready) return;
    const src = mapRef.current?.getSource('detections') as GeoJSONSource | undefined;
    hoverRef.current = null;
    src?.setData(props.detections);
    const pts = mapRef.current?.getSource('det-points') as GeoJSONSource | undefined;
    pts?.setData(centroidPoints(props.detections));
  }, [ready, props.detections]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    for (const l of REF_TILE_LAYERS) if (map.getLayer(l)) map.removeLayer(l);
    if (map.getSource('ref-tiles')) map.removeSource('ref-tiles');
    if (!props.referenceTilesUrl) return;
    map.addSource('ref-tiles', {
      type: 'vector',
      tiles: [props.referenceTilesUrl],
      minzoom: 0,
      maxzoom: 10,
      attribution: 'Slick archive &amp; ship matches: SkyTruth Cerulean, CC BY-SA 4.0 | Imagery: contains modified Copernicus Sentinel-1 data',
    });
    const before = map.getLayer('det-fill') ? 'det-fill' : undefined;
    map.addLayer(
      { id: 'ref-tile-fill', type: 'fill', source: 'ref-tiles', 'source-layer': 'default', paint: { 'fill-color': '#8fb3cc', 'fill-opacity': 0.25 } },
      before,
    );
    // Thick outline keeps tiny slicks visible as dots when zoomed out.
    map.addLayer(
      {
        id: 'ref-tile-line',
        type: 'line',
        source: 'ref-tiles',
        'source-layer': 'default',
        paint: { 'line-color': '#8fb3cc', 'line-width': ['interpolate', ['linear'], ['zoom'], 2, 2.5, 7, 1.5, 11, 1] },
      },
      before,
    );
  }, [ready, props.referenceTilesUrl]);

  useEffect(() => {
    if (!ready) return;
    const src = mapRef.current?.getSource('upload') as GeoJSONSource | undefined;
    src?.setData(props.uploadDetections ?? emptyCollection());
  }, [ready, props.uploadDetections]);

  useEffect(() => {
    if (!ready) return;
    const map = mapRef.current;
    const src = map?.getSource('reference') as GeoJSONSource | undefined;
    src?.setData(props.reference ?? emptyCollection());
    const pts = map?.getSource('ref-points') as GeoJSONSource | undefined;
    pts?.setData(props.reference ? centroidPoints(props.reference) : emptyCollection());
    if (!props.reference) popupRef.current?.remove();
  }, [ready, props.reference]);

  useEffect(() => {
    if (!ready) return;
    const src = mapRef.current?.getSource('footprint') as GeoJSONSource | undefined;
    const fc: FeatureCollection<Geometry> = props.footprint
      ? featureCollection([{ type: 'Feature', properties: {}, geometry: bboxToPolygon(props.footprint) }])
      : emptyCollection();
    src?.setData(fc);
  }, [ready, props.footprint]);

  useEffect(() => {
    if (!ready) return;
    mapRef.current?.setFilter('det-selected', ['==', ['to-number', ['get', 'id']], props.selectedId ?? -1]);
  }, [ready, props.selectedId, props.detections]);

  useEffect(() => {
    if (!ready) return;
    mapRef.current?.setFilter('upload-selected', ['==', ['id'], props.selectedUploadIndex ?? -1]);
  }, [ready, props.selectedUploadIndex, props.uploadDetections]);

  // Hover from the detection list.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const prev = listHoverRef.current;
    if (prev !== null && prev !== hoverRef.current) {
      map.setFeatureState({ source: 'detections', id: prev }, { hover: false });
    }
    listHoverRef.current = props.hoveredId;
    if (props.hoveredId !== null) {
      map.setFeatureState({ source: 'detections', id: props.hoveredId }, { hover: true });
    }
  }, [ready, props.hoveredId]);

  // Image overlays (SAR quicklook / probability heatmap), drawn beneath vector layers.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const desired = new Map(props.overlays.map((o) => [o.key, o]));
    for (const key of Array.from(overlayKeys.current)) {
      if (!desired.has(key)) {
        const id = `ov-${key}`;
        if (map.getLayer(id)) map.removeLayer(id);
        if (map.getSource(id)) map.removeSource(id);
        overlayKeys.current.delete(key);
      }
    }
    for (const o of props.overlays) {
      const id = `ov-${o.key}`;
      const corners = bboxToImageCorners(o.bounds);
      if (overlayKeys.current.has(o.key)) {
        (map.getSource(id) as ImageSource | undefined)?.setCoordinates(corners);
        continue;
      }
      map.addSource(id, { type: 'image', url: o.url, coordinates: corners });
      map.addLayer(
        {
          id,
          type: 'raster',
          source: id,
          paint: {
            'raster-opacity': o.kind === 'prob' ? 0.85 : 0.9,
            'raster-fade-duration': 0,
            'raster-resampling': o.kind === 'prob' ? 'nearest' : 'linear',
          },
        },
        FIRST_DATA_LAYER,
      );
      overlayKeys.current.add(o.key);
    }
    // Probability heatmaps above SAR images.
    for (const o of props.overlays) {
      if (o.kind === 'prob') map.moveLayer(`ov-${o.key}`, FIRST_DATA_LAYER);
    }
  }, [ready, props.overlays]);

  return <div ref={containerRef} className="map-container" aria-label="Map of oil slick detections" role="region" />;
});
