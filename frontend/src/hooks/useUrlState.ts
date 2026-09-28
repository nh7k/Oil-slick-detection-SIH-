import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { ClassId } from '../api/types';
import { CLASS_IDS, isValidIsoDay } from '../lib/format';

export interface Filters {
  start: string;
  end: string;
  minConfidence: number;
  classes: ClassId[];
  minArea: number | null;
  maxArea: number | null;
  showReference: boolean;
  showQuicklooks: boolean;
}

export interface MapView {
  lat: number;
  lon: number;
  zoom: number;
}

export const DEFAULT_MIN_CONFIDENCE = 0.9; // below ~0.9 most of our detections do not match any reference slick (measured 2026-09-28)
export const DEFAULT_VIEW: MapView = { lat: 20, lon: 10, zoom: 2 };

/** Last 12 months, ending today (the user's local calendar day). */
export function defaultDateRange(): { start: string; end: string } {
  const now = new Date();
  const local = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  const start = new Date(now);
  start.setFullYear(start.getFullYear() - 1);
  return { start: local(start), end: local(now) };
}

function parseNum(v: string | null): number | null {
  if (v === null || v.trim() === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function parseFilters(sp: URLSearchParams): Filters {
  const def = defaultDateRange();
  let start = def.start;
  let end = def.end;
  const dr = sp.get('date_range');
  if (dr) {
    const [a, b] = dr.split('_');
    if (isValidIsoDay(a) && isValidIsoDay(b)) {
      start = a <= b ? a : b;
      end = a <= b ? b : a;
    }
  }
  const mc = parseNum(sp.get('min_confidence'));
  const minConfidence = mc === null ? DEFAULT_MIN_CONFIDENCE : Math.min(1, Math.max(0, mc));

  let classes: ClassId[] = [...CLASS_IDS];
  const cl = sp.get('classes');
  if (cl !== null) {
    classes = cl
      .split(/[_,]/)
      .map((s) => Number(s))
      .filter((n): n is ClassId => (CLASS_IDS as number[]).includes(n));
  }
  const minArea = parseNum(sp.get('min_area'));
  const maxArea = parseNum(sp.get('max_area'));
  return {
    start,
    end,
    minConfidence,
    classes,
    minArea: minArea !== null && minArea >= 0 ? minArea : null,
    maxArea: maxArea !== null && maxArea >= 0 ? maxArea : null,
    showReference: sp.get('reference') !== '0',
    showQuicklooks: sp.get('quicklooks') === '1',
  };
}

export function parseView(sp: URLSearchParams): MapView {
  const lat = parseNum(sp.get('lat'));
  const lon = parseNum(sp.get('lon'));
  const zoom = parseNum(sp.get('zoom'));
  return {
    lat: lat !== null && Math.abs(lat) <= 85 ? lat : DEFAULT_VIEW.lat,
    lon: lon !== null && Math.abs(lon) <= 180 ? lon : DEFAULT_VIEW.lon,
    zoom: zoom !== null && zoom >= 0 && zoom <= 22 ? zoom : DEFAULT_VIEW.zoom,
  };
}

function writeFilters(sp: URLSearchParams, f: Filters): void {
  sp.set('min_confidence', String(Number(f.minConfidence.toFixed(2))));
  sp.set('classes', f.classes.slice().sort().join('_'));
  sp.set('date_range', `${f.start}_${f.end}`);
  if (f.minArea !== null) sp.set('min_area', String(f.minArea));
  else sp.delete('min_area');
  if (f.maxArea !== null) sp.set('max_area', String(f.maxArea));
  else sp.delete('max_area');
  if (f.showReference) sp.delete('reference');
  else sp.set('reference', '0');
  if (f.showQuicklooks) sp.set('quicklooks', '1');
  else sp.delete('quicklooks');
}

/** Filters and map position mirrored to the URL query string (shareable links). */
export function useUrlState() {
  const [searchParams, setSearchParams] = useSearchParams();

  const filtersKey = [
    'date_range',
    'min_confidence',
    'classes',
    'min_area',
    'max_area',
    'reference',
    'quicklooks',
  ]
    .map((k) => `${k}=${searchParams.get(k) ?? ''}`)
    .join('&');

  // Only re-derive filters when filter params change (not on every map move).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const filters = useMemo(() => parseFilters(searchParams), [filtersKey]);

  const setFilters = useCallback(
    (patch: Partial<Filters>) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          writeFilters(next, { ...parseFilters(prev), ...patch });
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setView = useCallback(
    (v: MapView) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('lat', v.lat.toFixed(4));
          next.set('lon', v.lon.toFixed(4));
          next.set('zoom', v.zoom.toFixed(2));
          if (!next.has('date_range')) writeFilters(next, parseFilters(prev));
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return { filters, setFilters, setView, searchParams };
}
