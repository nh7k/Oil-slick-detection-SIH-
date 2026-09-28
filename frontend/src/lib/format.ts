import type { BBox, ClassId } from '../api/types';

export const CLASS_IDS: ClassId[] = [1, 2, 3];

/** Default names per the API contract; overridden by /api/model-info when available. */
export const DEFAULT_CLASS_NAMES: Record<ClassId, string> = {
  1: 'Infrastructure',
  2: 'Natural seep',
  3: 'Vessel',
};

export const CLASS_COLORS: Record<ClassId, string> = {
  1: '#d99a4e', // infrastructure — muted amber
  2: '#7fb77e', // natural seep — sage green
  3: '#d77a9a', // vessel — dusty rose
};

export const CLASS_DESCRIPTIONS: Record<ClassId, string> = {
  1: 'Platforms, pipelines',
  2: 'Seabed seepage',
  3: 'Discharge from ships',
};

export function classColor(cls: number | null | undefined): string {
  return CLASS_COLORS[(cls ?? 0) as ClassId] ?? '#8a939d';
}

export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function fmtPct(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(digits)}%`;
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().replace('T', ' ').replace(/:\d\d(\.\d+)?Z$/, ' UTC');
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 10);
}

export function fmtLatLon(lat: number | null | undefined, lon: number | null | undefined): string {
  if (lat == null || lon == null || !Number.isFinite(lat) || !Number.isFinite(lon)) return '—';
  return `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}

export function fmtBBox(b: BBox | null | undefined): string {
  if (!b || b.length !== 4) return '—';
  return b.map((v) => v.toFixed(4)).join(', ');
}

/** YYYY-MM-DD for a Date in UTC. */
export function isoDay(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export function isValidIsoDay(s: string | null | undefined): s is string {
  return !!s && /^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(new Date(`${s}T00:00:00Z`).getTime());
}

export function addDays(day: string, n: number): string {
  const d = new Date(`${day}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return isoDay(d);
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }
}

export function downloadJson(obj: unknown, filename: string, mime = 'application/geo+json'): void {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
