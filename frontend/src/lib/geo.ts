import type { Feature, FeatureCollection, GeoJsonProperties, Geometry, Polygon, Position } from 'geojson';
import type { BBox } from '../api/types';

function visitPositions(geom: Geometry | null | undefined, fn: (p: Position) => void): void {
  if (!geom) return;
  switch (geom.type) {
    case 'Point':
      fn(geom.coordinates);
      break;
    case 'MultiPoint':
    case 'LineString':
      geom.coordinates.forEach(fn);
      break;
    case 'MultiLineString':
    case 'Polygon':
      geom.coordinates.forEach((r) => r.forEach(fn));
      break;
    case 'MultiPolygon':
      geom.coordinates.forEach((poly) => poly.forEach((r) => r.forEach(fn)));
      break;
    case 'GeometryCollection':
      geom.geometries.forEach((g) => visitPositions(g, fn));
      break;
  }
}

/** Bounding box of a set of features computed from their (API-supplied) geometry. */
export function featuresBBox(features: Feature[]): BBox | null {
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  for (const f of features) {
    visitPositions(f.geometry, (p) => {
      const [x, y] = p;
      if (x < minx) minx = x;
      if (y < miny) miny = y;
      if (x > maxx) maxx = x;
      if (y > maxy) maxy = y;
    });
  }
  if (!Number.isFinite(minx)) return null;
  return [minx, miny, maxx, maxy];
}

export function isValidBBox(b: unknown): b is BBox {
  return Array.isArray(b) && b.length === 4 && b.every((v) => typeof v === 'number' && Number.isFinite(v));
}

/** Rectangle polygon from a bbox (used for scene footprints returned by the API). */
export function bboxToPolygon(b: BBox): Polygon {
  const [minx, miny, maxx, maxy] = b;
  const ring: Position[] = [
    [minx, miny],
    [maxx, miny],
    [maxx, maxy],
    [minx, maxy],
    [minx, miny],
  ];
  return { type: 'Polygon', coordinates: [ring] };
}

/** MapLibre image-source corner order: top-left, top-right, bottom-right, bottom-left. */
export function bboxToImageCorners(b: BBox): [[number, number], [number, number], [number, number], [number, number]] {
  const [minx, miny, maxx, maxy] = b;
  return [
    [minx, maxy],
    [maxx, maxy],
    [maxx, miny],
    [minx, miny],
  ];
}

export function emptyCollection<G extends Geometry = Geometry, P = Record<string, unknown>>(): FeatureCollection<G, P> {
  return { type: 'FeatureCollection', features: [] };
}

export function clampBBox(b: BBox): BBox {
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  return [clamp(b[0], -180, 180), clamp(b[1], -90, 90), clamp(b[2], -180, 180), clamp(b[3], -90, 90)];
}

export function featureCollection<G extends Geometry, P = GeoJsonProperties>(features: Feature<G, P>[]): FeatureCollection<G, P> {
  return { type: 'FeatureCollection', features };
}
