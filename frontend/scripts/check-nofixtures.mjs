#!/usr/bin/env node
// Fails (exit 1) if frontend/src contains mock/fixture geodata:
//   - any geodata files (.geojson, .kml, .kmz, .shp, .gpx, .topojson)
//   - any JSON file under src (data should come from the API, not from bundled files)
//   - literal coordinate arrays: `coordinates: [[…` / `"coordinates": [ -12.3 …`
//   - runs of 3+ literal [lon, lat] decimal pairs (a hand-written ring)
//   - literal FeatureCollections with inline features
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, extname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = join(root, 'src');

const FORBIDDEN_EXT = new Set(['.geojson', '.kml', '.kmz', '.shp', '.gpx', '.topojson', '.json']);
const CODE_EXT = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs']);

const PATTERNS = [
  {
    name: 'literal coordinates array',
    // coordinates: [ [ [ 12.3   or  "coordinates": [-12.3
    re: /["']?coordinates["']?\s*:\s*\[\s*(?:\[\s*){0,3}-?\d/g,
  },
  {
    name: 'hand-written coordinate ring (3+ literal [x, y] pairs)',
    re: /\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\](?:\s*,\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]){2,}/g,
  },
  {
    name: 'inline FeatureCollection with literal features',
    re: /["']?type["']?\s*:\s*["']FeatureCollection["']\s*,\s*["']?features["']?\s*:\s*\[\s*\{/g,
  },
];

function walk(dir) {
  const out = [];
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of entries) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

function lineOf(text, index) {
  return text.slice(0, index).split('\n').length;
}

const problems = [];
const files = walk(srcDir);
for (const file of files) {
  const rel = relative(root, file).replaceAll('\\', '/');
  const ext = extname(file).toLowerCase();
  if (FORBIDDEN_EXT.has(ext)) {
    problems.push(`${rel}: forbidden data file type "${ext}" in src/`);
    continue;
  }
  if (!CODE_EXT.has(ext)) continue;
  const text = readFileSync(file, 'utf8');
  for (const { name, re } of PATTERNS) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      problems.push(`${rel}:${lineOf(text, m.index)}: ${name}: ${m[0].slice(0, 60).replace(/\s+/g, ' ')}`);
    }
  }
}

if (problems.length) {
  console.error('check:nofixtures FAILED — hardcoded geodata found in frontend/src:');
  for (const p of problems) console.error('  - ' + p);
  process.exit(1);
}
console.log(`check:nofixtures OK — scanned ${files.length} files in src/, no fixtures or literal geometries.`);
