import { useState } from 'react';
import { SourcesSection } from './SourcesSection';
import type { ClassId, DetectionFeature } from '../api/types';
import {
  classColor,
  copyText,
  downloadJson,
  fmtBBox,
  fmtDateTime,
  fmtLatLon,
  fmtNum,
  fmtPct,
} from '../lib/format';

interface Props {
  requestedId: string | null;
  feature: DetectionFeature | null;
  isUpload: boolean;
  loading: boolean;
  error: string | null;
  classNames: Record<ClassId, string>;
  sarActive: boolean;
  probActive: boolean;
  overlayBusy: 'sar' | 'prob' | null;
  overlayError: string | null;
  onToggleSar: () => void;
  onToggleProb: () => void;
  onZoom: () => void;
  onClose: () => void;
}

function ConfBar({ label, value }: { label: string; value: number | null | undefined }) {
  const pct = value == null || !Number.isFinite(value) ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="conf-row">
      <span className="conf-label">{label}</span>
      <span className="conf-bar" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={value ?? undefined} aria-label={label}>
        <span style={{ width: `${pct}%` }} />
      </span>
      <span className="conf-val">{fmtPct(value)}</span>
    </div>
  );
}

export function DetailPanel(props: Props) {
  const { feature, loading, error, requestedId, classNames, isUpload } = props;
  const [copied, setCopied] = useState(false);

  const p = feature?.properties;
  const idLabel = p?.id ?? feature?.id ?? requestedId;

  const onCopy = async () => {
    if (!p || p.centroid_lat == null || p.centroid_lon == null) return;
    const ok = await copyText(`${p.centroid_lat.toFixed(6)}, ${p.centroid_lon.toFixed(6)}`);
    setCopied(ok);
    if (ok) window.setTimeout(() => setCopied(false), 1500);
  };

  const onDownload = () => {
    if (!feature) return;
    downloadJson(feature, `ocean-police-detection-${idLabel ?? 'feature'}.geojson`);
  };

  return (
    <aside className="detail-panel" aria-label="Detection details">
      <div className="detail-head">
        <div>
          <div className="eyebrow">{isUpload ? 'Upload result (temporary)' : 'Detection'}</div>
          <h2>
            {p ? (
              <>
                <span className="swatch lg" style={{ background: classColor(p.cls) }} />
                {classNames[p.cls] ?? p.cls_name ?? `Class ${p.cls}`}
              </>
            ) : (
              'Detection'
            )}
            {idLabel != null && <span className="muted"> #{String(idLabel)}</span>}
          </h2>
        </div>
        <button className="icon-btn" onClick={props.onClose} aria-label="Close details" title="Close">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {loading && <div className="detail-body muted">Loading detection…</div>}
      {error && !loading && <div className="detail-body text-error">{error}</div>}

      {p && !loading && (
        <div className="detail-body">
          <div className="block">
            <h3 className="section-title">Confidence (per-pixel oil probability)</h3>
            <ConfBar label="Median" value={p.confidence} />
            <ConfBar label="Mean" value={p.confidence_mean} />
            <ConfBar label="Max" value={p.confidence_max} />
          </div>

          <div className="block">
            <h3 className="section-title">Geometry</h3>
            <dl className="kv">
              <dt>Area</dt>
              <dd>{fmtNum(p.area_km2, 3)} km²</dd>
              <dt>Length</dt>
              <dd>{fmtNum(p.length_km, 2)} km</dd>
              <dt>Perimeter</dt>
              <dd>{fmtNum(p.perimeter_km, 2)} km</dd>
              <dt title="Polsby–Popper compactness: 4πA/P² (1 = circle)">Polsby–Popper</dt>
              <dd>{fmtNum(p.polsby_popper, 3)}</dd>
              <dt>Centroid (lat, lon)</dt>
              <dd className="with-action">
                <span className="mono">{fmtLatLon(p.centroid_lat, p.centroid_lon)}</span>
                {p.centroid_lat != null && (
                  <button className="mini-btn" onClick={onCopy} title="Copy coordinates">
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                )}
              </dd>
              <dt>BBox</dt>
              <dd className="mono small">{fmtBBox(p.bbox)}</dd>
            </dl>
          </div>

          <div className="block">
            <h3 className="section-title">Provenance</h3>
            <dl className="kv">
              <dt>Scene</dt>
              <dd className="mono small break">{p.scene_id ?? '—'}</dd>
              <dt>Acquired</dt>
              <dd>{fmtDateTime(p.acquired_at)}</dd>
              <dt>Model</dt>
              <dd className="mono small">{p.model_version ?? '—'}</dd>
              <dt>Run</dt>
              <dd className="mono small">{p.run_id ?? '—'}</dd>
            </dl>
          </div>

          {!isUpload && p.id != null && <SourcesSection detectionId={p.id} />}

          <div className="actions">
            <button
              className={`btn ${props.sarActive ? 'btn-active' : ''}`}
              onClick={props.onToggleSar}
              disabled={!p.scene_id || props.overlayBusy !== null}
              aria-pressed={props.sarActive}
            >
              {props.overlayBusy === 'sar' ? 'Loading…' : props.sarActive ? 'Hide SAR image' : 'Show SAR image'}
            </button>
            <button
              className={`btn ${props.probActive ? 'btn-active' : ''}`}
              onClick={props.onToggleProb}
              disabled={!p.scene_id || props.overlayBusy !== null}
              aria-pressed={props.probActive}
            >
              {props.overlayBusy === 'prob'
                ? 'Loading…'
                : props.probActive
                  ? 'Hide probability heatmap'
                  : 'Show probability heatmap'}
            </button>
            <button className="btn" onClick={props.onZoom}>
              Zoom to slick
            </button>
            <button className="btn" onClick={onDownload}>
              Download GeoJSON
            </button>
          </div>
          {props.overlayError && <p className="hint warn">{props.overlayError}</p>}
        </div>
      )}
    </aside>
  );
}
