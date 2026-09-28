import { useEffect, useRef, useState } from 'react';
import { api, errorMessage } from '../api/client';
import type { BBox, Health, ModelInfo } from '../api/types';
import { SearchBox } from './SearchBox';
import { fmtNum } from '../lib/format';

interface Props {
  health: Health | null;
  healthError: string | null;
  onUploadClick: () => void;
  onToggleSidebar: () => void;
  onPlace: (bbox: BBox) => void;
}

export function Header({ health, healthError, onUploadClick, onToggleSidebar, onPlace }: Props) {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const popRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const ctrl = new AbortController();
    setInfoLoading(true);
    setInfoError(null);
    api
      .modelInfo(ctrl.signal)
      .then(setInfo)
      .catch((e) => {
        if (!ctrl.signal.aborted) setInfoError(errorMessage(e));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setInfoLoading(false);
      });
    const onDoc = (e: MouseEvent) => {
      if (popRef.current && !popRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      ctrl.abort();
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  let badgeClass = 'status-ind is-muted';
  let badgeText = 'Checking…';
  let badgeTitle = 'Checking API status';
  if (healthError) {
    badgeClass = 'status-ind is-error';
    badgeText = 'Model offline';
    badgeTitle = `API offline: ${healthError}`;
  } else if (health) {
    if (health.model_loaded) {
      badgeClass = 'status-ind is-ok';
      badgeText = 'Model ready';
      badgeTitle = `Model ${health.model_version ?? 'unversioned'} loaded. Click for details.`;
    } else {
      badgeClass = 'status-ind is-warn';
      badgeText = 'No model loaded';
      badgeTitle = 'API is up but no model is loaded. Click for details.';
    }
  }

  return (
    <header className="app-header">
      <button className="icon-btn sidebar-toggle" onClick={onToggleSidebar} aria-label="Toggle filter panel" title="Toggle panel">
        <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
      <div className="brand">
        <svg className="brand-mark" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M12 3.5c2.9 4.1 5.5 7.3 5.5 10.2a5.5 5.5 0 0 1-11 0c0-2.9 2.6-6.1 5.5-10.2z" fill="currentColor" />
        </svg>
        <h1>
          Ocean Police <span className="brand-sub">Oil slick detection</span>
        </h1>
      </div>
      <SearchBox onPick={onPlace} />
      <div className="header-actions">
        <div className="badge-wrap" ref={popRef}>
          <button
            className={badgeClass}
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            title={badgeTitle}
          >
            <span className="dot" />
            {badgeText}
          </button>
          {open && (
            <div className="popover" role="dialog" aria-label="Model information">
              <div className="popover-title">Model and API status</div>
              {healthError && <p className="text-error">{healthError}</p>}
              {health && (
                <dl className="kv">
                  <dt>API</dt>
                  <dd>{health.status}</dd>
                  <dt>Detections in DB</dt>
                  <dd>{fmtNum(health.db_detections, 0)}</dd>
                </dl>
              )}
              {infoLoading && <p className="muted">Loading model info…</p>}
              {infoError && <p className="text-error">Model info unavailable: {infoError}</p>}
              {info && (
                <dl className="kv">
                  <dt>Loaded</dt>
                  <dd>{info.loaded ? 'yes' : 'no'}</dd>
                  <dt>Version</dt>
                  <dd>{info.model_version ?? '—'}</dd>
                  <dt>Architecture</dt>
                  <dd>{info.architecture ?? '—'}</dd>
                  <dt>Classes</dt>
                  <dd>
                    {Object.entries(info.classes ?? {})
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(', ') || '—'}
                  </dd>
                  <dt>Trained on</dt>
                  <dd>{info.trained_on ?? '—'}</dd>
                  {info.thresholds && Object.keys(info.thresholds).length > 0 && (
                    <>
                      <dt>Thresholds</dt>
                      <dd>
                        <code className="small-code">{JSON.stringify(info.thresholds)}</code>
                      </dd>
                    </>
                  )}
                  <dt>Metrics</dt>
                  <dd>
                    {info.metrics ? (
                      <code className="small-code">{JSON.stringify(info.metrics, null, 1)}</code>
                    ) : (
                      'not reported'
                    )}
                  </dd>
                </dl>
              )}
            </div>
          )}
        </div>
        <button className="btn" onClick={onUploadClick}>
          <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
            <path d="M12 16V4m0 0l-5 5m5-5l5 5M4 20h16" stroke="currentColor" strokeWidth="1.75" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="hide-sm">Upload GeoTIFF</span>
        </button>
      </div>
    </header>
  );
}
