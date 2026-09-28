import { useState } from 'react';
import { api, errorMessage } from '../api/client';
import type { BBox, JobStatus, SearchSceneItem } from '../api/types';
import { fmtDateTime } from '../lib/format';

export interface TrackedJob {
  jobId: string;
  sceneId: string;
  status: JobStatus | null;
  error: string | null;
}

interface Props {
  viewBBox: BBox | null;
  start: string;
  end: string;
  jobs: TrackedJob[];
  onRun: (sceneId: string) => Promise<void>;
  onHoverScene: (bounds: BBox | null) => void;
  onZoomScene: (bounds: BBox) => void;
  onDismissJob: (jobId: string) => void;
}

function statusClass(s: string | undefined) {
  switch (s) {
    case 'done':
      return 'status status-ok';
    case 'error':
      return 'status status-error';
    case 'running':
      return 'status status-run';
    default:
      return 'status status-muted';
  }
}

export function ProcessPanel({ viewBBox, start, end, jobs, onRun, onHoverScene, onZoomScene, onDismissJob }: Props) {
  const [items, setItems] = useState<SearchSceneItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const search = async () => {
    if (!viewBBox) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.searchScenes({ bbox: viewBBox, start, end, limit: 20 });
      setItems(Array.isArray(res.items) ? res.items : []);
    } catch (e) {
      setError(errorMessage(e));
      setItems(null);
    } finally {
      setLoading(false);
    }
  };

  const run = async (sceneId: string) => {
    setSubmitting(sceneId);
    setSubmitError(null);
    try {
      await onRun(sceneId);
    } catch (e) {
      setSubmitError(`${sceneId}: ${errorMessage(e)}`);
    } finally {
      setSubmitting(null);
    }
  };

  const activeFor = (sceneId: string) =>
    jobs.find((j) => j.sceneId === sceneId && (!j.status || j.status.status === 'queued' || j.status.status === 'running'));

  return (
    <div className="panel-section process">
      <p className="muted small">
        Search Sentinel-1 GRD scenes that intersect the current map view ({start} to {end}), then run the
        detection model on one.
      </p>
      <button className="btn btn-primary btn-block" onClick={search} disabled={!viewBBox || loading}>
        {loading ? 'Searching…' : 'Search scenes in current view'}
      </button>
      {error && <p className="hint text-error">Scene search failed: {error}</p>}
      {submitError && <p className="hint text-error">Could not start job — {submitError}</p>}

      {jobs.length > 0 && (
        <div className="jobs">
          <h3 className="section-title">Jobs</h3>
          <ul className="rows">
            {jobs.map((j) => (
              <li key={j.jobId} className="job">
                <div className="job-head">
                  <span className={statusClass(j.status?.status)}>{j.status?.status ?? 'submitted'}</span>
                  <span className="mono small break">{j.sceneId}</span>
                  <button className="mini-btn" onClick={() => onDismissJob(j.jobId)} aria-label="Dismiss job">
                    ×
                  </button>
                </div>
                {j.status?.message && <div className="small muted">{j.status.message}</div>}
                {j.status?.status === 'done' && (
                  <div className="small">
                    {j.status.n_detections ?? 0} detection{j.status.n_detections === 1 ? '' : 's'} added — map refreshed.
                  </div>
                )}
                {j.error && <div className="small text-error">Polling error: {j.error}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {items && (
        <div className="scene-results">
          <h3 className="section-title">
            {items.length} scene{items.length === 1 ? '' : 's'} found
          </h3>
          {items.length === 0 && (
            <p className="muted small">No Sentinel-1 scenes intersect this view in the selected date range.</p>
          )}
          <ul className="rows" onMouseLeave={() => onHoverScene(null)}>
            {items.map((s) => {
              const active = activeFor(s.scene_id);
              return (
                <li key={s.scene_id} className="scene-item" onMouseEnter={() => onHoverScene(s.bounds)}>
                  <div className="scene-meta">
                    <span className="mono small break">{s.scene_id}</span>
                    <span className="small muted">
                      {fmtDateTime(s.acquired_at)}
                      {s.orbit_direction ? ` · ${s.orbit_direction.toLowerCase()}` : ''}
                    </span>
                  </div>
                  <div className="scene-actions">
                    <button className="mini-btn" onClick={() => onZoomScene(s.bounds)} title="Zoom to scene footprint">
                      Zoom
                    </button>
                    <button
                      className="btn btn-small btn-primary"
                      disabled={!!active || submitting === s.scene_id}
                      onClick={() => run(s.scene_id)}
                    >
                      {submitting === s.scene_id ? 'Submitting…' : active ? 'Running…' : 'Run detection'}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
