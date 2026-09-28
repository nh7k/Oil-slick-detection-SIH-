import { useEffect, useState } from 'react';
import { api, type SourcesResponse } from '../api/client';

/** Candidate polluters for one of our detections (AIS-based attribution of the overlapping reference slick). */
export function SourcesSection({ detectionId }: { detectionId: number | string }) {
  const [data, setData] = useState<SourcesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setData(null);
    setError(null);
    api
      .sources(detectionId, ctrl.signal)
      .then(setData)
      .catch((e: unknown) => {
        if ((e as { name?: string })?.name === 'AbortError') return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, [detectionId]);

  return (
    <div className="block">
      <h3 className="section-title">Possible source (ships nearby)</h3>
      {!data && !error && <p className="hint">Looking up vessels near this slick at acquisition time…</p>}
      {error && <p className="hint warn">Source lookup failed: {error}</p>}
      {data && data.sources.length === 0 && (
        <p className="hint">{data.note ?? 'No candidate sources found.'}</p>
      )}
      {data && data.sources.length > 0 && (
        <>
          <table className="src-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Type</th>
                <th>MMSI / ID</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {data.sources.slice(0, 8).map((s, i) => (
                <tr key={`${s.source_type}-${s.mmsi_or_structure_id}`}>
                  <td>{i + 1}</td>
                  <td>{s.source_type ?? '—'}</td>
                  <td className="mono small">
                    {s.vessel_lookup_url ? (
                      <a href={s.vessel_lookup_url} target="_blank" rel="noreferrer" title="Look up this vessel (external)">
                        {s.mmsi_or_structure_id}
                      </a>
                    ) : (
                      s.mmsi_or_structure_id
                    )}
                  </td>
                  <td className="mono small">{s.score == null ? '—' : s.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Higher score = stronger match (AIS track vs slick position and time). A candidate is not proof of guilt.
          </p>
        </>
      )}
    </div>
  );
}
