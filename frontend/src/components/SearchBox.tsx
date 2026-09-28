import { useRef, useState } from 'react';
import { api, errorMessage, type Place } from '../api/client';
import type { BBox } from '../api/types';

/** Place / sea search (OpenStreetMap Nominatim via our API). Search runs on Enter only, per Nominatim policy. */
export function SearchBox({ onPick }: { onPick: (bbox: BBox) => void }) {
  const [q, setQ] = useState('');
  const [results, setResults] = useState<Place[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = q.trim();
    if (query.length < 2) {
      setError('Type at least 2 letters');
      return;
    }
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setBusy(true);
    setError(null);
    try {
      const r = await api.geocode(query, ctrl.signal);
      setResults(r.results);
      if (r.results.length === 1) pick(r.results[0]);
      if (r.results.length === 0) setError('No place found');
    } catch (err) {
      if (!ctrl.signal.aborted) setError(errorMessage(err));
    } finally {
      if (!ctrl.signal.aborted) setBusy(false);
    }
  };

  const pick = (p: Place) => {
    onPick(p.bbox);
    setResults(null);
    setQ(p.name.split(',')[0]);
  };

  return (
    <div className="searchbox">
      <form onSubmit={submit} role="search">
        <input
          type="search"
          value={q}
          placeholder="Search a country, sea or place (e.g. Arabian Sea, Gujarat, Nigeria)"
          aria-label="Search a place"
          onChange={(e) => {
            setQ(e.target.value);
            setError(null);
          }}
        />
        <button type="submit" className="btn btn-small" disabled={busy}>
          {busy ? '…' : 'Go'}
        </button>
      </form>
      {error && <div className="searchbox-msg">{error}</div>}
      {results && results.length > 1 && (
        <ul className="searchbox-results">
          {results.map((p) => (
            <li key={`${p.name}-${p.bbox.join(',')}`}>
              <button type="button" onClick={() => pick(p)}>
                <span>{p.name}</span>
                <small>{p.type}</small>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
