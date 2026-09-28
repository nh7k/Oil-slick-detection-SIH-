import { useMemo, useState } from 'react';
import type { ClassId, DetectionFeature } from '../api/types';
import { classColor, fmtDate, fmtNum, fmtPct } from '../lib/format';

type SortKey = 'confidence' | 'area' | 'date';

interface Props {
  features: DetectionFeature[];
  classNames: Record<ClassId, string>;
  selectedId: number | null;
  loading: boolean;
  truncated: boolean;
  onSelect: (id: number) => void;
  onHover: (id: number | null) => void;
}

const PAGE = 200;

export function DetectionList({ features, classNames, selectedId, loading, truncated, onSelect, onHover }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('confidence');
  const [desc, setDesc] = useState(true);
  const [shown, setShown] = useState(PAGE);

  const sorted = useMemo(() => {
    const val = (f: DetectionFeature): number => {
      const p = f.properties;
      if (sortKey === 'confidence') return p.confidence ?? -1;
      if (sortKey === 'area') return p.area_km2 ?? -1;
      const t = p.acquired_at ? Date.parse(p.acquired_at) : NaN;
      return Number.isNaN(t) ? -1 : t;
    };
    const arr = features.slice();
    arr.sort((a, b) => (desc ? val(b) - val(a) : val(a) - val(b)));
    return arr;
  }, [features, sortKey, desc]);

  const clickSort = (k: SortKey) => {
    if (k === sortKey) setDesc((d) => !d);
    else {
      setSortKey(k);
      setDesc(true);
    }
  };

  const SortBtn = ({ k, label }: { k: SortKey; label: string }) => (
    <button className={`sort-btn ${sortKey === k ? 'active' : ''}`} onClick={() => clickSort(k)} aria-pressed={sortKey === k}>
      {label}
      {sortKey === k && <span aria-hidden="true">{desc ? ' ↓' : ' ↑'}</span>}
    </button>
  );

  return (
    <div className="panel-section det-list">
      <div className="sort-bar">
        <span className="muted small">Sort by</span>
        <SortBtn k="confidence" label="Confidence" />
        <SortBtn k="area" label="Area" />
        <SortBtn k="date" label="Date" />
      </div>
      {truncated && (
        <p className="hint warn">Result limit reached — zoom in or tighten filters to see every detection in this area.</p>
      )}
      {features.length === 0 ? (
        <p className="muted empty-list">{loading ? 'Loading detections…' : 'No detections in the current view and filters.'}</p>
      ) : (
        <ul className="rows" onMouseLeave={() => onHover(null)}>
          {sorted.slice(0, shown).map((f) => {
            const p = f.properties;
            const id = Number(p.id ?? f.id);
            return (
              <li key={id}>
                <button
                  className={`row ${selectedId === id ? 'selected' : ''}`}
                  onClick={() => onSelect(id)}
                  onMouseEnter={() => onHover(id)}
                  onFocus={() => onHover(id)}
                >
                  <span className="swatch" style={{ background: classColor(p.cls) }} />
                  <span className="row-main">
                    <span className="row-title">
                      {classNames[p.cls] ?? p.cls_name ?? `Class ${p.cls}`} <span className="muted">#{id}</span>
                    </span>
                    <span className="row-sub">
                      {fmtDate(p.acquired_at)} · {fmtNum(p.area_km2, 2)} km²
                    </span>
                  </span>
                  <span className="row-conf">
                    {fmtPct(p.confidence, 0)}
                    <span className="mini-bar">
                      <span style={{ width: `${Math.round((p.confidence ?? 0) * 100)}%` }} />
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {sorted.length > shown && (
        <button className="btn btn-ghost btn-block" onClick={() => setShown((s) => s + PAGE)}>
          Show more ({sorted.length - shown} remaining)
        </button>
      )}
    </div>
  );
}
