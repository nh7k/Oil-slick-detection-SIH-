import { useEffect, useState } from 'react';
import type { ClassId } from '../api/types';
import type { Filters } from '../hooks/useUrlState';
import { defaultDateRange, DEFAULT_MIN_CONFIDENCE } from '../hooks/useUrlState';
import { CLASS_COLORS, CLASS_DESCRIPTIONS, CLASS_IDS } from '../lib/format';

interface Props {
  filters: Filters;
  classNames: Record<ClassId, string>;
  onChange: (patch: Partial<Filters>) => void;
  referenceStatus: string | null;
  quicklookStatus: string | null;
}

function AreaInput({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: number | null;
  onCommit: (v: number | null) => void;
}) {
  const [text, setText] = useState(value === null ? '' : String(value));
  useEffect(() => setText(value === null ? '' : String(value)), [value]);
  const commit = () => {
    const t = text.trim();
    if (t === '') return onCommit(null);
    const n = Number(t);
    if (Number.isFinite(n) && n >= 0) onCommit(n);
    else setText(value === null ? '' : String(value));
  };
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input
        type="number"
        min={0}
        step="any"
        inputMode="decimal"
        placeholder="any"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === 'Enter' && commit()}
      />
    </label>
  );
}

export function FilterPanel({ filters, classNames, onChange, referenceStatus, quicklookStatus }: Props) {
  // Local slider value for smooth dragging; commit on release.
  const [conf, setConf] = useState(filters.minConfidence);
  useEffect(() => setConf(filters.minConfidence), [filters.minConfidence]);

  const toggleClass = (c: ClassId) => {
    const has = filters.classes.includes(c);
    const next = has ? filters.classes.filter((x) => x !== c) : [...filters.classes, c].sort();
    onChange({ classes: next as ClassId[] });
  };

  const reset = () => {
    const d = defaultDateRange();
    onChange({
      start: d.start,
      end: d.end,
      minConfidence: DEFAULT_MIN_CONFIDENCE,
      classes: [...CLASS_IDS],
      minArea: null,
      maxArea: null,
    });
  };

  return (
    <div className="panel-section filters">
      <section>
        <h3 className="section-title">Date range (acquisition, UTC)</h3>
        <div className="row-2">
          <label className="field">
            <span className="field-label">From</span>
            <input
              type="date"
              value={filters.start}
              max={filters.end}
              onChange={(e) => e.target.value && onChange({ start: e.target.value })}
            />
          </label>
          <label className="field">
            <span className="field-label">To</span>
            <input
              type="date"
              value={filters.end}
              min={filters.start}
              onChange={(e) => e.target.value && onChange({ end: e.target.value })}
            />
          </label>
        </div>
      </section>

      <section>
        <h3 className="section-title">
          Minimum confidence <span className="value-pill">{conf.toFixed(2)}</span>
        </h3>
        <input
          className="range"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={conf}
          aria-label="Minimum confidence"
          onChange={(e) => setConf(Number(e.target.value))}
          onPointerUp={() => onChange({ minConfidence: conf })}
          onKeyUp={() => onChange({ minConfidence: conf })}
          onBlur={() => conf !== filters.minConfidence && onChange({ minConfidence: conf })}
        />
        <div className="range-scale">
          <span>0</span>
          <span>0.5</span>
          <span>1</span>
        </div>
      </section>

      <section>
        <h3 className="section-title">Source class</h3>
        <div className="class-toggles">
          {CLASS_IDS.map((c) => {
            const on = filters.classes.includes(c);
            return (
              <button
                key={c}
                className={`class-toggle ${on ? 'on' : ''}`}
                style={{ ['--cls' as string]: CLASS_COLORS[c] }}
                aria-pressed={on}
                onClick={() => toggleClass(c)}
                title={CLASS_DESCRIPTIONS[c]}
              >
                <span className="swatch" />
                <span className="class-name">{classNames[c]}</span>
                <span className="class-desc">{CLASS_DESCRIPTIONS[c]}</span>
              </button>
            );
          })}
        </div>
        {filters.classes.length === 0 && <p className="hint warn">No classes selected — nothing will be shown.</p>}
      </section>

      <section>
        <h3 className="section-title">Area (km²)</h3>
        <div className="row-2">
          <AreaInput label="Min" value={filters.minArea} onCommit={(v) => onChange({ minArea: v })} />
          <AreaInput label="Max" value={filters.maxArea} onCommit={(v) => onChange({ maxArea: v })} />
        </div>
        {filters.minArea !== null && filters.maxArea !== null && filters.minArea > filters.maxArea && (
          <p className="hint warn">Min area is greater than max area.</p>
        )}
      </section>

      <section>
        <h3 className="section-title">Layers</h3>
        <label className="switch">
          <input
            type="checkbox"
            checked={filters.showQuicklooks}
            onChange={(e) => onChange({ showQuicklooks: e.target.checked })}
          />
          <span className="switch-track" />
          <span>
            Scene SAR quicklooks
            <small>Grayscale Sentinel-1 images of processed scenes in view</small>
          </span>
        </label>
        {quicklookStatus && <p className="hint">{quicklookStatus}</p>}
        <label className="switch">
          <input
            type="checkbox"
            checked={filters.showReference}
            onChange={(e) => onChange({ showReference: e.target.checked })}
          />
          <span className="switch-track" />
          <span>
            Slick archive
            <small>Global archive of past slicks, with nearby ships</small>
          </span>
        </label>
        {referenceStatus && <p className="hint">{referenceStatus}</p>}
      </section>

      <button className="btn btn-ghost btn-block" onClick={reset}>
        Reset filters
      </button>
    </div>
  );
}
