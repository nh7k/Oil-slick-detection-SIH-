import { useState } from 'react';
import type { ClassId } from '../api/types';
import { CLASS_COLORS, CLASS_IDS } from '../lib/format';

interface Props {
  classNames: Record<ClassId, string>;
  count: number;
  truncated: boolean;
  showReference: boolean;
  hasUpload: boolean;
  overlays: number;
  onClearOverlays?: () => void;
}

export function Legend({ classNames, count, truncated, showReference, hasUpload, overlays, onClearOverlays }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className={`legend ${collapsed ? 'collapsed' : ''}`}>
      <button className="legend-head" onClick={() => setCollapsed((c) => !c)} aria-expanded={!collapsed}>
        <span>
          {truncated ? `${count}+` : count} detection{count === 1 ? '' : 's'} in view
        </span>
        <svg className={`chev ${collapsed ? 'up' : ''}`} viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
          <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {!collapsed && (
        <ul>
          <li className="legend-heading">Our model</li>
          {CLASS_IDS.map((c) => (
            <li key={c}>
              <span className="swatch" style={{ background: CLASS_COLORS[c] }} />
              {classNames[c]}
            </li>
          ))}
          {showReference && (
            <>
              <li className="legend-heading">Other sources</li>
              <li>
                <span className="swatch swatch-dashed ref" />
                Slick archive
              </li>
            </>
          )}
          {hasUpload && (
            <li>
              <span className="swatch swatch-dashed upload" />
              Upload result (temporary)
            </li>
          )}
          {overlays > 0 && (
            <li className="legend-overlays">
              <span className="swatch swatch-img" />
              {overlays} image overlay{overlays === 1 ? '' : 's'}
              {onClearOverlays && (
                <button className="mini-btn" onClick={onClearOverlays}>
                  clear
                </button>
              )}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
