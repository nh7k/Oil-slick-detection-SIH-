import { useEffect, useRef, useState } from 'react';
import { api, errorMessage, isAbortError } from '../api/client';
import type { PredictGeotiffResponse } from '../api/types';

interface Props {
  open: boolean;
  onClose: () => void;
  onResult: (res: PredictGeotiffResponse) => void;
}

export function UploadDialog({ open, onClose, onResult }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const ctrlRef = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    const d = dialogRef.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  useEffect(() => () => ctrlRef.current?.abort(), []);

  const close = () => {
    ctrlRef.current?.abort();
    setBusy(false);
    setError(null);
    onClose();
  };

  const pick = (f: File | undefined | null) => {
    setError(null);
    if (!f) return;
    if (!/\.(tif|tiff)$/i.test(f.name)) {
      setError('Please choose a GeoTIFF (.tif / .tiff) file.');
      return;
    }
    setFile(f);
  };

  const submit = async () => {
    if (!file) return;
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setBusy(true);
    setError(null);
    try {
      const res = await api.predictGeotiff(file, ctrl.signal);
      if (!res || !res.detections || !Array.isArray(res.detections.features)) {
        throw new Error('Malformed response from /api/predict-geotiff');
      }
      onResult(res);
      setFile(null);
      onClose();
    } catch (e) {
      if (!isAbortError(e)) setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <dialog ref={dialogRef} className="dialog" onCancel={(e) => { e.preventDefault(); close(); }} aria-labelledby="upload-title">
      <div className="dialog-head">
        <h2 id="upload-title">Upload a Sentinel-1 GeoTIFF</h2>
        <button className="icon-btn" onClick={close} aria-label="Close">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      <p className="muted small">
        The file is sent to the Ocean Police API and run through the detection model. Resulting polygons are shown as a
        temporary, yellow-outlined layer.
      </p>
      <label
        className={`dropzone ${drag ? 'drag' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          type="file"
          accept=".tif,.tiff,image/tiff"
          onChange={(e) => pick(e.target.files?.[0])}
          disabled={busy}
        />
        {file ? (
          <span>
            <strong>{file.name}</strong>
            <br />
            <span className="muted small">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
          </span>
        ) : (
          <span>
            Drop a <strong>.tif</strong> here or <u>browse</u>
          </span>
        )}
      </label>
      {error && <p className="hint text-error">{error}</p>}
      {busy && (
        <div className="progress-indeterminate" aria-label="Processing">
          <span />
        </div>
      )}
      <div className="dialog-actions">
        <button className="btn btn-ghost" onClick={close}>
          {busy ? 'Cancel' : 'Close'}
        </button>
        <button className="btn btn-primary" onClick={submit} disabled={!file || busy}>
          {busy ? 'Running model…' : 'Run detection'}
        </button>
      </div>
    </dialog>
  );
}
