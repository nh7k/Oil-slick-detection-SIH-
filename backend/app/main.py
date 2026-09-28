"""OilWatch API (FastAPI).

Every detection this API returns was produced by the ONNX model in models/
running on real Sentinel-1 data; each row links to a run with model_version.
There is no fallback / demo data path.

Run:  uvicorn app.main:app --app-dir backend --port 8000
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from oilspill import config

# The imagery/model pipeline needs rasterio (GDAL). If Windows Smart App Control (or a
# broken install) blocks its native DLLs, keep serving the map, reference layer, ship
# attribution and stored detections, and report the problem instead of crashing.
#
# OILSPILL_API_PROCESSING=0 (the cloud web server) skips the pipeline entirely: scenes
# are processed by the scheduled batch job, and the web server only serves the website,
# stored detections, overlay images and the reference/ship lookups.
import os as _os

PROCESSING_ENABLED = _os.getenv("OILSPILL_API_PROCESSING", "1") != "0"
if PROCESSING_ENABLED:
    try:
        from . import pipeline
        PIPELINE_ERROR: str | None = None
    except ImportError as _exc:  # pragma: no cover - environment specific
        pipeline = None  # type: ignore[assignment]
        PIPELINE_ERROR = f"{type(_exc).__name__}: {_exc}"
else:
    pipeline = None  # type: ignore[assignment]
    PIPELINE_ERROR = None
from .database import (Detection, Run, Scene, detection_feature, load_overlay, make_engine,
                       safe_key)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("oilwatch")

engine = make_engine()
SessionLocal = sessionmaker(engine, expire_on_commit=False)
executor = ThreadPoolExecutor(max_workers=1)  # one scene at a time: bounded RAM
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if pipeline is None:
        log.error("scene processing disabled - pipeline import failed: %s", PIPELINE_ERROR)
        lm = None
    else:
        lm = pipeline.load_model()
    log.info("model: %s", lm.card.model_version if lm else "NONE (train on Colab, copy into models/)")
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="OilWatch API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[config.FRONTEND_ORIGIN, "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    try:
        v = tuple(float(x) for x in bbox.split(","))
    except ValueError as exc:
        raise HTTPException(422, "bbox must be minx,miny,maxx,maxy") from exc
    if len(v) != 4 or v[0] > v[2] or v[1] > v[3] or not (-180 <= v[0] <= 180 and -90 <= v[1] <= 90):
        raise HTTPException(422, "bbox must be minx,miny,maxx,maxy in lon/lat")
    return v  # type: ignore[return-value]


def parse_date(s: str | None, default: date) -> date:
    if not s:
        return default
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise HTTPException(422, f"invalid date {s!r}, expected YYYY-MM-DD") from exc


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _need_pipeline():
    if pipeline is None:
        if not PROCESSING_ENABLED:
            raise HTTPException(503, "On-demand processing is off on this server. New Sentinel-1 scenes "
                                     "are processed automatically every 4 hours.")
        raise HTTPException(503, f"Scene processing unavailable on this machine: {PIPELINE_ERROR}")


def _model_card() -> dict | None:
    """model_card.json without loading the network (enough for status + model info)."""
    p = config.MODELS_DIR / "model_card.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@app.get("/api/health")
def health():
    lm = pipeline.get_model() if pipeline else None
    card = lm.card.raw if lm else (_model_card() if not PROCESSING_ENABLED else None)
    with SessionLocal() as s:
        n = s.scalar(select(func.count(Detection.id))) or 0
    return {"status": "ok", "model_loaded": card is not None, "pipeline_error": PIPELINE_ERROR,
            "processing": "on-demand" if lm else ("scheduled" if not PROCESSING_ENABLED else "unavailable"),
            "model_version": card.get("model_version") if card else None, "db_detections": n}


@app.get("/api/model-info")
def model_info():
    lm = pipeline.load_model() if pipeline else None
    if lm is None and not PROCESSING_ENABLED and (c := _model_card()):
        return {"loaded": True, "model_version": c.get("model_version"), "architecture": c.get("architecture"),
                "classes": c.get("classes"), "thresholds": c.get("thresholds", {}), "metrics": c.get("metrics"),
                "trained_on": c.get("trained_on"), "db_clip": c.get("db_clip"), "license": c.get("license"),
                "processing": "scheduled (every 4 hours)"}
    if lm is None:
        return {"loaded": False, "model_version": None, "architecture": None,
                "classes": {str(k): v for k, v in config.MODEL_CLASSES.items() if k},
                "thresholds": {}, "metrics": None, "trained_on": None,
                "message": "No model in models/. Run notebooks/colab_train.ipynb on Colab and copy "
                           "model.onnx + model_card.json into models/."}
    c = lm.card
    return {"loaded": True, "model_version": c.model_version, "architecture": c.architecture,
            "classes": c.classes, "thresholds": c.thresholds, "metrics": c.metrics,
            "trained_on": c.trained_on, "load_s": round(lm.predictor.load_s, 3),
            "db_clip": c.db_clip, "license": (c.raw or {}).get("license")}


# ---------------------------------------------------------------------------
# detections
# ---------------------------------------------------------------------------

@app.get("/api/detections")
def list_detections(
    bbox: str | None = None, start: str | None = None, end: str | None = None,
    min_confidence: float = Query(0.0, ge=0, le=1), classes: str | None = None,
    min_area_km2: float | None = Query(None, ge=0), max_area_km2: float | None = Query(None, ge=0),
    scene_id: str | None = None, limit: int = Query(2000, ge=1, le=10000),
):
    q = (select(Detection, Scene, Run).join(Scene, Detection.scene_fk == Scene.id)
         .join(Run, Detection.run_fk == Run.id).where(Detection.confidence >= min_confidence))
    b = parse_bbox(bbox)
    if b:
        q = q.where(Detection.maxx >= b[0], Detection.minx <= b[2], Detection.maxy >= b[1], Detection.miny <= b[3])
    if start:
        q = q.where(Detection.acquired_at >= datetime.combine(parse_date(start, date.min), datetime.min.time(), UTC))
    if end:
        q = q.where(Detection.acquired_at < datetime.combine(parse_date(end, date.max) + timedelta(days=1),
                                                             datetime.min.time(), UTC))
    if classes:
        try:
            cl = [int(c) for c in classes.replace("_", ",").split(",") if c]
        except ValueError as exc:
            raise HTTPException(422, "classes must be comma-separated ints") from exc
        q = q.where(Detection.cls.in_(cl))
    if min_area_km2 is not None:
        q = q.where(Detection.area_km2 >= min_area_km2)
    if max_area_km2 is not None:
        q = q.where(Detection.area_km2 <= max_area_km2)
    if scene_id:
        q = q.where(Scene.scene_id == scene_id)
    q = q.order_by(Detection.confidence.desc()).limit(limit)
    with SessionLocal() as s:
        feats = [detection_feature(d, sc, r) for d, sc, r in s.execute(q).all()]
    return {"type": "FeatureCollection", "features": feats, "numberReturned": len(feats)}


@app.get("/api/detections/{det_id}")
def get_detection(det_id: int):
    with SessionLocal() as s:
        row = s.execute(select(Detection, Scene, Run).join(Scene, Detection.scene_fk == Scene.id)
                        .join(Run, Detection.run_fk == Run.id).where(Detection.id == det_id)).first()
        if row is None:
            raise HTTPException(404, f"detection {det_id} not found")
        return detection_feature(*row)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

@app.get("/api/scenes")
def list_scenes(bbox: str | None = None, start: str | None = None, end: str | None = None):
    q = select(Scene, func.count(Detection.id)).outerjoin(Detection, Detection.scene_fk == Scene.id).group_by(Scene.id)
    b = parse_bbox(bbox)
    if b:
        q = q.where(Scene.maxx >= b[0], Scene.minx <= b[2], Scene.maxy >= b[1], Scene.miny <= b[3])
    if start:
        q = q.where(Scene.acquired_at >= datetime.combine(parse_date(start, date.min), datetime.min.time(), UTC))
    if end:
        q = q.where(Scene.acquired_at < datetime.combine(parse_date(end, date.max) + timedelta(days=1),
                                                         datetime.min.time(), UTC))
    with SessionLocal() as s:
        rows = s.execute(q.order_by(Scene.acquired_at.desc())).all()
    out = []
    for sc, n in rows:
        ov = _overlay_meta(safe_key(sc.scene_id))
        out.append({"scene_id": sc.scene_id, "acquired_at": sc.acquired_at.isoformat() if sc.acquired_at else None,
                    "bounds": [sc.minx, sc.miny, sc.maxx, sc.maxy], "image_bounds": ov["image_bounds"] if ov else None,
                    "n_detections": n, "orbit_direction": sc.orbit_direction,
                    "processed_at": sc.processed_at.isoformat() if sc.processed_at else None})
    return {"scenes": out}


def _overlay_meta(key: str) -> dict | None:
    with SessionLocal() as s:
        row = load_overlay(s, key, "quicklook")
        if row is not None:
            return row.meta
    return pipeline.overlay_meta(key) if pipeline else None


def _overlay_file(scene_id: str, name: str):
    from fastapi.responses import Response

    key = safe_key(scene_id)
    kind = name.removesuffix(".png")
    with SessionLocal() as s:
        row = load_overlay(s, key, kind)
        if row is not None:
            headers = {"X-Image-Bounds": json.dumps(row.meta.get("image_bounds")), "Cache-Control": "max-age=3600"}
            return Response(row.png, media_type="image/png", headers=headers)
    if pipeline is not None:  # local uploads keep their overlays on disk only
        p = pipeline.QUICKLOOK_DIR / key / name
        if p.exists():
            ov = pipeline.overlay_meta(key) or {}
            headers = {"X-Image-Bounds": json.dumps(ov.get("image_bounds")), "Cache-Control": "max-age=3600"}
            return FileResponse(p, media_type="image/png", headers=headers)
    raise HTTPException(404, f"no {name} for scene {scene_id}")


@app.get("/api/scenes/{scene_id}/quicklook.png")
def quicklook(scene_id: str):
    return _overlay_file(scene_id, "quicklook.png")


@app.get("/api/scenes/{scene_id}/prob.png")
def prob_png(scene_id: str):
    return _overlay_file(scene_id, "prob.png")


@app.get("/api/scenes/{scene_id}/overlay")
def overlay(scene_id: str):
    ov = _overlay_meta(safe_key(scene_id))
    if ov is None:
        raise HTTPException(404, "scene not processed")
    return ov


@app.get("/api/search-scenes")
def search_scenes(bbox: str, start: str | None = None, end: str | None = None,
                  limit: int = Query(20, ge=1, le=100)):
    _need_pipeline()
    b = parse_bbox(bbox)
    today = date.today()
    s = parse_date(start, today - timedelta(days=30)).isoformat()
    e = parse_date(end, today).isoformat()
    try:
        items = pipeline.search_scenes(b, s, e, limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Planetary Computer search failed: {exc}") from exc
    with SessionLocal() as ses:
        done = set(ses.scalars(select(Scene.scene_id)))
    for it in items:
        it["processed"] = it["scene_id"] in done
    return {"items": items}


# ---------------------------------------------------------------------------
# processing jobs
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    scene_id: str


def _job_update(job_id: str, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)


def _run_job(job_id: str, scene_id: str):
    _job_update(job_id, status="running", message="reading Sentinel-1 scene + running model")
    try:
        res = pipeline.process_pc_scene(scene_id, SessionLocal)
        _job_update(job_id, status="done", message=f"{res['n_detections']} slicks detected",
                    n_detections=res["n_detections"], scene_id=res["scene_id"], timings=res["timings"])
    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        _job_update(job_id, status="error", message=f"{type(exc).__name__}: {exc}")


@app.post("/api/process-scene")
def process_scene(req: ProcessRequest):
    _need_pipeline()
    if pipeline.get_model() is None:
        raise HTTPException(503, "No model loaded. Train with notebooks/colab_train.ipynb and copy "
                                 "model.onnx + model_card.json into models/.")
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued", "message": "queued",
                        "scene_id": req.scene_id, "n_detections": None}
    executor.submit(_run_job, job_id, req.scene_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return j


@app.post("/api/predict-geotiff")
async def predict_geotiff(file: UploadFile = File(...)):
    _need_pipeline()
    if pipeline.get_model() is None:
        raise HTTPException(503, "No model loaded.")
    name = (file.filename or "").lower()
    if not name.endswith((".tif", ".tiff")):
        raise HTTPException(415, "upload a GeoTIFF (.tif/.tiff)")
    with tempfile.TemporaryDirectory() as td:
        stem = Path(file.filename).stem[:80] or "upload"
        path = Path(td) / f"upload_{uuid.uuid4().hex[:6]}_{stem}.tif"
        size = 0
        with path.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file larger than {config.MAX_UPLOAD_BYTES // 2**20} MB")
                fh.write(chunk)
        try:
            with open(path, "rb") as fh:
                magic = fh.read(4)
            if magic[:2] not in (b"II", b"MM"):
                raise HTTPException(415, "not a TIFF file")
            res = pipeline.process_upload(path)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"scene_id": res["overlay_key"], "timings": res["timings"], "overlay": res["overlay"],
            "detections": res["detections"]}


# ---------------------------------------------------------------------------
# reference layer: SkyTruth Cerulean's own published slicks (clearly separate)
# ---------------------------------------------------------------------------

_cer_client = None


_tile_http = None
TILE_CACHE = config.DATA_DIR / "app" / "cerulean_tiles"

#: Oil classes shown by default on cerulean.skytruth.org (ANTHRO..COIN_VESSEL) and the
#: human-review classes it keeps (anything except NOT_OIL / LAND / SEA_ICE / ARTEFACT).
REF_CLASSES = "2,3,4,5,6,7,8"
REF_HITL_CLASSES = "2,3,4,5,6,7,8,9"


def reference_filter(min_confidence: float, slick_confidence: float) -> str:
    """CQL2 filter matching the Cerulean website's default view.

    Filtering on machine_confidence alone returns ~15x more slicks than the Cerulean
    site shows (measured: 4,226 vs 287 over Indian seas, 12 months) - most of the
    extra are low slick_confidence look-alikes or slicks humans rejected.
    """
    return (f"machine_confidence >= {min_confidence} AND slick_confidence >= {slick_confidence} "
            f"AND cls IN ({REF_CLASSES}) AND (hitl_cls IS NULL OR hitl_cls IN ({REF_HITL_CLASSES}))")


@app.get("/api/reference/tiles/{z}/{x}/{y}.pbf")
def cerulean_tile(z: int, x: int, y: int, start: str | None = None, end: str | None = None,
                  min_confidence: float = Query(0.7, ge=0, le=1),
                  slick_confidence: float = Query(0.8, ge=0, le=1)):
    """Global SkyTruth Cerulean slicks as Mapbox vector tiles (their public tipg API), cached on disk.

    Only light properties are requested and the time/confidence filter is applied
    server-side, so a tile is ~50-300 kB instead of tens of MB.
    """
    import hashlib

    import httpx
    from fastapi.responses import Response

    global _tile_http
    if not (0 <= z <= 14 and 0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(422, "invalid tile")
    today = date.today()
    s = parse_date(start, today - timedelta(days=365)).isoformat()
    e = parse_date(end, today).isoformat()
    params = {
        "properties": "id,machine_confidence,cls,hitl_cls,slick_timestamp,area",
        "datetime": f"{s}T00:00:00Z/{e}T23:59:59Z",
        "filter-lang": "cql2-text",
        "filter": reference_filter(min_confidence, slick_confidence),
        "limit": 10000,
    }
    key = hashlib.sha1(f"{z}/{x}/{y}?{sorted(params.items())}".encode()).hexdigest()[:20]
    cache = TILE_CACHE / f"{z}_{x}_{y}_{key}.pbf"
    headers = {"Cache-Control": "max-age=3600"}
    if cache.exists() and (datetime.now().timestamp() - cache.stat().st_mtime) < 6 * 3600:
        return Response(cache.read_bytes(), media_type="application/vnd.mapbox-vector-tile", headers=headers)
    if _tile_http is None:
        _tile_http = httpx.Client(timeout=90, headers={"User-Agent": "oilspill/0.1 (open-source Cerulean reproduction)"})
    url = f"{config.CERULEAN_BASE}/collections/public.slick_plus/tiles/WebMercatorQuad/{z}/{x}/{y}"
    try:
        r = _tile_http.get(url, params=params)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Cerulean tile error: {exc}") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"Cerulean tile HTTP {r.status_code}")
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return Response(r.content, media_type="application/vnd.mapbox-vector-tile", headers=headers)


@app.get("/api/reference/slick/{slick_id}")
def cerulean_slick(slick_id: int):
    """One SkyTruth Cerulean slick + its candidate sources (vessels / infrastructure)."""
    base = f"{config.CERULEAN_BASE}/collections"
    try:
        s = _cerulean().get(f"{base}/public.slick_plus/items", {"ids": str(slick_id), "limit": 1})
        src = _cerulean().get(f"{base}/public.source_plus/items",
                              {"limit": 50, "filter-lang": "cql2-text", "filter": f"slick_id = {slick_id}"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Cerulean API error: {exc}") from exc
    feats = s.get("features", [])
    if not feats:
        raise HTTPException(404, f"Cerulean slick {slick_id} not found")
    p = feats[0]["properties"]
    keep = ("id", "slick_timestamp", "machine_confidence", "slick_confidence", "cls", "hitl_cls",
            "hitl_cls_name", "area", "length", "s1_scene_id", "slick_url")
    sources = []
    for f in sorted(src.get("features", []), key=lambda f: f["properties"].get("source_rank") or 99):
        q = f["properties"]
        ident = q.get("mmsi_or_structure_id")
        sources.append({
            "source_type": q.get("source_type"), "mmsi_or_structure_id": ident,
            "score": q.get("source_collated_score"), "rank": q.get("source_rank"),
            "cerulean_source_url": q.get("source_url"),
            "vessel_lookup_url": f"https://www.marinetraffic.com/en/ais/details/ships/mmsi:{ident}"
            if q.get("source_type") == "VESSEL" and ident else None,
        })
    return {"slick": {k: p.get(k) for k in keep}, "sources": sources,
            "attribution": "SkyTruth Cerulean (cerulean.skytruth.org), CC BY-SA 4.0"}


_geo_cache: dict[str, list] = {}
_geo_lock = threading.Lock()
_geo_last = [0.0]


@app.get("/api/geocode")
def geocode(q: str = Query(..., min_length=2, max_length=120)):
    """Place/sea/country search via OpenStreetMap Nominatim.

    Nominatim usage policy: identify the app, <= 1 request/second, cache results,
    no autocomplete (the UI searches on Enter only).
    """
    import time

    import httpx

    key = q.strip().lower()
    if key in _geo_cache:
        return {"results": _geo_cache[key]}
    with _geo_lock:
        wait = 1.0 - (time.monotonic() - _geo_last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = httpx.get("https://nominatim.openstreetmap.org/search",
                          params={"q": q, "format": "jsonv2", "limit": 6},
                          headers={"User-Agent": "OilWatch/0.1 (local oil-slick research demo)"},
                          timeout=20)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"geocoder error: {exc}") from exc
        finally:
            _geo_last[0] = time.monotonic()
    out = []
    for p in r.json():
        bb = p.get("boundingbox")
        if not bb or len(bb) != 4:
            continue
        s, n, w, e = (float(v) for v in bb)
        out.append({"name": p.get("display_name", ""), "type": p.get("type") or p.get("category") or "",
                    "bbox": [w, s, e, n]})
    _geo_cache[key] = out
    return {"results": out}


def _cerulean():
    global _cer_client
    from oilspill.cerulean_api import CeruleanClient

    if _cer_client is None:
        _cer_client = CeruleanClient(cache_dir=config.DATA_DIR / "app" / "cerulean_cache", timeout=30.0, max_retries=2)
    return _cer_client


def _sources_note(matched: list, sources: list, acquired: datetime) -> str | None:
    """Plain-language reason when no ship is listed."""
    if sources:
        return None
    if not matched:
        return ("No archive record overlaps this slick, so no ship match is available for it yet.")
    age_h = (datetime.now(UTC) - acquired.astimezone(UTC)).total_seconds() / 3600
    if age_h < 96:
        return (f"This slick was imaged {age_h:.0f} h ago. Ship positions (AIS) arrive up to 3 days after the "
                "satellite pass, so the ship match is not published yet. Check again later.")
    return "No ship or platform was matched near this slick."


@app.get("/api/detections/{det_id}/sources")
def detection_sources(det_id: int):
    """Candidate polluters (vessels / infrastructure) for one of OUR detections.

    We have no AIS feed of our own. Instead we find the SkyTruth Cerulean slick(s)
    that overlap our polygon in the same Sentinel-1 pass (+-1 day) and return
    Cerulean's published source attribution for them (AIS vessels within
    -8 h/+6 h scored on proximity, timing and trajectory match). If Cerulean did
    not detect the same slick, the list is honestly empty.
    """
    from shapely.geometry import shape

    feat = get_detection(det_id)
    geom = shape(feat["geometry"])
    acq = feat["properties"]["acquired_at"]
    if not acq:
        return {"detection_id": det_id, "matched_cerulean_slicks": [], "sources": [],
                "note": "detection has no acquisition time"}
    t = datetime.fromisoformat(acq)
    minx, miny, maxx, maxy = geom.bounds
    pad = 0.02
    try:
        cer = list(_cerulean().iter_slicks(
            bbox=(minx - pad, miny - pad, maxx + pad, maxy + pad),
            datetime_range=f"{(t - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}/"
                           f"{(t + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            max_features=200))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Cerulean API error: {exc}") from exc

    matched = []
    for f in cer:
        try:
            g = shape(f["geometry"])
        except Exception:  # noqa: BLE001
            continue
        if not g.intersects(geom):
            continue
        inter = g.intersection(geom).area
        matched.append({"slick_id": f["properties"]["id"], "overlap_of_ours": inter / geom.area if geom.area else 0,
                        "iou": inter / g.union(geom).area, "cls": f["properties"].get("cls"),
                        "machine_confidence": f["properties"].get("machine_confidence"),
                        "slick_url": f["properties"].get("slick_url")})
    matched.sort(key=lambda m: -m["iou"])

    sources = []
    if matched:
        ids = ",".join(str(m["slick_id"]) for m in matched[:10])
        try:
            payload = _cerulean().get(f"{config.CERULEAN_BASE}/collections/public.source_plus/items",
                                      {"limit": 100, "filter-lang": "cql2-text", "filter": f"slick_id IN ({ids})"})
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Cerulean API error: {exc}") from exc
        best = {}
        for f in payload.get("features", []):
            p = f["properties"]
            key = (p.get("source_type"), p.get("mmsi_or_structure_id"))
            if key not in best or (p.get("source_collated_score") or -9) > (best[key].get("source_collated_score") or -9):
                best[key] = p
        for p in sorted(best.values(), key=lambda p: -(p.get("source_collated_score") or -9)):
            ident = p.get("mmsi_or_structure_id")
            is_vessel = p.get("source_type") == "VESSEL"
            sources.append({
                "source_type": p.get("source_type"), "mmsi_or_structure_id": ident,
                "score": p.get("source_collated_score"), "rank": p.get("source_rank"),
                "cerulean_slick_id": p.get("slick_id"), "cerulean_source_url": p.get("source_url"),
                "vessel_lookup_url": f"https://www.marinetraffic.com/en/ais/details/ships/mmsi:{ident}" if is_vessel and ident else None,
            })
    return {"detection_id": det_id, "acquired_at": acq, "matched_cerulean_slicks": matched[:10],
            "sources": sources,
            "method": "Cerulean (SkyTruth) AIS-based attribution of the overlapping Cerulean slick",
            "attribution": "SkyTruth Cerulean (cerulean.skytruth.org), CC BY-SA 4.0",
            "note": _sources_note(matched, sources, t)}


@app.get("/api/reference/cerulean")
def cerulean_reference(bbox: str, start: str | None = None, end: str | None = None,
                       min_confidence: float = Query(0.7, ge=0, le=1), limit: int = Query(500, ge=1, le=2000)):
    global _cer_client
    from oilspill.cerulean_api import CeruleanClient

    b = parse_bbox(bbox)
    today = date.today()
    s = parse_date(start, today - timedelta(days=365)).isoformat()
    e = parse_date(end, today).isoformat()
    if _cer_client is None:
        _cer_client = CeruleanClient(cache_dir=config.DATA_DIR / "app" / "cerulean_cache", timeout=30.0, max_retries=2)
    try:
        feats = list(_cer_client.iter_slicks(cql2=reference_filter(min_confidence, 0.8), bbox=b,
                                             datetime_range=f"{s}T00:00:00Z/{e}T23:59:59Z",
                                             sortby="-slick_timestamp",
                                             max_features=limit, limit=min(limit, 500)))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Cerulean API error: {exc}") from exc
    keep = ("id", "machine_confidence", "slick_confidence", "cls", "hitl_cls", "slick_timestamp", "area", "s1_scene_id", "slick_url")
    for f in feats:
        f["properties"] = {k: f["properties"].get(k) for k in keep}
    return {"type": "FeatureCollection", "features": feats,
            "attribution": "SkyTruth Cerulean (cerulean.skytruth.org), CC BY-SA 4.0"}


# ---------------------------------------------------------------------------
# Website (built React app). Registered last so every /api route wins.
# ---------------------------------------------------------------------------

_DIST = Path(_os.getenv("OILSPILL_FRONTEND_DIST", str(Path(__file__).resolve().parents[2] / "frontend" / "dist")))

if _DIST.is_dir():
    @app.get("/{path:path}", include_in_schema=False)
    def website(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        f = (_DIST / path).resolve()
        if path and f.is_file() and _DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(_DIST / "index.html")  # client-side routes like /slicks/123
