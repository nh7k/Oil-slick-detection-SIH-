"""The single end-to-end detection pipeline used by the API and the batch job.

scene (PC item or uploaded GeoTIFF) -> grid raster -> ONNX model -> probabilities
-> Cerulean-style polygons -> DB rows + quicklook/probability PNGs (Web Mercator).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

from oilspill import config
from oilspill.infer import ModelCard, OnnxPredictor, predict_scene
from oilspill.landmask import buffered_land
from oilspill.postprocess import extract_slicks
from oilspill.s1_source import catalog
from oilspill.scene import SceneRaster, prepare_geotiff, prepare_item, to_uint8

log = logging.getLogger(__name__)

APP_DATA = config.DATA_DIR / "app"
QUICKLOOK_DIR = APP_DATA / "scenes"


@dataclass
class LoadedModel:
    card: ModelCard
    predictor: OnnxPredictor
    path: Path


_model: LoadedModel | None = None
_model_lock = threading.Lock()


def model_dir() -> Path:
    return config.MODELS_DIR


def load_model() -> LoadedModel | None:
    """Load models/model.onnx + model_card.json if present (produced by the Colab notebook)."""
    global _model
    with _model_lock:
        onnx_path, card_path = model_dir() / "model.onnx", model_dir() / "model_card.json"
        if not (onnx_path.exists() and card_path.exists()):
            _model = None
            return None
        if _model is None or _model.path != onnx_path:
            card = ModelCard.load(card_path)
            pred = OnnxPredictor(onnx_path)
            log.info("loaded model %s in %.2fs", card.model_version, pred.load_s)
            _model = LoadedModel(card, pred, onnx_path)
        return _model


def get_model() -> LoadedModel | None:
    return _model or load_model()


# ---------------------------------------------------------------------------
# PNG overlays (reprojected to EPSG:3857 so MapLibre's linear image stretch is exact)
# ---------------------------------------------------------------------------

def _to_mercator(arr: np.ndarray, bounds, resampling) -> tuple[np.ndarray, list[float]]:
    h, w = arr.shape[-2:]
    src_t = from_bounds(*bounds, w, h)
    miny, maxy = max(bounds[1], -85.0), min(bounds[3], 85.0)
    dst_t, dw, dh = calculate_default_transform("EPSG:4326", "EPSG:3857", w, h,
                                                bounds[0], miny, bounds[2], maxy)
    bands = arr if arr.ndim == 3 else arr[None]
    out = np.zeros((bands.shape[0], dh, dw), dtype=bands.dtype)
    for i in range(bands.shape[0]):
        reproject(bands[i], out[i], src_transform=src_t, src_crs="EPSG:4326", dst_transform=dst_t,
                  dst_crs="EPSG:3857", resampling=resampling, src_nodata=0, dst_nodata=0)
    # corners of the mercator raster back in lon/lat
    from rasterio.warp import transform as tr

    x0, y0 = dst_t.c, dst_t.f
    x1, y1 = x0 + dw * dst_t.a, y0 + dh * dst_t.e
    lons, lats = tr("EPSG:3857", "EPSG:4326", [x0, x1], [y1, y0])
    return (out if arr.ndim == 3 else out[0]), [lons[0], lats[0], lons[1], lats[1]]


def _magma(p: np.ndarray) -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps["magma"]
    rgba = (cmap(np.clip(p, 0, 1)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(p >= 0.2, np.clip(80 + p * 175, 0, 255), 0).astype(np.uint8)
    return rgba


#: Longest side of stored overlay PNGs; keeps each image to a few hundred kB in the database.
OVERLAY_MAX_PX = 1600


def _png_bytes(img: Image.Image) -> bytes:
    import io

    if max(img.size) > OVERLAY_MAX_PX:
        f = OVERLAY_MAX_PX / max(img.size)
        img = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_overlays(scene: SceneRaster, probs: np.ndarray, card: ModelCard, key: str) -> dict:
    """Render quicklook + probability PNGs. Returns meta plus the PNG bytes under "images"."""
    out = QUICKLOOK_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    u8 = to_uint8(scene.db, card.db_clip)
    merc, merc_bounds = _to_mercator(u8, scene.bounds, Resampling.average)
    ql = _png_bytes(Image.fromarray(np.stack([merc, np.where(merc > 0, 255, 0).astype(np.uint8)], axis=-1), "LA"))

    p_oil = probs[1:].sum(axis=0)
    p8 = np.clip(np.round(p_oil * 254) + 1, 1, 255).astype(np.uint8)
    p8[~scene.valid] = 0
    pm, _ = _to_mercator(p8, scene.bounds, Resampling.bilinear)
    pf = np.where(pm > 0, (pm.astype(np.float32) - 1) / 254, 0)
    pr = _png_bytes(Image.fromarray(_magma(pf), "RGBA"))
    meta = {"bounds": list(scene.bounds), "image_bounds": merc_bounds}
    # local copies too (handy for debugging; the database copy is authoritative)
    (out / "quicklook.png").write_bytes(ql)
    (out / "prob.png").write_bytes(pr)
    (out / "overlay.json").write_text(json.dumps(meta))
    return {**meta, "images": {"quicklook": ql, "prob": pr}}


def overlay_meta(key: str) -> dict | None:
    p = QUICKLOOK_DIR / key / "overlay.json"
    return json.loads(p.read_text()) if p.exists() else None  # API falls back to the database


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

from app.database import safe_key  # noqa: E402  (shared with the API, which may run without this module)


def run_on_scene(scene: SceneRaster, lm: LoadedModel, extra_timings: dict | None = None) -> tuple[list, dict, dict]:
    t0 = time.perf_counter()
    probs, tinf = predict_scene(scene, lm.predictor, lm.card, batch_size=4)
    t1 = time.perf_counter()
    try:
        land = buffered_land(scene.bounds)
    except Exception as exc:  # noqa: BLE001
        log.warning("land mask unavailable (%s); continuing without it", exc)
        land = None
    feats = extract_slicks(probs, scene.window, scene.valid, land=land, overrides=lm.card.thresholds)
    t2 = time.perf_counter()
    overlay = write_overlays(scene, probs, lm.card, safe_key(scene.scene_id))
    timings = {**(extra_timings or {}), **{k: round(v, 3) for k, v in tinf.items()},
               "predict_total_s": round(t1 - t0, 3), "postprocess_s": round(t2 - t1, 3),
               "overlay_s": round(time.perf_counter() - t2, 3)}
    return feats, timings, overlay


def fetch_item(scene_id: str) -> dict:
    from oilspill.s1_source import find_item

    item = find_item(scene_id)
    if item is None:
        raise LookupError(f"scene {scene_id} not found on Planetary Computer")
    return item


def process_pc_scene(scene_id: str, session_factory) -> dict:
    from app.database import store_run

    lm = get_model()
    if lm is None:
        raise RuntimeError("no model loaded: put model.onnx + model_card.json into models/")
    started = datetime.now().astimezone()
    item = fetch_item(scene_id)
    t0 = time.perf_counter()
    scene = prepare_item(item)
    prep = {"read_s": scene.meta.get("read_s"), "warp_s": scene.meta.get("warp_s"),
            "prepare_total_s": round(time.perf_counter() - t0, 3)}
    feats, timings, overlay = run_on_scene(scene, lm, prep)
    acq = scene.meta.get("datetime")
    meta = {"scene_id": item["id"], "bounds": list(scene.bounds), "started_at": started,
            "acquired_at": datetime.fromisoformat(acq.replace("Z", "+00:00")) if acq else None,
            "orbit_direction": scene.meta.get("orbit_direction")}
    from app.database import save_overlays

    images = overlay.pop("images")
    with session_factory() as s:
        save_overlays(s, safe_key(item["id"]), images, overlay)
        run = store_run(s, meta, lm.card.model_version, lm.card.thresholds, timings, feats)  # commits
        run_id = run.id
    return {"scene_id": item["id"], "n_detections": len(feats), "run_id": run_id,
            "timings": timings, "overlay": overlay}


def process_upload(path: Path) -> dict:
    lm = get_model()
    if lm is None:
        raise RuntimeError("no model loaded: put model.onnx + model_card.json into models/")
    scene = prepare_geotiff(path)
    feats, timings, overlay = run_on_scene(scene, lm)
    overlay.pop("images", None)  # upload overlays are served from the local files only
    return {"scene_id": scene.scene_id, "overlay_key": safe_key(scene.scene_id), "timings": timings,
            "overlay": overlay, "detections": {"type": "FeatureCollection", "features": feats,
                                               "model_version": lm.card.model_version}}


def sea_fraction(footprint: dict) -> float:
    """Fraction of a scene footprint that is sea (Natural Earth land), in degree-area."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    from oilspill.landmask import land_in_bounds

    fp = shape(footprint)
    land = [g.intersection(fp) for g in land_in_bounds(fp.bounds)]
    land_area = unary_union(land).area if land else 0.0
    return max(0.0, 1.0 - land_area / fp.area) if fp.area else 0.0


def search_scenes(bbox: tuple[float, float, float, float], start: str, end: str, limit: int = 20) -> list[dict]:
    search = catalog().search(collections=[config.PC_COLLECTION], bbox=list(bbox),
                              datetime=f"{start}T00:00:00Z/{end}T23:59:59Z", limit=min(limit, 250),
                              sortby=[{"field": "datetime", "direction": "desc"}])
    out = []
    for it in search.items():
        if "vv" not in it.assets or "_IW_" not in it.id:
            continue
        out.append({"scene_id": it.id, "acquired_at": it.properties.get("datetime"),
                    "bounds": list(it.bbox), "orbit_direction": it.properties.get("sat:orbit_state"),
                    "footprint": it.geometry})
        if len(out) >= limit:
            break
    return out
