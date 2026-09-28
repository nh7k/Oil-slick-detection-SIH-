"""Phase 12 batch job: detect slicks in recent Sentinel-1 scenes over an AOI.

Same pipeline + same DB as the API. Idempotent: scenes already processed with the
current model version are skipped.

    python scripts/run_batch.py                       # default AOI (Arabian Sea / Kerala), last 14 days
    python scripts/run_batch.py --bbox 74.5,9.5,76.5,11 --start 2026-06-15 --end 2026-06-25
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from oilspill import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default=None, help="minx,miny,maxx,maxy (default: all Indian seas)")
    ap.add_argument("--start", default=(date.today() - timedelta(days=14)).isoformat())
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--max-scenes", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="re-process scenes already done by this model")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("batch")

    # Only one batch at a time (the 4-hourly task may fire while a backfill runs).
    import os
    import time as _time
    lock = config.DATA_DIR / "run_batch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and _time.time() - lock.stat().st_mtime < 12 * 3600:
        log.info("another batch is running (%s); exiting", lock)
        return 0
    lock.write_text(str(os.getpid()))
    try:
        return _run(args, log)
    finally:
        lock.unlink(missing_ok=True)


def _run(args, log) -> int:
    from app import pipeline
    from app.database import Run, Scene, make_engine

    lm = pipeline.load_model()
    if lm is None:
        log.error("no model in %s - train with notebooks/colab_train.ipynb first", config.MODELS_DIR)
        return 2
    session_factory = sessionmaker(make_engine(), expire_on_commit=False)
    bboxes = ([tuple(float(v) for v in args.bbox.split(","))] if args.bbox
              else [a.bbox for a in config.INDIA_AOIS])
    items, seen = [], set()
    for bbox in bboxes:
        for it in pipeline.search_scenes(bbox, args.start, args.end, limit=args.max_scenes):
            if it["scene_id"] not in seen:
                seen.add(it["scene_id"])
                items.append(it)
    items.sort(key=lambda it: it["acquired_at"] or "", reverse=True)
    log.info("%d scenes in %s %s..%s", len(items), bboxes, args.start, args.end)

    with session_factory() as s:
        done = set(s.scalars(select(Scene.scene_id).join(Run, Run.scene_fk == Scene.id)
                             .where(Run.model_version == lm.card.model_version)))
    total = 0
    for it in items:
        if it["scene_id"] in done and not args.force:
            log.info("skip %s (already processed by %s)", it["scene_id"], lm.card.model_version)
            continue
        try:
            sea = pipeline.sea_fraction(it["footprint"])
            if sea < config.MIN_SEA_FRACTION:
                log.info("skip %s (%.0f%% sea - land scene)", it["scene_id"], sea * 100)
                continue
            res = pipeline.process_pc_scene(it["scene_id"], session_factory)
            total += res["n_detections"]
            log.info("%s -> %d slicks  %s", it["scene_id"], res["n_detections"], res["timings"])
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", it["scene_id"], exc)
    log.info("batch done: %d new detections", total)

    # Keep the free-tier database small: overlay images older than 60 days are dropped
    # (detections themselves are kept).
    from datetime import UTC, datetime

    from app.database import OverlayImage
    cutoff = datetime.now(UTC) - timedelta(days=60)
    with session_factory() as s:
        n = s.query(OverlayImage).filter(OverlayImage.created_at < cutoff).delete()
        s.commit()
    if n:
        log.info("pruned %d old overlay images", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
