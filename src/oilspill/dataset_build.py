"""Phase 2: build the training dataset from real Cerulean labels + real S1 imagery.

Per scene we store, on the Cerulean grid:

* ``scenes/<scene>.tif``  - uint16 pseudo-dB*100 (0 = nodata)
* ``masks/<scene>.tif``   - uint8 {0,1,2,3,255} training mask (T1 + T2 labels)
* ``masks/<scene>_t1.tif`` - uint8 evaluation mask from human-reviewed slicks only
  (255 everywhere if the scene has no reviewed positive)
* ``status/<scene>.json`` - done/failed + stats (makes the build resumable)

Training crops are sampled on the fly from these rasters (see ``train.py``),
so no tile explosion on disk.

Run:  python -m oilspill.dataset_build --scenes-per-aoi 60 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .cerulean_api import CeruleanClient, label_tier, model_class
from .georef import write_grid_geotiff
from .labels import rasterize_slicks, slick_geometry, tier1_mask
from .s1_source import find_item
from .scene import prepare_item, save_scene

log = logging.getLogger(__name__)

POS_SQL = ",".join(str(c) for c in config.POSITIVE_CLS)
NEG_SQL = ",".join(str(c) for c in config.HARD_NEGATIVE_CLS)


def data_paths(root: Path) -> dict[str, Path]:
    p = {
        "scenes": root / "scenes",
        "masks": root / "masks",
        "status": root / "status",
        "meta": root / "meta",
    }
    for d in p.values():
        d.mkdir(parents=True, exist_ok=True)
    return p


def end_date() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT00:00:00Z")


def select_scenes(
    client: CeruleanClient, scenes_per_aoi: int, start: str = config.DATASET_START_DATE,
) -> pd.DataFrame:
    """Pick scene ids per AOI: reviewed-positive first, then hard-negative, then T2."""
    dt = f"{start}T00:00:00Z/{end_date()}"
    rows: list[dict] = []
    for aoi in config.AOIS:
        seen: dict[str, str] = {}
        queries = [
            ("T1", f"hitl_cls IN ({POS_SQL})", scenes_per_aoi),
            ("HN", f"hitl_cls IN ({NEG_SQL})", max(scenes_per_aoi // 5, 3)),
            ("T2", f"cls IN ({POS_SQL}) AND machine_confidence >= {config.T2_MIN_CONFIDENCE}",
             scenes_per_aoi * 15),
        ]
        for why, cql, max_feats in queries:
            if len(seen) >= scenes_per_aoi:
                break
            try:
                feats = list(client.iter_slicks(cql2=cql, bbox=aoi.bbox, datetime_range=dt,
                                                max_features=max_feats, limit=200))
            except Exception as exc:  # noqa: BLE001
                log.warning("query %s/%s failed: %s", aoi.name, why, exc)
                continue
            # The API returns rows in id order (i.e. clustered in time); shuffle so the
            # chosen scenes spread across the whole date range.
            random.Random(config.SPLIT_SEED).shuffle(feats)
            for f in feats:
                sid = f["properties"].get("s1_scene_id")
                if sid and sid not in seen:
                    seen[sid] = why
                    rows.append({
                        "scene_id": sid, "aoi": aoi.name, "reason": why,
                        "slick_timestamp": f["properties"].get("slick_timestamp"),
                    })
                if len(seen) >= scenes_per_aoi:
                    break
        log.info("AOI %s: %d scenes", aoi.name, len(seen))
    return pd.DataFrame(rows)


def build_one(client: CeruleanClient, scene_id: str, aoi: str, paths: dict[str, Path]) -> dict:
    status_path = paths["status"] / f"{scene_id}.json"
    if status_path.exists():
        st = json.loads(status_path.read_text())
        if st.get("status") == "done":
            return st

    try:
        feats = client.slicks_in_scene(scene_id)
        item = find_item(scene_id)
        if item is None:
            raise RuntimeError("no Planetary Computer item")
        scene = prepare_item(item)
        mask, stats = rasterize_slicks(feats, scene.window)
        mask[~scene.valid] = config.IGNORE_INDEX
        t1 = tier1_mask(feats, scene.window)
        t1[~scene.valid] = config.IGNORE_INDEX

        # hard-negative + positive centroids -> crop sampling hints
        hints = []
        for f in feats:
            g = slick_geometry(f)
            if g is None:
                continue
            cls = model_class(f["properties"])
            tier = label_tier(f["properties"])
            c = g.representative_point()
            kind = "neg" if cls == 0 else ("pos" if cls in config.OIL_CLASSES and tier else "ign")
            hints.append({"lon": c.x, "lat": c.y, "kind": kind, "cls": cls, "tier": tier})

        pc_id = item["id"]
        save_scene(scene, paths["scenes"] / f"{pc_id}.tif")
        write_grid_geotiff(paths["masks"] / f"{pc_id}.tif", mask, scene.window, dtype="uint8")
        write_grid_geotiff(paths["masks"] / f"{pc_id}_t1.tif", t1, scene.window, dtype="uint8")

        valid = scene.valid
        b = scene.bounds
        st = {
            "status": "done", "scene_id": scene_id, "pc_id": pc_id, "aoi": aoi,
            "datetime": scene.meta.get("datetime"),
            "orbit_direction": scene.meta.get("orbit_direction"),
            "bounds": b, "center_lon": (b[0] + b[2]) / 2, "center_lat": (b[1] + b[3]) / 2,
            "shape": list(scene.db.shape), "valid_px": int(valid.sum()),
            "pos_px": {str(k): int((mask == k).sum()) for k in config.OIL_CLASSES},
            "t1_pos_px": int(np.isin(t1, config.OIL_CLASSES).sum()),
            "n_slicks": len(feats), "label_stats": stats, "hints": hints,
            "db_p1": float(np.percentile(scene.db[valid], 1)) if valid.any() else None,
            "db_p99": float(np.percentile(scene.db[valid], 99)) if valid.any() else None,
            "read_s": scene.meta.get("read_s"), "warp_s": scene.meta.get("warp_s"),
        }
    except Exception as exc:  # noqa: BLE001
        st = {"status": "failed", "scene_id": scene_id, "aoi": aoi,
              "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
        log.warning("scene %s failed: %s", scene_id, st["error"])
    status_path.write_text(json.dumps(st))
    return st


def make_splits(done: list[dict], seed: int = config.SPLIT_SEED) -> pd.DataFrame:
    """Scene-grouped, spatial (2-degree cell) + temporal split.

    * test  = scenes in the last TEMPORAL_TEST_MONTHS  UNION  scenes in ~15% of cells
    * val   = scenes in another ~15% of cells
    * train = the rest
    A scene belongs to exactly one split (assert below).
    """
    df = pd.DataFrame([{k: d[k] for k in ("pc_id", "scene_id", "aoi", "datetime",
                                          "center_lon", "center_lat", "t1_pos_px")}
                       for d in done])
    df["cell"] = (np.floor(df.center_lon / config.SPLIT_CELL_DEG).astype(int).astype(str) + "_"
                  + np.floor(df.center_lat / config.SPLIT_CELL_DEG).astype(int).astype(str))
    df["acq"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    cutoff = df["acq"].max() - pd.DateOffset(months=config.TEMPORAL_TEST_MONTHS)

    rng = np.random.default_rng(seed)
    cells = np.array(sorted(df.cell.unique()))
    rng.shuffle(cells)
    n_test = max(1, round(len(cells) * config.TEST_CELL_FRAC))
    n_val = max(1, round(len(cells) * config.VAL_CELL_FRAC))
    test_cells, val_cells = set(cells[:n_test]), set(cells[n_test:n_test + n_val])

    def assign(r) -> str:
        if r.cell in test_cells or r.acq >= cutoff:
            return "test"
        if r.cell in val_cells:
            return "val"
        return "train"

    df["split"] = df.apply(assign, axis=1)
    # Spatial leakage guard: a cell used by test (spatially) must not appear in train/val.
    assert not (set(df[df.split == "train"].cell) & test_cells), "test cell leaked into train"
    assert not (set(df[df.split == "train"].cell) & val_cells), "val cell leaked into train"
    assert df.pc_id.is_unique, "scene appears twice"
    return df.drop(columns=["acq"])


def compute_clip(done: list[dict], split_df: pd.DataFrame) -> tuple[float, float]:
    train_ids = set(split_df[split_df.split == "train"].pc_id)
    p1 = [d["db_p1"] for d in done if d["pc_id"] in train_ids and d.get("db_p1")]
    p99 = [d["db_p99"] for d in done if d["pc_id"] in train_ids and d.get("db_p99")]
    return float(np.median(p1)), float(np.median(p99))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(config.DATA_DIR / "dataset"))
    ap.add_argument("--scenes-per-aoi", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.root)
    paths = data_paths(root)
    client = CeruleanClient(cache_dir=root / "cache" / "cerulean")

    sel_path = paths["meta"] / "selected_scenes.csv"
    if sel_path.exists():
        sel = pd.read_csv(sel_path)
    else:
        sel = select_scenes(client, args.scenes_per_aoi)
        sel.to_csv(sel_path, index=False)
    if args.max_scenes:
        sel = sel.head(args.max_scenes)
    log.info("building %d scenes with %d workers", len(sel), args.workers)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, client, r.scene_id, r.aoi, paths): r.scene_id
                for r in sel.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            st = fut.result()
            results.append(st)
            if i % 10 == 0 or i == len(futs):
                n_ok = sum(r["status"] == "done" for r in results)
                log.info("progress %d/%d (done=%d)", i, len(futs), n_ok)

    done = [r for r in results if r["status"] == "done"]
    failed = [r for r in results if r["status"] != "done"]
    # de-duplicate (two Cerulean ids can map to one PC item)
    uniq = {d["pc_id"]: d for d in done}
    done = list(uniq.values())
    if len(done) < 5:
        log.error("only %d scenes built; aborting split", len(done))
        return 1

    split_df = make_splits(done)
    clip = compute_clip(done, split_df)
    split_df.to_csv(paths["meta"] / "splits.csv", index=False)
    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "n_selected": int(len(sel)), "n_done": len(done), "n_failed": len(failed),
        "db_clip": clip,
        "split_counts": split_df.split.value_counts().to_dict(),
        "t1_scenes_by_split": split_df[split_df.t1_pos_px > 0].split.value_counts().to_dict(),
        "pos_px_total": {str(k): int(sum(d["pos_px"][str(k)] for d in done)) for k in config.OIL_CLASSES},
        "failures": [{"scene_id": f["scene_id"], "error": f.get("error")} for f in failed][:50],
        "grid": {"pixel_deg": config.PIXEL_DEG, "crs": config.CRS},
    }
    (paths["meta"] / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("dataset meta: %s", json.dumps({k: meta[k] for k in
             ("n_done", "n_failed", "db_clip", "split_counts", "t1_scenes_by_split", "pos_px_total")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
