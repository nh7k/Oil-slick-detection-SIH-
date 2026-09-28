"""Phase 6: full-scene evaluation with the exact production pipeline.

* thresholds (p_seed, p_trim) are tuned on VAL scenes only
* TEST is evaluated once, against human-reviewed (Tier-1) masks only
  (unreviewed machine labels are ignore, never counted as hit or miss)
* pixel metrics (binary oil + per class), instance metrics (polygon IoU >= 0.1),
  confusion matrix

Works with either the torch checkpoint (Colab) or the exported ONNX model.

Run: python -m oilspill.evaluate --data DATA --model RUN/export  (ONNX)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import shape

from . import config
from .infer import ModelCard, OnnxPredictor, predict_scene
from .postprocess import extract_slicks
from .scene import load_scene
from .metrics import ConfMat, _prf

log = logging.getLogger(__name__)


def _read_mask(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as d:
        return d.read(1)


def instance_scores(feats: list[dict], gt_mask: np.ndarray, window, iou_thr: float) -> tuple[int, int, int]:
    """Match predicted polygons to GT connected components by IoU (pixel space)."""
    from scipy import ndimage

    gt_oil = np.isin(gt_mask, config.OIL_CLASSES)
    ignore = gt_mask == config.IGNORE_INDEX
    lab, n = ndimage.label(gt_oil, structure=np.ones((3, 3)))
    matched_gt = set()
    tp = fp = 0
    for f in feats:
        pm = rasterize([(shape(f["geometry"]), 1)], out_shape=gt_mask.shape,
                       transform=window.transform, fill=0, dtype="uint8").astype(bool)
        if (pm & ignore).sum() > 0.5 * pm.sum():
            continue  # prediction lies on unreviewed/ignored area -> not scored
        ids = np.unique(lab[pm])
        ids = ids[ids > 0]
        best = 0.0
        best_id = None
        for i in ids:
            g = lab == i
            iou = (g & pm).sum() / (g | pm).sum()
            if iou > best:
                best, best_id = iou, i
        if best >= iou_thr:
            tp += 1
            matched_gt.add(best_id)
        else:
            fp += 1
    fn = n - len(matched_gt)
    return tp, fp, fn


class TorchPredictor:
    """Wrap best.pt for GPU evaluation on Colab (same predictor interface as ONNX)."""

    def __init__(self, ckpt: Path):
        import torch

        from .train import build_model

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(ck["encoder"], None, ck["arch"]).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.no_grad(), self.torch.autocast(self.dev.type, enabled=self.dev.type == "cuda"):
            return self.model(self.torch.from_numpy(batch).to(self.dev)).float().cpu().numpy()


def run_scenes(data: Path, ids: list[str], predictor, card: ModelCard, grid: list[dict],
               suffixes: tuple[str, ...]) -> dict:
    """Probabilities computed ONCE per scene, then scored for every threshold/mask combo."""
    res = {(gi, sfx): {"cm": ConfMat(), "inst": [0, 0, 0]} for gi in range(len(grid)) for sfx in suffixes}
    n_scenes = 0
    for pid in ids:
        sp = data / "scenes" / f"{pid}.tif"
        gts = {sfx: data / "masks" / f"{pid}{sfx}.tif" for sfx in suffixes}
        if not sp.exists() or not all(p.exists() for p in gts.values()):
            continue
        gts = {sfx: _read_mask(p) for sfx, p in gts.items()}
        if not any(np.isin(g, config.OIL_CLASSES).any() for g in gts.values()):
            continue
        scene = load_scene(sp)
        probs, _ = predict_scene(scene, predictor, card, batch_size=8)
        n_scenes += 1
        for gi, ov in enumerate(grid):
            feats = extract_slicks(probs, scene.window, scene.valid, overrides=ov)
            pred = np.zeros(scene.db.shape, np.uint8)
            if feats:
                pred = rasterize([(shape(f["geometry"]), f["properties"]["cls"]) for f in feats],
                                 out_shape=pred.shape, transform=scene.window.transform, fill=0,
                                 dtype="uint8")
            for sfx, gt in gts.items():
                if not np.isin(gt, config.OIL_CLASSES).any():
                    continue
                r = res[(gi, sfx)]
                r["cm"].update(pred, gt)
                tp, fp, fn = instance_scores(feats, gt, scene.window, config.EVAL_INSTANCE_IOU)
                r["inst"][0] += tp
                r["inst"][1] += fp
                r["inst"][2] += fn
        log.info("evaluated %s (%d)", pid, n_scenes)
    out = {}
    for (gi, sfx), r in res.items():
        s = r["cm"].summary()
        s["instance"] = _prf(*r["inst"])
        s["thresholds"] = grid[gi]
        s["n_scenes"] = n_scenes
        out[(gi, sfx)] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="dir with model.onnx + model_card.json")
    ap.add_argument("--torch-ckpt", default=None, help="use best.pt on GPU instead of ONNX (Colab)")
    ap.add_argument("--max-val-scenes", type=int, default=40)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data, mdir = Path(args.data), Path(args.model)
    card = ModelCard.load(mdir / "model_card.json")
    predictor = TorchPredictor(Path(args.torch_ckpt)) if args.torch_ckpt else OnnxPredictor(mdir / "model.onnx")
    splits = pd.read_csv(data / "meta" / "splits.csv")

    grid = [{"p_seed": s, "p_trim": t} for s in (0.6, 0.75, 0.9) for t in (0.3, 0.5)]
    val_ids = splits[splits.split == "val"].pc_id.tolist()[: args.max_val_scenes]
    # Val has few human-reviewed scenes, so tune on the full (T1+T2) mask.
    val = run_scenes(data, val_ids, predictor, card, grid, suffixes=("",))
    best_i = max(range(len(grid)),
                 key=lambda i: val[(i, "")]["instance"]["f1_dice"] + val[(i, "")]["OIL_binary"]["f1_dice"])
    best = grid[best_i]
    log.info("val-tuned thresholds: %s", best)

    test_ids = splits[splits.split == "test"].pc_id.tolist()
    test = run_scenes(data, test_ids, predictor, card, [best], suffixes=("_t1", ""))
    t1, allm = test[(0, "_t1")], test[(0, "")]
    report = {
        "model_version": card.model_version,
        "tuned_thresholds": best,
        "val_grid": {str(grid[i]): {"oil_pixel_f1": val[(i, "")]["OIL_binary"]["f1_dice"],
                                    "instance_f1": val[(i, "")]["instance"]["f1_dice"]}
                     for i in range(len(grid))},
        "test_T1_human_reviewed": t1,
        "test_T1plusT2_agreement_with_cerulean": allm,
        "instance_iou_threshold": config.EVAL_INSTANCE_IOU,
        "note": ("T1 = human-reviewed Cerulean slicks only (unreviewed = ignore). T1+T2 measures agreement "
                 "with Cerulean's own model output, not ground truth. Cerulean's published pixel F1 0.532 "
                 "was measured on THEIR private test set and is NOT comparable."),
    }
    (mdir / "test_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("TEST T1 (%d scenes): oil pixel P=%.3f R=%.3f F1=%.3f IoU=%.3f | instance P=%.3f R=%.3f F1=%.3f",
             t1["n_scenes"], t1["OIL_binary"]["precision"], t1["OIL_binary"]["recall"],
             t1["OIL_binary"]["f1_dice"], t1["OIL_binary"]["iou"], t1["instance"]["precision"],
             t1["instance"]["recall"], t1["instance"]["f1_dice"])
    raw = card.raw
    raw["thresholds"].update(best)
    raw["metrics"] = {"test_T1_oil_pixel": t1["OIL_binary"], "test_T1_instance": t1["instance"],
                      "test_T1plusT2_oil_pixel": allm["OIL_binary"],
                      "test_T1plusT2_instance": allm["instance"],
                      "per_class_T1": {k: t1[k] for k in ("INFRA", "NATURAL", "VESSEL")}}
    (mdir / "model_card.json").write_text(json.dumps(raw, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
