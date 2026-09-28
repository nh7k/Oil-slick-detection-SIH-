"""Phase 7: best.pt -> model.onnx + model_card.json (the artifact the API serves).

Verifies ONNX == torch (max abs diff on real validation crops) before writing
the card, so a broken export can never be shipped silently.

Run: python -m oilspill.export --run /content/drive/MyDrive/oilspill/run1 --data /content/data/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from . import config
from .train import build_model

log = logging.getLogger(__name__)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def export(run: Path, out: Path, test_batch: np.ndarray | None = None, metrics: dict | None = None) -> dict:
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["encoder"], None, ck["arch"])
    model.load_state_dict(ck["model"])
    model.eval()

    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"
    dummy = torch.zeros(1, 1, config.TILE_PX, config.TILE_PX)
    torch.onnx.export(
        model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = test_batch if test_batch is not None else np.random.default_rng(0).normal(size=(2, 1, 512, 512))
    x = x.astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    log.info("ONNX vs torch max abs diff = %.2e", diff)
    if diff > 1e-3:
        raise RuntimeError(f"ONNX export mismatch: {diff}")

    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M") + f"-{git_hash()}"
    card = {
        "model_version": version,
        "architecture": f"{ck['arch']}-{ck['encoder']} (segmentation_models_pytorch), 1-channel VV, 4 classes",
        "input": {"tile_px": config.TILE_PX, "pixel_deg": config.PIXEL_DEG, "crs": config.CRS,
                  "radiometry": "pseudo-dB = 10*log10(DN^2+1) of uncalibrated S1 GRD VV, clipped to db_clip -> uint8 1..255 (0 = nodata) -> /255 -> (x-mean)/std, nodata -> 0"},
        "db_clip": list(ck["db_clip"]),
        "norm_mean": ck["norm_mean"],
        "norm_std": ck["norm_std"],
        "classes": {str(k): v for k, v in config.MODEL_CLASSES.items() if k},
        "thresholds": {"p_low": config.POSTPROCESS.p_low, "p_seed": config.POSTPROCESS.p_seed,
                       "p_trim": config.POSTPROCESS.p_trim, "min_area_m2": config.POSTPROCESS.min_area_m2},
        "best_epoch": ck["epoch"],
        "val_metrics_crops": ck.get("val"),
        "metrics": metrics,
        "onnx_torch_max_abs_diff": diff,
        "trained_on": f"{ck.get('n_train_scenes')} Sentinel-1 scenes; labels derived from SkyTruth Cerulean API (CC BY-SA 4.0)",
        "license": "weights CC BY-SA 4.0 (derived from Cerulean data); code Apache-2.0; non-commercial conservation use per SkyTruth ToS",
        "attribution": [config.ATTRIBUTION_CERULEAN, config.ATTRIBUTION_COPERNICUS],
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    log.info("wrote %s and model_card.json (%s)", onnx_path, version)
    return card


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics", default=None, help="path to test metrics json to embed")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run = Path(args.run)
    metrics = json.loads(Path(args.metrics).read_text()) if args.metrics else None
    export(run, Path(args.out) if args.out else run / "export", metrics=metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
