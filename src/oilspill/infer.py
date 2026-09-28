"""Sliding-window inference over a whole scene.

Model-framework agnostic: a *predictor* is any callable mapping a float32 batch
``(N, 1, 512, 512)`` to logits ``(N, 4, 512, 512)``. :class:`OnnxPredictor`
is what the API/batch use; ``train.py`` wraps the torch model the same way, so
evaluation and serving share this exact code.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .scene import SceneRaster, to_uint8

log = logging.getLogger(__name__)

Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass
class ModelCard:
    model_version: str
    architecture: str
    db_clip: tuple[float, float]
    norm_mean: float
    norm_std: float
    classes: dict[str, str]
    thresholds: dict[str, float]
    metrics: dict | None = None
    trained_on: str | None = None
    raw: dict | None = None

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_version=d["model_version"], architecture=d["architecture"],
            db_clip=tuple(d["db_clip"]), norm_mean=float(d["norm_mean"]),
            norm_std=float(d["norm_std"]), classes=d["classes"],
            thresholds=d.get("thresholds", {}), metrics=d.get("metrics"),
            trained_on=d.get("trained_on"), raw=d,
        )


class OnnxPredictor:
    def __init__(self, onnx_path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), so, providers=providers)
        self.load_s = time.perf_counter() - t0
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]


def _ramp_weights(tile: int, overlap: int) -> np.ndarray:
    """2-D blending weights: 1 in the centre, linear ramp over the overlap."""
    w1 = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        w1[:overlap] = ramp
        w1[-overlap:] = ramp[::-1]
    return np.outer(w1, w1)


def _starts(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, stride))
    s.append(n - tile)
    return s


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def normalise(u8: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint8 grid raster -> model input. Nodata (0) maps to exactly 0.0.

    Shared by training (train.py) and inference so the two cannot diverge.
    """
    x = ((u8.astype(np.float32) / 255.0) - mean) / std
    x[u8 == config.NODATA] = 0.0
    return x


def predict_scene(
    scene: SceneRaster,
    predictor: Predictor,
    card: ModelCard,
    batch_size: int = 4,
    tta: bool = False,
    tile: int = config.TILE_PX,
    stride: int = config.INFER_STRIDE,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return per-class probabilities ``(4, H, W)`` float32 and timings."""
    t0 = time.perf_counter()
    u8 = to_uint8(scene.db, card.db_clip)
    valid = u8 != config.NODATA
    H, W = u8.shape
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    if pad_h or pad_w:
        u8 = np.pad(u8, ((0, pad_h), (0, pad_w)))
    Hp, Wp = u8.shape
    x = normalise(u8, card.norm_mean, card.norm_std)

    acc = np.zeros((config.N_CLASSES, Hp, Wp), dtype=np.float32)
    wsum = np.zeros((Hp, Wp), dtype=np.float32)
    wt = _ramp_weights(tile, tile - stride)

    coords = [(r, c) for r in _starts(Hp, tile, stride) for c in _starts(Wp, tile, stride)
              if (u8[r:r + tile, c:c + tile] != 0).any()]
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.stack([x[r:r + tile, c:c + tile] for r, c in chunk])[:, None]
        probs = softmax(predictor(batch))
        if tta:
            probs = probs + softmax(predictor(batch[..., ::-1].copy()))[..., ::-1]
            probs = probs + softmax(predictor(batch[..., ::-1, :].copy()))[..., ::-1, :]
            probs /= 3.0
        for (r, c), p in zip(chunk, probs, strict=True):
            acc[:, r:r + tile, c:c + tile] += p * wt
            wsum[r:r + tile, c:c + tile] += wt
    t_inf = time.perf_counter() - t1

    wsum[wsum == 0] = 1.0
    probs = (acc / wsum)[:, :H, :W]
    # [SRC] Cerulean zeroes non-background probability where the input is nodata.
    probs[1:, ~valid] = 0.0
    probs[0, ~valid] = 1.0
    return probs, {"preprocess_s": t_pre, "inference_s": t_inf, "n_tiles": len(coords)}
