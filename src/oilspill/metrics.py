"""Segmentation metrics (numpy only, shared by training and evaluation)."""

from __future__ import annotations

import numpy as np

from . import config


class ConfMat:
    def __init__(self, n: int = config.N_CLASSES):
        self.m = np.zeros((n, n), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        v = target != config.IGNORE_INDEX
        self.m += np.bincount(target[v].astype(np.int64) * self.m.shape[0] + pred[v].astype(np.int64),
                              minlength=self.m.size).reshape(self.m.shape)

    def summary(self) -> dict:
        m = self.m.astype(np.float64)
        out = {}
        for c in range(1, m.shape[0]):
            tp, fp, fn = m[c, c], m[:, c].sum() - m[c, c], m[c, :].sum() - m[c, c]
            out[config.MODEL_CLASSES[c]] = _prf(tp, fp, fn)
        tp = m[1:, 1:].sum()
        fp = m[0, 1:].sum()
        fn = m[1:, 0].sum()
        out["OIL_binary"] = _prf(tp, fp, fn)
        out["confusion_matrix"] = self.m.tolist()
        return out


def _prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": p, "recall": r, "f1_dice": f, "iou": iou, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
