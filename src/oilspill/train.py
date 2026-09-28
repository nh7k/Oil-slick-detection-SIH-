"""Phase 4/5: train the U-Net (ResNet34, 1-channel VV, 4 classes) on the built dataset.

Crops are sampled on the fly from whole-scene rasters:
  50% centred on a labelled oil pixel, 20% on a hard-negative (human-rejected
  look-alike), 30% anywhere valid. Validation uses a FIXED list of crops from
  the val scenes, so val numbers are comparable epoch to epoch.

Resumable: ``last.pt`` (model/optimizer/scaler/scheduler/epoch/best/RNG) is
written every epoch; ``--resume`` continues after a Colab disconnect.

Run (Colab):  python -m oilspill.train --data /content/data/dataset --out /content/drive/MyDrive/oilspill/run1
Overfit test: python -m oilspill.train ... --overfit 20 --epochs 60
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config
from .georef import db_to_uint8
from .infer import normalise
from .metrics import ConfMat

log = logging.getLogger(__name__)
T = config.TILE_PX


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SceneStore:
    """All scenes of one split held in RAM as uint8 (image) + uint8 (mask)."""

    def __init__(self, root: Path, ids: list[str], clip: tuple[float, float], mask_suffix: str = ""):
        self.images, self.masks, self.ids = [], [], []
        self.pos_idx, self.neg_pts = [], []
        for pid in ids:
            sp, mp = root / "scenes" / f"{pid}.tif", root / "masks" / f"{pid}{mask_suffix}.tif"
            if not (sp.exists() and mp.exists()):
                continue
            with rasterio.open(sp) as d:
                db = d.read(1).astype(np.float32) / 100.0
            with rasterio.open(mp) as d:
                m = d.read(1)
            img = db_to_uint8(db, clip)
            self.images.append(img)
            self.masks.append(m)
            self.ids.append(pid)
            pos = np.flatnonzero(np.isin(m, config.OIL_CLASSES))
            if pos.size > 20000:
                pos = np.random.default_rng(0).choice(pos, 20000, replace=False)
            self.pos_idx.append(pos)
            st = json.loads((root / "status_by_pc" / f"{pid}.json").read_text()) \
                if (root / "status_by_pc" / f"{pid}.json").exists() else {}
            self.neg_pts.append(st.get("neg_px", []))
        log.info("loaded %d scenes (%.1f GB)", len(self.ids),
                 sum(i.nbytes + m.nbytes for i, m in zip(self.images, self.masks, strict=True)) / 1e9)

    def crop(self, k: int, cy: int, cx: int) -> tuple[np.ndarray, np.ndarray]:
        img, m = self.images[k], self.masks[k]
        H, W = img.shape
        y0 = int(np.clip(cy - T // 2, 0, max(H - T, 0)))
        x0 = int(np.clip(cx - T // 2, 0, max(W - T, 0)))
        ci = img[y0:y0 + T, x0:x0 + T]
        cm = m[y0:y0 + T, x0:x0 + T]
        if ci.shape != (T, T):
            pi = np.zeros((T, T), np.uint8)
            pm = np.full((T, T), config.IGNORE_INDEX, np.uint8)
            pi[:ci.shape[0], :ci.shape[1]] = ci
            pm[:cm.shape[0], :cm.shape[1]] = cm
            ci, cm = pi, pm
        return ci, cm


def augment(img: np.ndarray, m: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randrange(4)
    img, m = np.rot90(img, k), np.rot90(m, k)
    if rng.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    img = np.ascontiguousarray(img)
    m = np.ascontiguousarray(m)
    valid = img > 0
    if rng.random() < 0.8:  # brightness/contrast jitter (+-10%) on valid pixels only
        f = img.astype(np.float32)
        a, b = rng.uniform(0.9, 1.1), rng.uniform(-12, 12)
        f = (f - 128) * a + 128 + b
        if rng.random() < 0.3:
            f += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 3, f.shape)
        img = np.where(valid, np.clip(np.round(f), 1, 255), 0).astype(np.uint8)
    return img, m


class CropDataset(Dataset):
    def __init__(self, store: SceneStore, n: int, mean: float, std: float, seed: int = 0,
                 fixed: list[tuple[int, int, int]] | None = None, aug: bool = True):
        self.s, self.n, self.mean, self.std = store, n, mean, std
        self.seed, self.fixed, self.aug = seed, fixed, aug
        self.epoch = 0
        w = np.array([max(len(p), 1) ** 0.5 for p in store.pos_idx], dtype=np.float64)
        self.scene_w = w / w.sum()

    def __len__(self) -> int:
        return len(self.fixed) if self.fixed is not None else self.n

    def _sample(self, rng: random.Random) -> tuple[int, int, int]:
        s = self.s
        k = int(np.random.default_rng(rng.randrange(1 << 30)).choice(len(s.ids), p=self.scene_w))
        H, W = s.images[k].shape
        u = rng.random()
        if u < 0.5 and len(s.pos_idx[k]):
            flat = int(s.pos_idx[k][rng.randrange(len(s.pos_idx[k]))])
            cy, cx = divmod(flat, W)
            cy += rng.randint(-T // 3, T // 3)
            cx += rng.randint(-T // 3, T // 3)
        elif u < 0.7 and s.neg_pts[k]:
            cy, cx = s.neg_pts[k][rng.randrange(len(s.neg_pts[k]))]
            cy += rng.randint(-T // 4, T // 4)
            cx += rng.randint(-T // 4, T // 4)
        else:
            for _ in range(20):
                cy, cx = rng.randrange(H), rng.randrange(W)
                if s.images[k][cy, cx] > 0:
                    break
        return k, cy, cx

    def __getitem__(self, i: int):
        if self.fixed is not None:
            k, cy, cx = self.fixed[i]
            rng = random.Random(i)
        else:
            rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
            k, cy, cx = self._sample(rng)
        img, m = self.s.crop(k, cy, cx)
        if self.aug:
            img, m = augment(img, m, rng)
        x = normalise(img, self.mean, self.std)[None]
        return torch.from_numpy(x), torch.from_numpy(m.astype(np.int64))


def fixed_val_crops(store: SceneStore, max_crops: int = 400, seed: int = 1) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for k, (img, m) in enumerate(zip(store.images, store.masks, strict=True)):
        H, W = img.shape
        for y in range(0, max(H - T, 0) + 1, T):
            for x in range(0, max(W - T, 0) + 1, T):
                ti, tm = img[y:y + T, x:x + T], m[y:y + T, x:x + T]
                if (ti > 0).mean() < 0.5:
                    continue
                c = (k, y + T // 2, x + T // 2)
                (pos if np.isin(tm, config.OIL_CLASSES).any() else neg).append(c)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), max_crops // 2)
    return pos[:n_pos] + neg[:max_crops - n_pos]


# ---------------------------------------------------------------------------
# Model / loss / metrics
# ---------------------------------------------------------------------------

def build_model(encoder: str = "resnet34", weights: str | None = "imagenet", arch: str = "Unet"):
    import segmentation_models_pytorch as smp

    cls = getattr(smp, arch)
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=config.N_CLASSES)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, classes=(1, 2, 3), eps: float = 1.0):
    valid = (target != config.IGNORE_INDEX)
    probs = logits.float().softmax(1)
    t = target.clone()
    t[~valid] = 0
    onehot = F.one_hot(t, config.N_CLASSES).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    losses = []
    for c in classes:
        p, g = probs[:, c] * v[:, 0], onehot[:, c] * v[:, 0]
        inter = (p * g).sum()
        losses.append(1 - (2 * inter + eps) / (p.sum() + g.sum() + eps))
    # binary oil dice as well: the product metric we care most about
    p_oil = probs[:, 1:].sum(1) * v[:, 0]
    g_oil = (onehot[:, 1:].sum(1)) * v[:, 0]
    losses.append(1 - (2 * (p_oil * g_oil).sum() + eps) / (p_oil.sum() + g_oil.sum() + eps))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_split(data: Path) -> tuple[pd.DataFrame, dict]:
    meta = json.loads((data / "meta" / "dataset_meta.json").read_text())
    splits = pd.read_csv(data / "meta" / "splits.csv")
    return splits, meta


def index_status(data: Path) -> None:
    """Map status/<cerulean id>.json -> status_by_pc/<pc id>.json with hard-negative pixel coords."""
    out = data / "status_by_pc"
    out.mkdir(exist_ok=True)
    for p in (data / "status").glob("*.json"):
        st = json.loads(p.read_text())
        if st.get("status") != "done":
            continue
        b = st["bounds"]
        # bounds are the scene window outer edges -> pixel coords
        negs = []
        for h in st.get("hints", []):
            if h["kind"] == "neg":
                negs.append([int((b[3] - h["lat"]) / config.PIXEL_DEG), int((h["lon"] - b[0]) / config.PIXEL_DEG)])
        prev = out / f"{st['pc_id']}.json"
        if prev.exists():
            negs += json.loads(prev.read_text()).get("neg_px", [])
        prev.write_text(json.dumps({"neg_px": negs}))


def compute_norm(store: SceneStore) -> tuple[float, float]:
    vals = np.concatenate([i[i > 0][:: max(1, (i > 0).sum() // 200000)] for i in store.images])
    f = vals.astype(np.float64) / 255.0
    return float(f.mean()), float(f.std())


def class_weights(store: SceneStore) -> torch.Tensor:
    counts = np.zeros(config.N_CLASSES, dtype=np.float64)
    for m in store.masks:
        bc = np.bincount(m[m != config.IGNORE_INDEX].ravel(), minlength=config.N_CLASSES)[:config.N_CLASSES]
        counts += bc
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    w = w / w[0]
    w = np.minimum(w, 20.0)
    log.info("class pixel counts %s -> CE weights %s", counts.astype(int).tolist(), np.round(w, 2).tolist())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def validate(model, loader, device, amp: bool) -> dict:
    model.eval()
    cm = ConfMat()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        cm.update(logits.argmax(1).cpu().numpy(), y.numpy())
    return cm.summary()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples-per-epoch", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit", type=int, default=0, help="overfit N fixed train crops (sanity gate)")
    ap.add_argument("--max-hours", type=float, default=10.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("NO GPU - training on CPU will be extremely slow")
    amp = device.type == "cuda"

    index_status(data)
    splits, meta = load_split(data)
    clip = tuple(meta["db_clip"])
    train_ids = splits[splits.split == "train"].pc_id.tolist()
    val_ids = splits[splits.split == "val"].pc_id.tolist()
    assert not set(train_ids) & set(val_ids)
    train_store = SceneStore(data, train_ids, clip)
    val_store = SceneStore(data, val_ids, clip)
    mean, std = compute_norm(train_store)
    cw = class_weights(train_store).to(device)

    if args.overfit:
        fixed = fixed_val_crops(train_store, args.overfit, seed=3)
        train_ds = CropDataset(train_store, 0, mean, std, fixed=fixed, aug=False)
        val_ds = CropDataset(train_store, 0, mean, std, fixed=fixed, aug=False)
    else:
        train_ds = CropDataset(train_store, args.samples_per_epoch, mean, std, seed=0)
        val_ds = CropDataset(val_store, 0, mean, std, fixed=fixed_val_crops(val_store), aug=False)
    log.info("train crops/epoch=%d  val crops=%d  mean=%.4f std=%.4f clip=%s",
             len(train_ds), len(val_ds), mean, std, clip)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=bool(args.overfit),
                          num_workers=args.workers, pin_memory=amp, drop_last=not args.overfit,
                          persistent_workers=False)  # workers must see the new epoch's sampling seed
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=amp)

    model = build_model(args.encoder, "imagenet", args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(train_dl) / args.accum)
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_cfg = {"arch": args.arch, "encoder": args.encoder, "db_clip": clip, "norm_mean": mean,
               "norm_std": std, "class_weights": cw.tolist(), "args": vars(args),
               "n_train_scenes": len(train_store.ids), "n_val_scenes": len(val_store.ids)}
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    start_epoch, best, bad = 0, -1.0, 0
    last_path, best_path = out / "last.pt", out / "best.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())  # map_location moved it to cuda
        log.info("resumed from epoch %d (best oil dice %.4f)", ck["epoch"], best)

    log_path = out / "log.csv"
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_ds.epoch = epoch
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                ce = F.cross_entropy(logits.float(), y, weight=cw, ignore_index=config.IGNORE_INDEX)
                loss = ce + dice_loss(logits, y)
            if not torch.isfinite(loss):
                log.error("non-finite loss at epoch %d step %d; skipping batch", epoch, i)
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run_loss += loss.item()
            n += 1
        train_s = time.time() - t0
        vm = validate(model, val_dl, device, amp)
        oil = vm["OIL_binary"]["f1_dice"]
        vram = torch.cuda.max_memory_allocated() / 1e9 if amp else 0.0
        row = {"epoch": epoch, "loss": run_loss / max(n, 1), "val_oil_dice": oil,
               "val_oil_iou": vm["OIL_binary"]["iou"],
               "val_infra_f1": vm["INFRA"]["f1_dice"], "val_natural_f1": vm["NATURAL"]["f1_dice"],
               "val_vessel_f1": vm["VESSEL"]["f1_dice"], "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(train_s, 1), "vram_gb": round(vram, 2)}
        new = not log_path.exists()
        with log_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        log.info("epoch %d loss %.4f val_oil_dice %.4f (infra %.3f nat %.3f vessel %.3f) %.0fs vram %.1fGB",
                 epoch, row["loss"], oil, row["val_infra_f1"], row["val_natural_f1"],
                 row["val_vessel_f1"], train_s, vram)

        if oil > best:
            best, bad = oil, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": vm, **run_cfg}, best_path)
            (out / "best_val_metrics.json").write_text(json.dumps({"epoch": epoch, **vm}, indent=2))
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad": bad,
                    "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, last_path)
        if bad >= args.patience and not args.overfit:
            log.info("early stopping at epoch %d (best %.4f)", epoch, best)
            break
        if (time.time() - t_start) / 3600 > args.max_hours:
            log.info("time budget reached; stop (resume with --resume)")
            break
    log.info("done. best val oil dice = %.4f  -> %s", best, best_path)
    if args.overfit:
        ok = best > 0.9
        log.info("OVERFIT GATE %s (need > 0.9)", "PASSED" if ok else "FAILED")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
