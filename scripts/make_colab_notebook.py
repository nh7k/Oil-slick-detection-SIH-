"""Generate notebooks/colab_train.ipynb with the oilspill package embedded.

Why embedded: the Colab runtime is a different machine from this laptop and we
are not using GitHub, so cell 2 writes the exact package source (from src/) onto
the runtime. Re-run this script after any code change, then re-open the notebook.

    python scripts/make_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "oilspill"
OUT = ROOT / "notebooks" / "colab_train.ipynb"

SHIPPED = [
    "__init__.py", "config.py", "crsutil.py", "georef.py", "s1_source.py", "cerulean_api.py",
    "labels.py", "landmask.py", "scene.py", "dataset_build.py", "infer.py", "postprocess.py",
    "metrics.py", "train.py", "export.py", "evaluate.py",
]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n")}


def main() -> None:
    files = {f"src/oilspill/{n}": (PKG / n).read_text(encoding="utf-8") for n in SHIPPED}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    files["README.md"] = "oilspill (Colab copy)\n"

    cells = [
        md("""
# OilWatch — train the oil-slick model on Colab (Run All)

**What this does, end to end, on real data:**
1. mounts Google Drive (everything important is saved there → safe against disconnects)
2. installs the `oilspill` package (source embedded below — no GitHub needed)
3. checks the GPU
4. builds the dataset: Cerulean public slick polygons (labels) + Sentinel-1 GRD VV from Microsoft Planetary Computer, warped onto Cerulean's grid
5. overfit sanity gate → full training (U-Net ResNet34, AMP, resumable)
6. ONNX export + honest evaluation (human-reviewed test scenes only)
7. writes `MyDrive/oilspill/oilspill_model.zip` → download it and put `model.onnx` + `model_card.json` into the project's `models/` folder

**Runtime → Change runtime type → T4 GPU** first. Then **Run All**.
If Colab disconnects: reconnect and Run All again — every step resumes from Drive.

Data attribution: SkyTruth Cerulean (cerulean.skytruth.org), CC BY-SA 4.0 (non-commercial conservation use per SkyTruth ToS). Contains modified Copernicus Sentinel data 2023–2026.
"""),
        code("""
# 1) Google Drive (persistent storage)
import os, pathlib
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = pathlib.Path('/content/drive/MyDrive/oilspill')
except Exception as e:
    print('!! Drive mount failed:', e)
    print('!! Falling back to /content/oilspill_out (NOT persistent; download the zip at the end)')
    DRIVE = pathlib.Path('/content/oilspill_out')
DRIVE.mkdir(parents=True, exist_ok=True)
RUN = DRIVE / 'run1'
DATA = pathlib.Path('/content/data/dataset')
os.environ['OILSPILL_DATA_DIR'] = '/content/data'
print('persistent dir:', DRIVE)
"""),
        code("FILES = " + json.dumps(files) + """

# 2) write the package onto the runtime and install it
import pathlib, subprocess, sys
PKG_ROOT = pathlib.Path('/content/oilspill_pkg')
for rel, text in FILES.items():
    p = PKG_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'segmentation-models-pytorch==0.5.0', 'timm', 'rasterio', 'pystac-client',
                'planetary-computer', 'shapely>=2', 'pyshp', 'tenacity', 'python-dotenv',
                'onnx', 'onnxruntime', 'httpx', 'pyarrow'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', str(PKG_ROOT)], check=True)
import importlib, site; importlib.invalidate_caches()
sys.path.insert(0, str(PKG_ROOT / 'src'))
import oilspill, oilspill.config as C

def run_live(args):
    # run a step as a subprocess and stream its log into this notebook live
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, cwd='/content')
    for line in p.stdout:
        if 'HTTP Request:' in line:
            continue  # too chatty
        print(line, end='', flush=True)
    return p.wait()

print('oilspill installed; grid pixel =', C.PIXEL_DEG, 'deg')
"""),
        code("""
# 3) GPU check
import torch, subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout[:800])
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU! Runtime -> Change runtime type -> T4 GPU, then Run All again.'
print('GPU:', torch.cuda.get_device_name(0))
"""),
        code("""
# 4) Build the dataset (real labels + real imagery). ~20-40 min.
# Built DIRECTLY on Google Drive, one status file per scene, so a Colab disconnect
# loses nothing: re-run and it continues from the last finished scene.
import shutil, subprocess, sys, json
DATA_DRIVE = DRIVE / 'dataset'
OLD_TAR = DRIVE / 'dataset.tar'   # written by the first version of this notebook
if not (DATA / 'meta' / 'dataset_meta.json').exists() and OLD_TAR.exists():
    import tarfile
    print('restoring dataset from', OLD_TAR, '...')
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OLD_TAR) as t: t.extractall(DATA.parent)
if not (DATA / 'meta' / 'dataset_meta.json').exists() and not (DATA_DRIVE / 'meta' / 'dataset_meta.json').exists():
    r = run_live([sys.executable, '-m', 'oilspill.dataset_build', '--root', str(DATA_DRIVE),
                        '--scenes-per-aoi', '40', '--workers', '8'])
    assert r == 0, 'dataset build failed - scroll up for the error'
# copy to the fast local disk for training (Drive reads are slow)
if not (DATA / 'meta' / 'dataset_meta.json').exists():
    print('copying dataset Drive -> /content (fast local disk) ...')
    shutil.copytree(DATA_DRIVE, DATA, dirs_exist_ok=True, ignore=shutil.ignore_patterns('cache'))
meta = json.loads((DATA / 'meta' / 'dataset_meta.json').read_text())
print(json.dumps({k: meta[k] for k in ['n_done','n_failed','db_clip','split_counts','t1_scenes_by_split','pos_px_total']}, indent=1))
"""),
        code("""
# 5) Look at the data: SAR + label overlay for random training scenes
import numpy as np, pandas as pd, rasterio, matplotlib.pyplot as plt
sp = pd.read_csv(DATA / 'meta' / 'splits.csv')
cands = sp[(sp.split == 'train')].sample(min(4, (sp.split=='train').sum()), random_state=0)
fig, axes = plt.subplots(len(cands), 2, figsize=(12, 5 * len(cands)))
axes = np.atleast_2d(axes)
for ax, pid in zip(axes, cands.pc_id):
    with rasterio.open(DATA / 'scenes' / f'{pid}.tif') as d: db = d.read(1) / 100.0
    with rasterio.open(DATA / 'masks' / f'{pid}.tif') as d: m = d.read(1)
    lo, hi = meta['db_clip']
    ax[0].imshow(np.clip((db - lo) / (hi - lo), 0, 1), cmap='gray'); ax[0].set_title(pid[:40], fontsize=8)
    ax[1].imshow(np.clip((db - lo) / (hi - lo), 0, 1), cmap='gray')
    ov = np.ma.masked_where(~np.isin(m, [1, 2, 3]), m)
    ax[1].imshow(ov, cmap='autumn', alpha=0.8, vmin=1, vmax=3); ax[1].set_title('labels (1 infra, 2 natural, 3 vessel)')
    for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
"""),
        code("""
# 6) Sanity gate: the model must be able to overfit 16 crops (catches data/label/loss bugs)
import subprocess, sys
r = run_live([sys.executable, '-m', 'oilspill.train', '--data', str(DATA), '--out', str(DRIVE / 'overfit'),
                    '--overfit', '16', '--epochs', '150', '--batch-size', '8', '--workers', '0', '--lr', '1e-3'])
print('overfit gate exit code', r, '(0 = passed, 2 = failed)')
assert r == 0, 'Overfit gate failed - paste the log above to Claude'
"""),
        code("""
# 7) Full training (resumable: just re-run this cell after a disconnect)
import subprocess, sys
r = run_live([sys.executable, '-m', 'oilspill.train', '--data', str(DATA), '--out', str(RUN),
                    '--epochs', '40', '--samples-per-epoch', '2400', '--batch-size', '8',
                    '--workers', '2', '--resume', '--max-hours', '9'])
assert r == 0
"""),
        code("""
# 8) Training curves
import pandas as pd, matplotlib.pyplot as plt
log = pd.read_csv(RUN / 'log.csv')
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
log.plot(x='epoch', y='loss', ax=ax[0], title='train loss')
log.plot(x='epoch', y=['val_oil_dice', 'val_infra_f1', 'val_natural_f1', 'val_vessel_f1'], ax=ax[1], title='val (fixed crops)')
plt.show(); print(log.tail(5).to_string())
"""),
        code("""
# 9) Export ONNX (+ model_card.json) and verify ONNX == torch
import subprocess, sys
r = run_live([sys.executable, '-m', 'oilspill.export', '--run', str(RUN), '--out', str(RUN / 'export')])
assert r == 0
"""),
        code("""
# 10) Honest evaluation: thresholds tuned on VAL, TEST scored once on human-reviewed slicks
import subprocess, sys, json
r = run_live([sys.executable, '-m', 'oilspill.evaluate', '--data', str(DATA), '--model', str(RUN / 'export'),
                    '--torch-ckpt', str(RUN / 'best.pt')])
assert r == 0
rep = json.loads((RUN / 'export' / 'test_metrics.json').read_text())
t = rep['test_T1_human_reviewed']
print('tuned thresholds:', rep['tuned_thresholds'])
print('TEST (human-reviewed): oil pixel', {k: round(v, 3) for k, v in t['OIL_binary'].items() if k in ('precision','recall','f1_dice','iou')})
print('TEST (human-reviewed): instance', {k: round(v, 3) for k, v in t['instance'].items() if k in ('precision','recall','f1_dice')})
print('Cerulean published pixel F1 0.532 is on THEIR private test set - not directly comparable.')
"""),
        code("""
# 11) Visual check: full-scene prediction on a test scene with our polygons vs Cerulean labels
import numpy as np, matplotlib.pyplot as plt, rasterio, pandas as pd
from shapely.geometry import shape
from oilspill.infer import ModelCard, OnnxPredictor, predict_scene
from oilspill.postprocess import extract_slicks
from oilspill.scene import load_scene
from oilspill.evaluate import TorchPredictor
card = ModelCard.load(RUN / 'export' / 'model_card.json')
pred = TorchPredictor(RUN / 'best.pt')
sp = pd.read_csv(DATA / 'meta' / 'splits.csv')
pid = sp[(sp.split == 'test') & (sp.t1_pos_px > 0)].pc_id.iloc[0]
scene = load_scene(DATA / 'scenes' / f'{pid}.tif')
probs, timing = predict_scene(scene, pred, card, batch_size=8)
feats = extract_slicks(probs, scene.window, scene.valid, overrides=card.thresholds)
with rasterio.open(DATA / 'masks' / f'{pid}.tif') as d: m = d.read(1)
lo, hi = card.db_clip
b = scene.bounds; ext = [b[0], b[2], b[1], b[3]]
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
for a in ax: a.imshow(np.clip((scene.db - lo) / (hi - lo), 0, 1), cmap='gray', extent=ext)
ax[0].imshow(np.ma.masked_where(~np.isin(m, [1,2,3]), m), cmap='autumn', alpha=.7, extent=ext); ax[0].set_title('Cerulean labels')
ax[1].imshow(np.ma.masked_less(probs[1:].sum(0), 0.3), cmap='magma', alpha=.6, extent=ext, vmin=0, vmax=1)
for f in feats:
    g = shape(f['geometry'])
    for p in g.geoms: ax[1].plot(*p.exterior.xy, color='cyan', lw=1)
ax[1].set_title(f'our model: {len(feats)} slicks'); plt.show()
print(timing); [print(f['properties']['cls_name'], round(f['properties']['confidence'], 3), round(f['properties']['area_km2'], 2), 'km2') for f in feats[:10]]
"""),
        code("""
# 12) Package the model for the laptop: MyDrive/oilspill/oilspill_model.zip
import zipfile, json
z = DRIVE / 'oilspill_model.zip'
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
    for n in ['model.onnx', 'model_card.json', 'test_metrics.json']:
        zf.write(RUN / 'export' / n, n)
    zf.write(RUN / 'log.csv', 'train_log.csv')
print('WROTE', z, round(z.stat().st_size / 1e6, 1), 'MB')
print(json.dumps(json.loads((RUN / 'export' / 'model_card.json').read_text())['metrics'], indent=1)[:1500])
print()
print('NEXT: open drive.google.com -> My Drive -> oilspill -> download oilspill_model.zip,')
print('      then unzip it into the project folder  sih\\\\models\\\\')
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                        "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(files)} embedded files)")


if __name__ == "__main__":
    main()
