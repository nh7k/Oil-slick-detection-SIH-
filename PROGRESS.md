# PROGRESS

Running log so a future session can resume without re-deriving anything.
Newest phase last.

**Status: Phase 1 complete and passing. Phase 2 not started** (blocked on your
sign-off of the overlay images, per the spec).

---

## Phase 0 — environment verification ✅

**Date.** 2026-09-27

### What the workspace actually contained

All three pre-existing files were **0 bytes**:

| File | Size |
| --- | --- |
| `OCEAN_POLICE_BLUEPRINT.md` | 0 |
| `IMPLEMENTATION_PROMPT.md` | 0 |
| `oil_slick_detection.ipynb` | 0 |

So there was no blueprint to read and nothing in the notebook to reuse
(DECISIONS.md D0.1). The spec of record is the prompt text supplied through the
editor selection.

### Local toolchain

| Tool | Version | Note |
| --- | --- | --- |
| Python | 3.13.15 | not 3.12; only interpreter installed (D0.3) |
| git | 2.55.0.windows.3 | repo initialised here |
| Node | 24.19.0 | ≥ 20 ✅ |
| npm | 11.17.0 | |
| Docker | **not installed** | backend image cannot be verified locally (D0.4) |
| Free disk (C:) | 310 GB | |

### Colab

`google.colab-0.9.6` and `ms-toolsai.jupyter-2025.9.1` are installed. **I cannot
execute notebook cells** — the only notebook tool available to me is
`NotebookEdit` (authoring). So the division of labour is: I write cells, you run
them and paste outputs back (D0.2). Heavy logic stays in `src/oilspill`.

*Not yet measured (needs you to run a cell): `nvidia-smi`, Drive mount, Colab
Python version, package install on Colab.*

### Environment built

- venv at `C:\Users\itanm\.venvs\oilspill` — deliberately **outside** OneDrive
  (D0.5); bulky artifacts go to `OILSPILL_DATA_DIR=C:/oilspill-data`.
- Repo skeleton, `pyproject.toml`, `.gitignore`, `.env.example`, `.env`.
- `pip install -e .` succeeds; package imports.

**rasterio 1.5.1 installs from a wheel on Windows/Python 3.13 with no GDAL
system install**, so the "use Docker if rasterio fails on Windows" fallback is
not needed.

### The one real environment obstacle

**Windows Smart App Control is enforcing** (`VerifiedAndReputablePolicyState=1`)
and blocks the native extensions of **pyproj** and **pyogrio**:

```
ImportError: DLL load failed while importing _io_direct:
An Application Control policy has blocked this file.
```

Everything else imports, including **onnxruntime**, rasterio, shapely, scipy.
Rather than disable Smart App Control (irreversible, and not mine to change),
both packages were **removed as dependencies** (D0.6):

- `oilspill/crsutil.py` does all CRS work through rasterio's bundled
  GDAL 3.12.4 / PROJ 9.8.1;
- `oilspill/landmask.py` reads Natural Earth shapefiles with pure-Python pyshp.

### Commands run

```powershell
python -m venv C:\Users\itanm\.venvs\oilspill
pip install numpy rasterio "shapely>=2.0" pystac-client planetary-computer httpx `
            tenacity Pillow matplotlib tqdm pandas pyarrow python-dotenv scipy pyshp
pip install onnx onnxruntime fastapi "uvicorn[standard]" python-multipart `
            "sqlalchemy>=2" geoalchemy2 alembic pytest pytest-cov ruff
pip install -e .
pip freeze > requirements.lock.txt   # 101 packages
```

### Open issues

- Colab side entirely unverified (needs you to run cells).
- Docker unverified — no Docker on this machine.

---

## Phase 1 — real data spike ✅ **PASSED**

**Date.** 2026-09-27 · Script `scripts/spike_phase1.py` · Report
`reports/phase1/spike_results.json`

### Cerulean API — verified, and the spec's filter idiom was wrong

`/conformance` advertises `ogcapi-features-3/1.0/conf/filter`, so CQL2 works.
But **`hitl_cls IS NOT NULL` is silently ignored**:

| CQL2 filter | `numberMatched` | reviewed rows returned? |
| --- | --- | --- |
| *(none)* | 1,334,040 | no |
| `hitl_cls IS NOT NULL` | 1,326,855 | no |
| `hitl_cls IS NULL` | **1,326,855** (identical) | no |
| `hitl_cls > 0` | **7,185** | **yes** |
| `hitl_cls IN (3,4,5)` | 2,473 | yes |

Tier-1 predicate is therefore `hitl_cls > 0` (D1.1). **The entire human-reviewed
universe is 7,185 slicks** — that is the ceiling on our test set.

Also confirmed: `area` is in m² (D1.3); `s1_scene_id` is the full SAFE product
name.

### Scene matching — the UNVERIFIED id question is now answered

Exact-id lookup **missed on all three scenes**. The PC item id is the Cerulean id
**minus its trailing 4-hex token** (D1.2):

```
Cerulean  S1A_IW_GRDH_1SDV_20230917T000951_20230917T001016_050360_061036_7DC5
PC        S1A_IW_GRDH_1SDV_20230917T000951_20230917T001016_050360_061036
```

`find_item` now tries exact → id-minus-suffix → ±2 min prefix search. Verified
resolving in a single GET via `id_minus_suffix`.

### Results — 3 regions, 3/3 pass

| Slick | Region | Orbit | Contrast | Land | Footprint | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 3880640 | Congo basin (11.6°E, −5.9°S) | descending | **3.92 dB** | **0.0 px** | 1.0000 | PASS |
| 3098513 | Gulf of Mexico | ascending | **3.39 dB** | n/a | 0.9987 | PASS |
| 4823849 | Bay of Bengal (88.9°E, 19.8°N) | ascending | **3.86 dB** | n/a | 1.0000 | PASS |

- **(a) visual** — PASS. Overlays inspected: the polygon traces the dark feature
  pixel-accurately in all three (a sinuous natural seep, a Gulf of Mexico seep
  field, and a long thin vessel trail).
- **(b) contrast ≥ 2–3 dB** — PASS, 3.39–3.92 dB.
- **(c) land within ~2 px** — PASS on the one scene with coastline in view
  (0.0 px); n/a on the other two. Measured as SAR-brightness-edge to
  Natural-Earth-land distance (D1.5).
- **(d) STAC footprint** — PASS, as a containment *fraction* ≥ 0.99 rather than
  strict containment, because the STAC footprint is a coarse swath
  simplification (D1.6).
- **(e) both passes** — PASS: ascending and descending both present.

### Measured throughput (decides Phase 2 scale)

| Slick | Native | Decimated | GCPs | Read | Warp |
| --- | --- | --- | --- | --- | --- |
| 3880640 | 16826×25487 | 2103×3185 | 210 | 9.3 s | 0.24 s |
| 3098513 | 16734×25887 | 2091×3235 | 210 | 2.0 s | 0.22 s |
| 4823849 | 19444×25692 | 2430×3211 | 231 | 3.4 s | 0.35 s |

All assets carry overviews `[2, 4, 8, 16, 32, 64]`, so the decimated read is
served from the ×8 overview instead of a full download — a ~430 Mpx scene reads
in seconds, ~13 MB decoded. **No georeferencing fallback was needed**: no
WarpedVRT, no `gdalwarp -tps`, no Earth Engine. Plain
`rasterio.warp.reproject` with scaled GCPs was correct first time.

**Phase 2 verdict.** At ~5 s/scene of I/O, 300–600 scenes is comfortable (D1.7).

### Tests

`tests/test_georef.py` — **31 passed**. Covers the grid constants, global grid
consistency across independently built windows, `assert_on_grid` rejecting
half-pixel/rotation/CRS errors, the nodata-preserving uint8 mapping, UTM zone
selection including antimeridian wrapping, and metric area/perimeter/
Polsby-Popper against a known square.

### Commands run

```powershell
python scripts\spike_phase1.py --n-regions 3
python -m pytest tests/test_georef.py -q
```

### Open issues

- `config.DB_CLIP` is still `None` by design — it is measured in Phase 2 from
  training-tile percentiles and then frozen. Any code path needing it fails
  loudly rather than guessing.
- Criterion (c) only had a coastline in one of three scenes; worth re-checking
  on a coastal AOI during Phase 2.
- Colab has not been touched yet.

---

## Phase 2 — dataset construction ⏸ NOT STARTED

Blocked deliberately: the spec says to stop and show the Phase 1 overlays before
starting Phase 2.

---

## Phases 2–12 — code complete, waiting on Colab training (2026-09-27)

**Built and verified locally**

| Piece | File(s) | Verified how |
| --- | --- | --- |
| Dataset builder (Cerulean labels + PC imagery, resumable, scene/2°-cell/temporal splits, frozen dB clip) | `src/oilspill/dataset_build.py`, `scene.py` | smoke run on 6 real scenes: 5 built (~5 s each), 1 had no PC item (logged as failed) |
| Training (U-Net ResNet34, 1-ch VV, CE+Dice, AMP, resumable, overfit gate) | `src/oilspill/train.py` | numpy data path exercised on the smoke scenes; torch loop runs on Colab only (no torch on this laptop) |
| Export ONNX + model_card, ONNX==torch check | `src/oilspill/export.py` | runs on Colab |
| Evaluation (val-tuned thresholds, T1-only test, pixel + instance) | `src/oilspill/evaluate.py`, `metrics.py` | imports OK; runs on Colab |
| Inference (sliding window 512/384, ramp blending) + Cerulean-style polygons | `infer.py`, `postprocess.py` | `tests/test_pipeline_api.py` on a real Phase-1 scene |
| API (FastAPI + SQLite) | `backend/app/{main,pipeline,database}.py` | same tests: health, 415/422/404, real GeoTIFF upload -> MultiPolygon in lon/lat, DB idempotency + filters |
| Frontend (React + Vite + MapLibre) | `frontend/` | `npm run build`, `check:nofixtures` pass; runs against the local API |
| Batch job | `scripts/run_batch.py` | exits 2 with a clear message while no model exists |
| Colab notebook (Run All; package embedded, no GitHub) | `notebooks/colab_train.ipynb` via `scripts/make_colab_notebook.py` | all cells compile |

**Blocking step:** run `notebooks/colab_train.ipynb` on a Colab T4, download
`MyDrive/oilspill/oilspill_model.zip`, unzip into `models/`.

---

## Model trained + first real detections (2026-09-27)

**Model** `v20260927-0325-nogit`: U-Net ResNet34, 1-ch VV, trained on Colab T4 (166 train / 30 val scenes, 40 epochs, best val crop oil-Dice 0.6055 at epoch ≤35).

**Measured test metrics** (52 held-out scenes, thresholds tuned on val: p_seed 0.6, p_trim 0.5), from `models/test_metrics.json`:

| Test set | Oil pixel P | R | F1/Dice | IoU | Instance F1 |
| --- | --- | --- | --- | --- | --- |
| T1 = human-reviewed Cerulean slicks only | 0.215 | 0.805 | 0.339 | 0.204 | 0.249 |
| T1+T2 = agreement with Cerulean's own model | 0.459 | 0.686 | 0.550 | 0.380 | 0.277 |

Per-class F1 on T1 is weak (INFRA 0.13, NATURAL 0.00, VESSEL 0.14): the model finds oil
far better than it tells infra/natural/vessel apart. Cerulean's published pixel F1 0.532 is on
their private test set and NOT comparable.

**Post-evaluation change (D12.1):** `group_px = 6` now groups trimmed fragments within ~450 m
into one slick (one MultiPolygon per slick, like Cerulean). The metrics above were measured
BEFORE this change; instance metrics must be re-measured to be quoted for the new setting.

**First real run** (`scripts/run_batch.py`, laptop CPU, ONNX): 10 Sentinel-1 scenes over the
Arabian Sea/Kerala AOI, ~40–65 s per scene end to end. Cerulean slick 5029964
(2026-06-21, VESSEL) is detected by our model as VESSEL (detections 40/41/45/46, median
confidence 0.88–0.98), covering 63% of Cerulean's slick pixels; verified visually on the map
with the SAR quicklook overlay.

---

## India-wide coverage, live updates, ship attribution (2026-09-27)

- `config.INDIA_AOIS`: Arabian Sea (66–78E, 5–24.5N) + Bay of Bengal/Andaman (78–95E, 4–23N).
  Scenes < 5% sea (Natural Earth land) are skipped. ~9–10 scenes/day over Indian seas (measured: 112 in 12 days).
- Backfill of 2026-08-27..09-27 running (`C:/oilspill-data/backfill_1month.log`, 281 candidate scenes).
- Live: Windows Task Scheduler task **"Ocean Police live update"** every 4 h -> `scripts/live_update.cmd`
  -> `run_batch.py` (last 14 days, skips scenes already done; single-instance lock
  `C:/oilspill-data/run_batch.lock`). Log: `C:/oilspill-data/live_update.log`.
  Remove with: `schtasks /Delete /TN "Ocean Police live update" /F`.
- Ship attribution: `GET /api/detections/{id}/sources` finds the overlapping SkyTruth Cerulean
  slick (same pass ±1 day) and returns Cerulean's AIS-based candidate sources (MMSI, score,
  rank). Shown in the detail panel as "Possible source (ships nearby)". Verified: our detection
  #40 -> Cerulean slick 5029964 -> MMSI 538009014 rank 1. No Cerulean overlap = honest empty state.
