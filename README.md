# Ocean Police — Global Oil Slick Detection & Attribution

A 4-class U-Net on Sentinel-1 SAR imagery for detecting and attributing oil slicks
worldwide: infrastructure discharge, natural seeps, and vessel discharge. Includes
a FastAPI service, a MapLibre map interface, and AIS-based ship attribution.

> **Status: Phase 1 of 15 complete.** Georeferencing is verified end to end on
> real data. There is no trained model yet, and therefore no measured detection
> metrics. Nothing in this repository reports a number that was not measured —
> see [PROGRESS.md](PROGRESS.md) for exactly what has and has not been run.

**Non-commercial.** The labels are CC BY-SA 4.0 and SkyTruth's terms restrict use
to environmental conservation. See [DATA_LICENSES.md](DATA_LICENSES.md).

---

## What it does

The detection pipeline, end to end:

- **Input.** Sentinel-1 GRD, VV only, raw digital numbers — no calibration, no
  speckle filtering, no terrain correction.
- **Grid.** morecantile `WorldCRS84Quad`, zoom 9, tile scale 2 → an EPSG:4326
  pixel of `180 / 2**9 / 512 = 0.0006866455078125°` (≈76 m at the equator).
- **Model.** U-Net / ResNet34, 1 input channel, 4 classes:
  `0 BACKGROUND, 1 INFRA, 2 NATURAL, 3 VESSEL`.
- **Post-processing.** Hysteresis (islands above `1e-4`, kept only if some pixel
  ≥ `0.9`, trimmed at `0.5`), polygonisation, a scene-edge rule, polygon NMS at
  IoU 0.2, then per-slick confidence and class.

Every constant lives in [`src/oilspill/config.py`](src/oilspill/config.py) and
nowhere else.

---

## Verified so far

**Georeferencing works.** Planetary Computer's `sentinel-1-grd` COGs carry no
affine geotransform — only ~210 GCPs — and the STAC `proj:transform` is a bbox
approximation that must not be used. Warping through the GCPs onto the Cerulean
grid was correct on the first attempt, with no WarpedVRT, `gdalwarp -tps` or
Earth Engine fallback needed.

Three human-reviewed slicks from three regions and both orbit directions:

| Slick | Region | Orbit | dB contrast | Land offset | In footprint |
| --- | --- | --- | --- | --- | --- |
| 3880640 | Congo basin | descending | 3.92 dB | 0.0 px | 1.0000 |
| 3098513 | Gulf of Mexico | ascending | 3.39 dB | n/a | 0.9987 |
| 4823849 | Bay of Bengal | ascending | 3.86 dB | n/a | 1.0000 |

Overlays are in [`reports/phase1/`](reports/phase1/).

**Two API facts the docs get wrong**, both measured (see
[DECISIONS.md](DECISIONS.md)):

1. `hitl_cls IS NOT NULL` is **silently ignored** by the Cerulean tipg
   deployment — it returns the same `numberMatched` as `IS NULL`. Use
   `hitl_cls > 0`. The whole human-reviewed universe is **7,185 slicks**.
2. A Planetary Computer item id is the Cerulean `s1_scene_id` **minus its
   trailing 4-hex token**. Exact-id lookup misses every time.

---

## Setup

Python ≥ 3.12.

```bash
git clone <this repo> && cd oilspill
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # then point OILSPILL_DATA_DIR at a roomy disk
```

`OILSPILL_DATA_DIR` holds tiles and caches and will grow to many GB — keep it
out of any cloud-synced folder.

No GDAL system install is needed; rasterio's wheel bundles GDAL 3.12 / PROJ 9.8,
and this package deliberately does **not** depend on pyproj, geopandas or
pyogrio (DECISIONS.md D0.6), so there is no OGR driver stack to fight.

### Reproduce the Phase 1 spike

```bash
python scripts/spike_phase1.py --n-regions 3
pytest -q
```

Writes overlays, a GeoTIFF with the real transform, and per-criterion
measurements to `reports/phase1/`.

---

## Layout

```
src/oilspill/
  config.py        every grid constant, class map, threshold and path
  georef.py        the Cerulean grid, GCP scaling, warping, assert_on_grid
  crsutil.py       UTM estimation and metric geometry, via rasterio's PROJ
  cerulean_api.py  cached, rate-limited OGC API Features client
  s1_source.py     Planetary Computer STAC, signing, scene-id matching
  labels.py        slick polygons -> label masks
  landmask.py      Natural Earth 10 m land, read without OGR
scripts/           spike_phase1.py, build_dataset.py, run_batch.py, fetch_model.py
backend/           FastAPI + SQLAlchemy 2 + Alembic
frontend/          Vite + React + TypeScript + MapLibre
tests/             pytest
reports/           measured outputs; nothing here is hand-written
```

---

## Metrics policy

Published pixel F1 for comparable U-Net architectures is around **0.532**. Those
were measured on private test sets and are **not comparable** to anything measured
here. Any table in this repository reports "ours (our test set)" and "published
(external test set, not comparable)" as separate rows, and our column stays empty
until Phase 6 has actually run.

---

## Limitations

- Trained on external reference data, so it inherits their biases and cannot
  exceed them; the Tier-1 (human-reviewed) supervision ceiling is 7,185 slicks.
- No calibration, so brightness varies with incidence angle and platform;
  `DB_CLIP` is frozen from training-set percentiles rather than physics.
- Look-alikes (low wind, biogenic films, rain cells) are the dominant error
  mode of every SAR oil detector, this one included.
- Not an operational or legal instrument. It is a research reproduction.

---

## Licences and attribution

Code **Apache-2.0** ([LICENSE](LICENSE)). Dataset, weights and detections
**CC BY-SA 4.0**, non-commercial in practice — full detail in
[DATA_LICENSES.md](DATA_LICENSES.md).

> SkyTruth (skytruth.org). 2026. CC BY-SA 4.0. Accessed 2026-09-27.

> Contains modified Copernicus Sentinel data 2026.

Basemap © OpenStreetMap contributors (ODbL). Coastlines from Natural Earth
(public domain).

**Not used:** External training data, weights and UI source are not public.
No attempt was made to obtain them; this is trained from scratch.
