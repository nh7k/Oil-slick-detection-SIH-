# DECISIONS

Every deviation from the implementation spec, with the evidence that forced it.
Newest first within each phase.

---

## Phase 0 — environment

### D0.1 The workspace files were empty

**Evidence.** On 2026-09-27, `OCEAN_POLICE_BLUEPRINT.md`, `IMPLEMENTATION_PROMPT.md`
and `oil_slick_detection.ipynb` were all **0 bytes** on disk
(`Get-ChildItem … | Select Length` → 0 for all three; `Get-Content -Raw` returned
0 characters). They carry the `ReparsePoint` attribute (OneDrive), but hydration
attempts returned no content — they are genuinely empty, not dehydrated.

**Consequence.**

- There is no blueprint to read, so references to "blueprint section 14"
  (the DB schema) cannot be followed. The schema is designed here instead and
  documented in this file (see D9.1 when Phase 9 lands).
- The notebook contains nothing to reuse; it is authored from scratch as a thin
  Colab driver.
- The spec of record is the implementation prompt text supplied through the
  editor selection.

### D0.2 I cannot execute Colab cells; you must run them

**Evidence.** The `google.colab-0.9.6` VS Code extension **is** installed
(alongside `ms-toolsai.jupyter-2025.9.1`). But the tools available to me in this
session include only `NotebookEdit` (authoring cells) — there is no
notebook-execute or kernel-attach tool of any kind. So the Phase 0 question
"can you execute notebook cells on the Colab kernel yourself?" has a definite
answer: **no**.

**Consequence.** The workflow is: I write/edit the notebook cells and the
package, then hand you a short numbered list of cells to run; you paste the
outputs back. All heavy logic therefore lives in `src/oilspill`, never in cells —
which the spec wanted regardless.

### D0.3 Local Python is 3.13, not 3.12

**Evidence.** `python --version` → `Python 3.13.15`; it is the only interpreter
registered (`py -0` lists `-V:3.13 *` alone).

**Consequence.** `requires-python = ">=3.12"`. rasterio 1.5.1 installs from a
wheel on 3.13/Windows (see D0.4), so this costs nothing locally. Colab is 3.12;
the package is version-agnostic across the two, and `requirements.lock.txt`
records each environment separately.

### D0.4 No Docker needed for local backend work — rasterio wheels work on Windows

**Evidence.** In a clean venv on Windows 11 / Python 3.13:
`rasterio-1.5.1`, `shapely-2.1.2`, `geopandas-1.1.4`, `pyproj-3.8.0`,
`pyogrio-0.13.0`, `onnxruntime-1.30.0`, `fastapi-0.141.1` all installed from
wheels with no build step and no GDAL system install.

**Also.** `docker --version` → not found. Docker is **not** installed on this
machine, so `backend/Dockerfile` and `deployment/docker-compose.yml` can be
authored but **cannot be verified locally**. That verification is deferred and
will be reported as unverified until Docker exists or Render builds the image.

### D0.5 The venv and data root live outside OneDrive

**Evidence.** The repo sits at `C:\Users\itanm\OneDrive\Desktop\sih`, inside an
actively syncing OneDrive folder.

**Consequence.** A venv (hundreds of MB) and the tile dataset (many GB) inside a
synced folder would upload continuously and could exhaust the OneDrive quota.
So:

- venv at `C:\Users\itanm\.venvs\oilspill`;
- bulky artifacts under `OILSPILL_DATA_DIR`, default `C:/oilspill-data`, read by
  `config.py` from `.env`.

`data/splits/` stays in the repo because it is small and must be version-tracked.

### D0.6 Windows Smart App Control blocks pyproj and pyogrio — both removed as dependencies

**Evidence.** `import pyproj` and `import pyogrio` fail on this machine. The
surfaced message is misleading ("GDAL DLL could not be found. It must be on the
system PATH.") because pyogrio's `_env.py` catches the real error and falls back
to a PATH scan. Loading the extension directly gives the true cause:

```
ImportError: DLL load failed while importing _io_direct:
An Application Control policy has blocked this file.
```

`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState`
is **1** — Smart App Control is *enforcing*. The DLLs are present
(`pyogrio.libs/` holds all 23, including `gdal-*.dll`); they are simply refused
at load time. There is no `Zone.Identifier` stream, so `Unblock-File` is not a
fix.

Import survey in the venv:

| Imports fine | Blocked |
| --- | --- |
| numpy, scipy, rasterio, shapely, onnx, **onnxruntime**, matplotlib, pandas, pyarrow, geopandas (as a module) | **pyproj**, **pyogrio** |

**Decision.** Do not disable Smart App Control. Turning it off is irreversible
without reinstalling Windows, it is the user's machine policy, and it is not
mine to change. Engineer around it instead:

- **All CRS work goes through rasterio's bundled PROJ.** rasterio's wheel ships
  GDAL 3.12.4 / PROJ 9.8.1 and loads fine; `rasterio.warp.transform` was
  verified to reproject EPSG:4326 → EPSG:32643 correctly. `oilspill.crsutil`
  wraps this and provides `estimate_utm_epsg`, `to_crs`, `buffer_metres` and
  `metric_properties`, replacing every use of pyproj and of
  `GeoSeries.estimate_utm_crs` / `.to_crs`.
- **Shapefiles are read with pyshp**, not OGR. `oilspill.landmask` reads
  `ne_10m_land.zip` with pure-Python pyshp and builds shapely geometries plus an
  STRtree, so no OGR is involved. Verified: 11 land polygons read for the
  Gulf-of-Guinea window.
- `pyproj` and `geopandas` are **dropped from the runtime dependencies**;
  `pyshp` is added. This also shrinks the API Docker image.

**Portability note.** This makes the package work on *more* machines, not fewer:
nothing now requires an OGR driver stack. Colab (Linux) is unaffected either way.

---

## Phase 1 — data spike

### D1.1 `hitl_cls IS NOT NULL` is silently ignored by the Cerulean API

**Evidence.** Measured against the live service on 2026-09-27 (`numberMatched`
from `/collections/public.slick_plus/items`):

| CQL2 `filter` (`filter-lang=cql2-text`) | `numberMatched` | `hitl_cls` in returned rows |
| --- | --- | --- |
| *(no filter)* | 1,334,040 | `None, None, None` |
| `hitl_cls IS NOT NULL` | 1,326,855 | `None, None, None` |
| `hitl_cls IS NULL` | **1,326,855** | `None, None, None` |
| `hitl_cls > 0` | 7,185 | `9, 3, 4` |
| `hitl_cls IN (3,4,5)` | 2,473 | `3, 4, 4` |
| `hitl_cls = 5` | 65 | `5, 5, 5` |

`IS NULL` and `IS NOT NULL` return an **identical** count, and the rows are not
filtered at all — the predicate is a no-op in this tipg deployment. Comparison
and `IN` predicates work correctly and do return reviewed rows.

**Decision.** The Tier-1 ("human-reviewed") predicate is `hitl_cls > 0`, not
`hitl_cls IS NOT NULL`. For Tier-2 ("no `hitl_cls`") the negative cannot be
expressed server-side at all, so `cerulean_api.unreviewed_slicks` filters
`hitl_cls` falsiness **client-side** after a `cls`/`machine_confidence` query.

**Scale consequence.** The entire Tier-1 universe is 7,185 slicks, of which
2,473 are in classes 3/4/5. That is the hard ceiling on our test set and is
worth stating in the README rather than discovering during Phase 6.

### D1.2 RESOLVED — the PC item id is the Cerulean scene id minus its last token

The spec flagged the Cerulean↔Planetary-Computer id-format relationship as
UNVERIFIED. It is now measured, and the answer is exact.

**Evidence.** All three Phase-1 scenes, 2026-09-27. An exact-id lookup **missed
every time**; the id with its trailing 4-hex token removed hit every time:

| Cerulean `s1_scene_id` | Planetary Computer item id |
| --- | --- |
| `S1A_…_058427_0739ED_999A` | `S1A_…_058427_0739ED` |
| `S1A_…_050360_061036_7DC5` | `S1A_…_050360_061036` |
| `S1A_…_063959_080BBB_227B` | `S1A_…_063959_080BBB` |

The dropped token is the SAFE product-unique-id. Cerulean keeps it; PC does not.

**Decision.** `s1_source.find_item` tries, in order: (1) the exact id, (2) the id
with a trailing `_XXXX` stripped, (3) a ±2 min datetime search matched on the
product-id prefix. Step 2 is what fires, turning a 100-item search into a single
GET. Step 3 is kept as a backstop for id spellings we have not seen. Verified
after the change: the scene resolves via `id_minus_suffix` in one request.

### D1.3 `area` from the Cerulean API is in square metres

**Evidence.** Sample rows: `area=16129036.88` and `area=4846477.77` for slicks
whose `machine_confidence` is ~0.94/0.96. Interpreted as m² these are 16.1 km²
and 4.8 km², which are plausible slick sizes; any other unit is not.

**Consequence.** The "area ≥ 5 km²" filter is `area > 5.0e6`.

### D1.4 GCP scaling for decimated reads uses plain division

GDAL GCP `(col, row)` are pixel-**corner** coordinates, so under decimation by
`f` a corner at full-res `c` maps to `c / f` with no ±0.5 correction.
`georef.read_decimated` additionally scales each axis by its *own* realised
ratio (`ds.height / out_h`, `ds.width / out_w`), because integer division makes
those differ from the nominal factor when the raster size is not an exact
multiple. Measured alignment is reported in `reports/phase1/spike_results.json`.

### D1.5 Criterion (c) is measured as coastline-to-brightness-edge distance

The criterion asks that "Natural Earth 10 m land lines up within ~2 px". Over
Sentinel-1 GRD, land is *bright*, not nodata, so there is no nodata edge to
compare against. `spike_phase1.measure_land_alignment` therefore rasterises
Natural Earth land on our grid, thresholds the SAR at the midpoint between mean
water and mean land brightness, and reports the **median distance from the
resulting brightness edge to the nearest NE-land pixel**, in grid pixels. It
returns `None` (criterion not applicable) when land covers <2% or >98% of the
window. This is a weaker test than a true coastline fit, and it is reported as
such rather than claimed as an exact registration measurement.

### D1.6 Criterion (d) is a containment *fraction*, not strict containment

**Evidence.** Slick 3098513 (Gulf of Mexico) sits 0.9987 inside its STAC
footprint — 0.13% of its area falls outside. Strict `footprint.contains(geom)`
therefore failed it, while its measured dB contrast was a healthy 3.39 dB and
the overlay shows correct alignment.

**Cause.** The STAC `geometry` is a coarse simplification of the real swath
outline (a handful of vertices for a 250 km swath). A slick lying along the
swath edge can cross that simplified boundary without anything being wrong.

**Decision.** `FOOTPRINT_MIN_FRAC = 0.99`. The spike records the actual fraction
in `spike_results.json` rather than a bare boolean, so the margin stays visible.

### D1.7 Phase 1 measured throughput — Phase 2 can afford 300–600 scenes

Per scene, whole-swath decimated read (factor 8) plus warp, measured on a home
Windows connection:

| Slick | Native size | Decimated | GCPs | Read | Warp |
| --- | --- | --- | --- | --- | --- |
| 3880640 | 16826×25487 | 2103×3185 | 210 | 9.3 s | 0.24 s |
| 3098513 | 16734×25887 | 2091×3235 | 210 | 2.0 s | 0.22 s |
| 4823849 | 19444×25692 | 2430×3211 | 231 | 3.4 s | 0.35 s |

Every asset exposed overviews `[2, 4, 8, 16, 32, 64]`, so the `out_shape` read
is served from the ×8 overview rather than pulling full resolution — which is
why a ~430 Mpx scene reads in seconds. Decoded output is ~6.7 M pixels ≈ 13 MB
per scene.

**Consequence.** At ~5 s/scene of I/O, 600 scenes is roughly an hour of reading,
so the Phase-2 target of 300–600 scenes is comfortable and the build should be
bounded by tiling and disk, not by network. Confirm on Colab before scaling —
Colab's egress is faster, but Drive writes are slower.
