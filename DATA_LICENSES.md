# Data, model and code licences

This project is **non-commercial**: education, research, portfolio and
conservation demonstration only. That restriction is not a preference — it is
inherited from the licence and terms of the label data (see
[Cerulean](#1-skytruth-cerulean-labels) below), and it propagates to the trained
weights and to the derived dataset.

---

## Summary

| Artifact | Licence | Commercial use |
| --- | --- | --- |
| Source code in this repository | Apache-2.0 | Yes |
| Derived training dataset (tiles + masks) | CC BY-SA 4.0 | **No** — see §1 |
| Trained weights (`best.pt`, `model.onnx`) | CC BY-SA 4.0 | **No** — see §1 |
| Detections produced by the model | CC BY-SA 4.0 | **No** — see §1 |
| Reports, metrics, figures | CC BY-SA 4.0 | **No** — see §1 |

The weights and dataset are share-alike because they are derived works of
CC BY-SA 4.0 labels. The code is separable from the labels and is Apache-2.0.

---

## 1. SkyTruth Cerulean labels

**Source.** <https://cerulean.skytruth.org> · API
<https://api.cerulean.skytruth.org>

**Licence.** CC BY-SA 4.0 — <https://creativecommons.org/licenses/by-sa/4.0/>

**Additional restriction.** SkyTruth's terms of service state that only
environmental conservation applications are permitted. This is *more*
restrictive than CC BY-SA 4.0 alone, and it is the reason the whole project is
non-commercial.

**What we use it for.** Slick polygons, classes, `hitl_cls` human review flags,
confidences and `s1_scene_id` values are used as supervision labels. We do not
redistribute the Cerulean database; we redistribute a derived dataset of image
tiles and rasterised masks, under the same CC BY-SA 4.0 licence.

**What we do not use.** Cerulean's training data, model weights and UI source
are not public. The `ceruleanml` GCS bucket returns 403. We made no attempt to
obtain any of them; this reproduction is trained from scratch.

**Required attribution.**

> SkyTruth Cerulean (cerulean.skytruth.org). 2026. CC BY-SA 4.0. Accessed
> 2026-09-27.

---

## 2. Copernicus Sentinel-1 imagery

**Source.** ESA / European Commission Copernicus programme, accessed through
Microsoft Planetary Computer,
<https://planetarycomputer.microsoft.com/api/stac/v1>, collection
`sentinel-1-grd`.

**Licence.** Copernicus Sentinel data are provided under the *Legal Notice on
the use of Copernicus Sentinel Data and Service Information*, which permits
free use and redistribution, including for commercial purposes, provided the
data are attributed and any modification is stated.

**Required attribution.**

> Contains modified Copernicus Sentinel data 2026.

"Modified" is accurate and required: we decimate, reproject onto an EPSG:4326
grid via GCPs, convert digital numbers to pseudo-dB and quantise to 8 bits.

**Planetary Computer.** Access is anonymous; assets are signed with short-lived
SAS tokens. Use is subject to the Planetary Computer terms of use.

---

## 3. Natural Earth (coastline / land mask)

**Source.** <https://www.naturalearthdata.com> — `ne_10m_land`.

**Licence.** Public domain. No attribution required, but offered:

> Made with Natural Earth.

---

## 4. OpenStreetMap basemap tiles (frontend)

**Source.** OpenStreetMap standard raster tiles.

**Licence.** Map data © OpenStreetMap contributors, ODbL 1.0.

**Required attribution**, displayed in the map control and the footer:

> © OpenStreetMap contributors

The frontend respects the OSM Tile Usage Policy: no bulk downloading, no tile
scraping, a configurable style URL (`VITE_MAP_STYLE`) so a self-hosted or
commercial tile source can be substituted for any deployment with real traffic.

---

## 5. Software dependencies

| Package | Licence |
| --- | --- |
| PyTorch, torchvision | BSD-3-Clause |
| segmentation-models-pytorch 0.5.0 | MIT |
| timm | Apache-2.0 |
| rasterio, shapely, pyproj, geopandas, pyogrio | BSD-3-Clause / BSD-2-Clause |
| GDAL (bundled in rasterio wheels) | MIT/X |
| ONNX, ONNX Runtime | Apache-2.0 / MIT |
| FastAPI, Starlette, uvicorn, pydantic | MIT / BSD-3-Clause |
| SQLAlchemy, Alembic, GeoAlchemy2 | MIT |
| albumentations | MIT |
| MapLibre GL JS | BSD-3-Clause |
| React, Vite, TypeScript | MIT / Apache-2.0 |

ImageNet-pretrained encoder weights come from `timm` and are used for
non-commercial research, consistent with the ImageNet terms of access.

---

## 6. Third-party evaluation dataset (optional, Phase 6)

**Source.** Trujillo-Acatitla et al., Zenodo record
[8346860](https://doi.org/10.5281/zenodo.8346860).

**Licence.** CC BY 4.0 — commercial use permitted, attribution required.

Used only as an out-of-distribution check. It is binary (oil / not oil) and
calibrated σ⁰ dB, so it is **not** directly comparable to our 4-class
uncalibrated-DN model; any number reported from it is labelled as such. Only the
subset required is downloaded — the full record is 40.7 GB.

---

## 7. What this means in practice

You may: read, run, fork and modify the code commercially (Apache-2.0); use the
dataset, weights and detections for research, education and environmental
conservation, with attribution, sharing derivatives alike.

You may not: use the weights, dataset or detections in a commercial product or
for any non-conservation purpose, or relicense them under more permissive terms.

Nothing here is legal advice. If your use is anywhere near the commercial line,
read SkyTruth's terms yourself and ask them.
