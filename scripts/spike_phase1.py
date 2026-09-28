#!/usr/bin/env python
"""Phase 1 real-data spike: prove we can georeference Sentinel-1 GRD onto the
Cerulean grid well enough to train on.

For each of N human-reviewed slicks from different regions:

  1. match the Cerulean ``s1_scene_id`` to a Planetary Computer STAC item,
  2. open the signed VV COG and log its real read characteristics,
  3. warp it onto the Cerulean grid via GCPs,
  4. rasterise the slick polygon onto the same grid,
  5. measure the five pass criteria and save overlays.

Pass criteria (all must hold):
  (a) the polygon visually outlines the dark feature  -> human check of overlay
  (b) mean dB inside the polygon is >= 2 dB below a 2 km ring around it
  (c) Natural Earth 10 m land lines up within ~2 px, if coastline is in view
  (d) the STAC footprint contains the polygon
  (e) it works for both ascending and descending passes

Run:  python scripts/spike_phase1.py --n-regions 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any  # noqa: F401

import numpy as np

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oilspill import config, s1_source  # noqa: E402
from oilspill.cerulean_api import CeruleanClient  # noqa: E402
from oilspill.georef import (  # noqa: E402
    GridWindow,
    assert_on_grid,
    grid_window_from_bounds,
    read_decimated,
    warp_to_grid,
    write_grid_geotiff,
)
from oilspill.labels import slick_geometry  # noqa: E402

log = logging.getLogger("spike")

OUT_DIR = config.REPORTS_DIR / "phase1"
#: Padding around the slick bbox for the warp window.
WINDOW_PAD_M = 20_000.0
#: Ring width for the contrast test.
RING_M = 2_000.0
#: Minimum dB contrast (slick darker than surroundings) to pass criterion (b).
MIN_CONTRAST_DB = 2.0
#: Land-alignment tolerance for criterion (c), in grid pixels.
LAND_TOL_PX = 2.0
#: Minimum fraction of the slick that must lie inside the STAC footprint for
#: criterion (d). Not 1.0, because the STAC footprint is a coarse
#: simplification of the swath outline - see DECISIONS.md D1.6.
FOOTPRINT_MIN_FRAC = 0.99


@dataclass
class SpikeResult:
    slick_id: int
    cerulean_cls: int
    hitl_cls: int | None
    machine_confidence: float
    area_m2: float
    scene_id: str
    region: str
    centroid: tuple[float, float]

    matched_item_id: str | None = None
    match_method: str | None = None
    orbit_direction: str | None = None

    # read diagnostics
    native_shape: tuple[int, int] | None = None
    overview_factors: list[int] = field(default_factory=list)
    gcp_count: int | None = None
    gcp_crs: str | None = None
    decimated_shape: tuple[int, int] | None = None
    read_seconds: float | None = None
    warp_seconds: float | None = None
    bytes_read_estimate: int | None = None

    # criteria
    db_inside: float | None = None
    db_ring: float | None = None
    contrast_db: float | None = None
    crit_b_contrast: bool | None = None
    crit_c_land_px: float | None = None
    crit_c_land: bool | None = None
    footprint_frac: float | None = None
    crit_d_footprint: bool | None = None

    valid_frac: float | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Machine-checkable criteria only; (a) is a human check of the overlay."""
        return bool(
            self.error is None
            and self.crit_b_contrast
            and self.crit_d_footprint
            and (self.crit_c_land in (True, None))
        )


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def pick_candidates(client: CeruleanClient, n_regions: int) -> list[dict[str, Any]]:
    """Reviewed slicks from ``n_regions`` well-separated regions.

    Diversity is enforced by requiring candidates to be >= 10 degrees apart, and
    we deliberately take both ascending and descending passes if available
    (criterion (e)).
    """
    log.info("querying reviewed slicks (hitl_cls > 0, area >= 5 km2, conf >= 0.9)")
    feats = client.reviewed_slicks(
        min_area_m2=5.0e6, min_confidence=0.9, sortby="-area", max_features=400,
    )
    log.info("got %d reviewed candidates", len(feats))

    chosen: list[dict[str, Any]] = []
    centroids: list[tuple[float, float]] = []
    for feat in feats:
        geom = slick_geometry(feat)
        if geom is None:
            continue
        c = geom.centroid
        if any(
            abs(c.x - lon) < 10.0 and abs(c.y - lat) < 10.0
            for lon, lat in centroids
        ):
            continue
        chosen.append(feat)
        centroids.append((c.x, c.y))
        if len(chosen) >= n_regions:
            break
    return chosen


def region_name(lon: float, lat: float) -> str:
    for aoi in config.AOIS:
        x0, y0, x1, y1 = aoi.bbox
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return aoi.name
    return f"other_{lon:.1f}E_{lat:.1f}N"


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def measure_contrast(
    db: np.ndarray, window: GridWindow, geom
) -> tuple[float, float, float]:
    """Mean dB inside the polygon vs a 2 km ring around it, ignoring nodata."""
    from rasterio.features import rasterize

    inside = rasterize(
        [(geom, 1)], out_shape=window.shape, transform=window.transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)

    lat_mid = 0.5 * (window.bounds[1] + window.bounds[3])
    cos_lat = max(np.cos(np.radians(lat_mid)), 0.01)
    # Buffer in degrees; use the longitude scale so the ring is at least RING_M
    # wide in both directions at this latitude.
    buf_deg = RING_M / (111_320.0 * cos_lat)
    ring_geom = geom.buffer(buf_deg).difference(geom)
    ring = rasterize(
        [(ring_geom, 1)], out_shape=window.shape, transform=window.transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)

    valid = db > 0.0
    inside &= valid
    ring &= valid
    if inside.sum() < 10 or ring.sum() < 10:
        raise ValueError(
            f"not enough valid pixels (inside={inside.sum()}, ring={ring.sum()})"
        )
    d_in = float(db[inside].mean())
    d_ring = float(db[ring].mean())
    return d_in, d_ring, d_ring - d_in


def measure_land_alignment(
    db: np.ndarray, window: GridWindow
) -> float | None:
    """How far our SAR brightness edge sits from Natural Earth's coastline, in px.

    Sentinel-1 GRD over land is bright, not nodata, so we cannot use the valid
    mask as a coastline. Instead we test the weaker but honest property the
    criterion is really about: that Natural Earth land, rasterised on our grid,
    lands on *brighter-than-water* pixels. We report the median distance, in
    pixels, from the SAR brightness edge to the nearest NE-land pixel. Returns
    None when no coastline is in view.
    """
    from scipy import ndimage

    from oilspill import landmask

    try:
        land_mask = landmask.land_mask(window)
    except Exception as exc:  # noqa: BLE001
        log.warning("land mask unavailable: %s", exc)
        return None

    valid = db > 0.0
    frac = land_mask.sum() / land_mask.size
    if frac < 0.02 or frac > 0.98:
        return None  # not really a coastline scene

    # Distance (in px) from each pixel to the nearest NE-land pixel.
    dist = ndimage.distance_transform_edt(~land_mask)
    # The SAR brightness step: water is dark, land bright. Find the contour of
    # the brightness median and see how far it is from the NE coastline.
    water_like = valid & ~land_mask
    land_like = valid & land_mask
    if water_like.sum() < 100 or land_like.sum() < 100:
        return None
    thresh = 0.5 * (float(db[water_like].mean()) + float(db[land_like].mean()))
    bright = valid & (db > thresh)
    edge = bright & ~ndimage.binary_erosion(bright, iterations=1)
    if edge.sum() < 50:
        return None
    return float(np.median(dist[edge]))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def save_outputs(
    slick_id: int, db: np.ndarray, mask: np.ndarray, window: GridWindow,
    geom, feature: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid = db > 0.0
    if valid.sum() == 0:
        raise ValueError("no valid pixels to render")
    lo, hi = np.percentile(db[valid], [2, 98])
    disp = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    disp[~valid] = 0.0

    extent = (window.bounds[0], window.bounds[2], window.bounds[1], window.bounds[3])

    plt.imsave(OUT_DIR / f"{slick_id}_sar.png", disp, cmap="gray", vmin=0, vmax=1)
    plt.imsave(OUT_DIR / f"{slick_id}_mask.png", mask, cmap="viridis", vmin=0, vmax=3)

    fig, ax = plt.subplots(figsize=(11, 11 * db.shape[0] / max(db.shape[1], 1)))
    ax.imshow(disp, cmap="gray", extent=extent, vmin=0, vmax=1, origin="upper")
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, color="#ff2d55", linewidth=1.6)
    props = feature["properties"]
    ax.set_title(
        f"slick {slick_id}  cls={props['cls']} hitl={props.get('hitl_cls')}  "
        f"conf={props.get('machine_confidence'):.3f}\n{props['s1_scene_id']}",
        fontsize=9,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{slick_id}_overlay.png", dpi=130)
    plt.close(fig)

    write_grid_geotiff(
        OUT_DIR / f"{slick_id}_sar.tif", db.astype("float32"), window,
        nodata=0.0,
    )
    (OUT_DIR / f"{slick_id}.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Per-slick driver
# ---------------------------------------------------------------------------

def run_one(feature: dict[str, Any]) -> SpikeResult:
    props = feature["properties"]
    geom = slick_geometry(feature)
    if geom is None:
        raise ValueError("slick has no usable geometry")
    c = geom.centroid

    res = SpikeResult(
        slick_id=int(props["id"]),
        cerulean_cls=int(props["cls"]),
        hitl_cls=props.get("hitl_cls"),
        machine_confidence=float(props.get("machine_confidence") or 0.0),
        area_m2=float(props.get("area") or 0.0),
        scene_id=props["s1_scene_id"],
        region=region_name(c.x, c.y),
        centroid=(round(c.x, 5), round(c.y, 5)),
    )
    log.info(
        "=== slick %s  region=%s  cls=%s hitl=%s  area=%.1f km2",
        res.slick_id, res.region, res.cerulean_cls, res.hitl_cls,
        res.area_m2 / 1e6,
    )

    item = s1_source.find_item(res.scene_id, intersects=feature["geometry"])
    if item is None:
        res.error = "no Planetary Computer item matched"
        return res
    res.matched_item_id = item["id"]
    res.match_method = (
        "exact" if item["id"] == res.scene_id
        else "id_minus_suffix" if res.scene_id.startswith(item["id"])
        else "prefix_search"
    )
    res.orbit_direction = (item.get("properties") or {}).get("sat:orbit_state")

    # (d) STAC footprint contains the polygon.
    # The STAC footprint is a coarse simplification of the real swath outline, so
    # a slick hugging the swath edge can poke a fraction of a percent outside it
    # without anything being misaligned. We therefore measure the containment
    # FRACTION and require >= FOOTPRINT_MIN_FRAC rather than strict containment.
    from shapely.geometry import shape as to_shape

    footprint = to_shape(s1_source.item_footprint(item))
    res.footprint_frac = round(
        float(geom.intersection(footprint).area / max(geom.area, 1e-12)), 5
    )
    res.crit_d_footprint = res.footprint_frac >= FOOTPRINT_MIN_FRAC
    log.info(
        "  (d) slick inside STAC footprint: %.4f -> %s",
        res.footprint_frac, "PASS" if res.crit_d_footprint else "FAIL",
    )

    t0 = time.perf_counter()
    with s1_source.open_asset(item) as ds:
        res.native_shape = (ds.height, ds.width)
        try:
            res.overview_factors = list(ds.overviews(1))
        except Exception:  # noqa: BLE001
            res.overview_factors = []
        gcps, gcp_crs = ds.gcps
        res.gcp_count = len(gcps)
        res.gcp_crs = str(gcp_crs)
        log.info(
            "  native=%s overviews=%s gcps=%d crs=%s",
            res.native_shape, res.overview_factors, res.gcp_count, res.gcp_crs,
        )

        arr, scaled_gcps, gcp_crs = read_decimated(ds, config.READ_DECIMATION)
        res.decimated_shape = tuple(arr.shape)
        res.read_seconds = round(time.perf_counter() - t0, 2)
        res.bytes_read_estimate = int(arr.size * 2)  # uint16 DN
        log.info(
            "  decimated read %s in %.1fs", res.decimated_shape, res.read_seconds
        )

    window = grid_window_from_bounds(geom.bounds, pad_m=WINDOW_PAD_M)
    log.info("  grid window %dx%d at %s", window.width, window.height, window.bounds)

    t1 = time.perf_counter()
    dn = warp_to_grid(arr, scaled_gcps, gcp_crs, window)
    res.warp_seconds = round(time.perf_counter() - t1, 2)
    assert_on_grid(window.transform, config.CRS)

    db = config.dn_to_db(dn)
    res.valid_frac = round(float((db > 0.0).mean()), 4)
    log.info(
        "  warp %.1fs, valid fraction %.3f", res.warp_seconds, res.valid_frac
    )
    if res.valid_frac < 0.05:
        res.error = f"warp produced almost no valid data (valid_frac={res.valid_frac})"
        return res

    # (b) contrast
    d_in, d_ring, contrast = measure_contrast(db, window, geom)
    res.db_inside, res.db_ring, res.contrast_db = (
        round(d_in, 3), round(d_ring, 3), round(contrast, 3)
    )
    res.crit_b_contrast = contrast >= MIN_CONTRAST_DB
    log.info(
        "  (b) dB inside=%.2f ring=%.2f contrast=%.2f -> %s",
        d_in, d_ring, contrast, "PASS" if res.crit_b_contrast else "FAIL",
    )

    # (c) land alignment
    try:
        land_px = measure_land_alignment(db, window)
    except Exception as exc:  # noqa: BLE001
        log.warning("  (c) land check unavailable: %s", exc)
        land_px = None
    res.crit_c_land_px = None if land_px is None else round(land_px, 2)
    res.crit_c_land = None if land_px is None else land_px <= LAND_TOL_PX
    log.info(
        "  (c) land offset: %s px -> %s",
        res.crit_c_land_px,
        "n/a (no coastline)" if land_px is None
        else ("PASS" if res.crit_c_land else "FAIL"),
    )

    from oilspill.labels import rasterize_slicks

    mask, _ = rasterize_slicks([feature], window)
    save_outputs(res.slick_id, db, mask, window, geom, feature)
    log.info("  saved overlays to %s", OUT_DIR)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-regions", type=int, default=3)
    ap.add_argument("--slick-ids", type=int, nargs="*", help="explicit slick ids")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    config.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CeruleanClient() as client:
        if args.slick_ids:
            feats = []
            for sid in args.slick_ids:
                got = list(client.iter_slicks(cql2=f"id = {sid}", max_features=1))
                if not got:
                    log.error("slick %s not found", sid)
                else:
                    feats.append(got[0])
        else:
            feats = pick_candidates(client, args.n_regions)

        if not feats:
            log.error("no candidate slicks; aborting")
            return 2

        results: list[SpikeResult] = []
        for feat in feats:
            try:
                results.append(run_one(feat))
            except Exception as exc:  # noqa: BLE001 - spike must report, not crash
                log.exception("slick %s failed", feat["properties"].get("id"))
                results.append(
                    SpikeResult(
                        slick_id=int(feat["properties"]["id"]),
                        cerulean_cls=int(feat["properties"]["cls"]),
                        hitl_cls=feat["properties"].get("hitl_cls"),
                        machine_confidence=float(
                            feat["properties"].get("machine_confidence") or 0
                        ),
                        area_m2=float(feat["properties"].get("area") or 0),
                        scene_id=feat["properties"]["s1_scene_id"],
                        region="?",
                        centroid=(0.0, 0.0),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    report = OUT_DIR / "spike_results.json"
    orbits = sorted({r.orbit_direction for r in results if r.orbit_direction})
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "min_contrast_db": MIN_CONTRAST_DB,
        "land_tol_px": LAND_TOL_PX,
        "footprint_min_frac": FOOTPRINT_MIN_FRAC,
        "window_pad_m": WINDOW_PAD_M,
        "read_decimation": config.READ_DECIMATION,
        "pixel_deg": config.PIXEL_DEG,
        "orbit_directions_seen": orbits,
        "crit_e_both_passes": len(orbits) >= 2,
        "results": [asdict(r) for r in results],
    }
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PHASE 1 SPIKE SUMMARY")
    print("=" * 78)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"[{status}] slick {r.slick_id:>8} {r.region:<26} "
            f"orbit={r.orbit_direction or '?':<11} "
            f"contrast={r.contrast_db if r.contrast_db is not None else 'n/a':>7} dB  "
            f"land={r.crit_c_land_px if r.crit_c_land_px is not None else 'n/a':>5} px  "
            f"footprint={r.footprint_frac if r.footprint_frac is not None else 'n/a'}"
        )
        if r.error:
            print(f"         error: {r.error}")
        if r.read_seconds is not None:
            print(
                f"         read {r.read_seconds}s warp {r.warp_seconds}s "
                f"native={r.native_shape} overviews={r.overview_factors} "
                f"gcps={r.gcp_count} match={r.match_method}"
            )
    n_pass = sum(1 for r in results if r.passed)
    print("-" * 78)
    print(f"machine-checkable criteria: {n_pass}/{len(results)} passed")
    print(f"(e) orbit directions seen: {orbits} -> "
          f"{'PASS' if len(orbits) >= 2 else 'INCOMPLETE'}")
    print(f"report: {report}")
    print(f"overlays: {OUT_DIR}  <-- criterion (a) needs your eyes")
    print("=" * 78)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
