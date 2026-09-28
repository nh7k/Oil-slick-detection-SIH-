"""Probabilities -> slick polygons, reproducing Cerulean's post-processing [SRC].

1. P(oil) = P(infra) + P(natural) + P(vessel)
2. islands of P(oil) > p_low (connected components)
3. keep an island only if max P(oil) >= p_seed
4. inside kept islands, trim to P(oil) >= p_trim and polygonize
5. drop polygons > edge_frac within edge_deg of the scene's nodata edge
6. confidence = MEDIAN P(oil) inside the polygon (mean/max also reported)
7. class = argmax of per-class probability sums inside the polygon
Plus our cleanup: make_valid, land buffer removal, min area, simplify.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely import make_valid
from shapely.geometry import MultiPolygon, mapping, shape
from shapely.ops import unary_union

from . import config
from .crsutil import metric_properties
from .georef import GridWindow

log = logging.getLogger(__name__)


def _polys(geom) -> list:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_polys(g))
        return out
    return []


def extract_slicks(
    probs: np.ndarray,
    window: GridWindow,
    valid: np.ndarray,
    cfg: config.PostprocessConfig = config.POSTPROCESS,
    land=None,
    overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return GeoJSON-like features (EPSG:4326) with Cerulean-style properties."""
    c = asdict(cfg)
    c.update(overrides or {})
    p_oil = probs[1:].sum(axis=0)
    p_oil[~valid] = 0.0

    islands, n = ndimage.label(p_oil > c["p_low"], structure=np.ones((3, 3)))
    if n == 0:
        return []
    maxes = ndimage.maximum(p_oil, islands, index=np.arange(1, n + 1))
    keep_ids = np.nonzero(np.asarray(maxes) >= c["p_seed"])[0] + 1
    if keep_ids.size == 0:
        return []
    kept = np.isin(islands, keep_ids)
    trimmed = kept & (p_oil >= c["p_trim"])
    if not trimmed.any():
        return []

    # Polygonize each connected trimmed region inside its own bounding slice
    # (never full-raster work per region - a scene can have thousands of specks).
    # Cerulean returns ONE multipolygon per slick, not one polygon per fragment.
    # A slick trimmed at p_trim often breaks into pieces a few pixels apart, so
    # fragments within `group_px` pixels of each other are grouped into one slick.
    g = int(c.get("group_px", 0))
    grouping = ndimage.binary_dilation(trimmed, iterations=g) if g > 0 else trimmed
    regions, nreg = ndimage.label(grouping, structure=np.ones((3, 3)))
    regions = np.where(trimmed, regions, 0)
    edge_zone = _edge_zone(valid, c["edge_deg"])
    px = config.PIXEL_DEG
    lat_mid = 0.5 * (window.bounds[1] + window.bounds[3])
    px_area_m2 = (px * 111_320.0) ** 2 * max(np.cos(np.radians(lat_mid)), 0.01)
    min_px = max(1, int(0.5 * c["min_area_m2"] / px_area_m2))  # cheap pre-filter; exact check below
    sizes = ndimage.sum_labels(np.ones_like(regions, dtype=np.int32), regions, index=np.arange(1, nreg + 1))
    slices = ndimage.find_objects(regions)
    land_mask = _burn(land, window) if land is not None else None
    t = window.transform
    feats: list[dict[str, Any]] = []
    for rid0, sl in enumerate(slices):
        rid = rid0 + 1
        if sl is None or sizes[rid0] < min_px:
            continue
        region = regions[sl] == rid
        # edge rule [SRC]
        if (edge_zone[sl] & region).sum() > c["edge_frac"] * region.sum():
            continue
        if land_mask is not None:
            region = region & ~land_mask[sl]
            if region.sum() < min_px:
                continue
        # fill pinholes (< max_hole_px) so speckle does not explode the vertex count
        holes = ndimage.binary_fill_holes(region) & ~region
        if holes.any():
            hl, nh = ndimage.label(holes)
            hs = ndimage.sum_labels(holes, hl, index=np.arange(1, nh + 1))
            small = np.nonzero(hs < c.get("max_hole_px", 16))[0] + 1
            region = region | np.isin(hl, small)
        sub_t = t @ t.translation(sl[1].start, sl[0].start)
        polys = []
        for gj, val in shapes(region.astype(np.uint8), mask=region, transform=sub_t, connectivity=8):
            if val:
                g = shape(gj)
                polys.extend(_polys(g if g.is_valid else make_valid(g)))
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else unary_union(polys)
        geom = geom.simplify(c["simplify_px"] * px, preserve_topology=True)
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            geom = MultiPolygon([geom])
        elif geom.geom_type != "MultiPolygon":
            geom = MultiPolygon(_polys(geom))
            if geom.is_empty:
                continue
        m = metric_properties(geom)
        if m["area_m2"] < c["min_area_m2"]:
            continue

        vals = p_oil[sl][region]
        cls_sums = probs[1:, sl[0], sl[1]][:, region].sum(axis=1)
        cls = int(np.argmax(cls_sums)) + 1
        cen = geom.centroid
        feats.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "cls": cls,
                "cls_name": config.MODEL_CLASSES[cls],
                "confidence": float(np.median(vals)),
                "confidence_mean": float(vals.mean()),
                "confidence_max": float(vals.max()),
                "class_prob_share": [float(v) for v in cls_sums / max(cls_sums.sum(), 1e-9)],
                "area_km2": m["area_m2"] / 1e6,
                "perimeter_km": m["perimeter_m"] / 1e3,
                "length_km": m["length_m"] / 1e3,
                "polsby_popper": m["polsby_popper"],
                "centroid_lon": cen.x,
                "centroid_lat": cen.y,
                "bbox": list(geom.bounds),
                "n_pixels": int(region.sum()),
            },
        })
    feats.sort(key=lambda f: -f["properties"]["confidence"])
    return feats


def _edge_zone(valid: np.ndarray, edge_deg: float) -> np.ndarray:
    """Pixels within edge_deg of nodata (the swath edge) or the raster border."""
    k = max(1, int(round(edge_deg / config.PIXEL_DEG)))
    invalid = ~valid
    invalid = np.pad(invalid, 1, constant_values=True)
    grown = ndimage.binary_dilation(invalid, iterations=k)[1:-1, 1:-1]
    return grown & valid


def _burn(geom, window: GridWindow) -> np.ndarray:
    return rasterize([(geom, 1)], out_shape=window.shape, transform=window.transform,
                     fill=0, dtype="uint8").astype(bool)


def to_feature_collection(features: list[dict[str, Any]], **props) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features, **props}
