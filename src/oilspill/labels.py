"""Turning Cerulean slick polygons into rasterised label masks on our grid."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config
from .cerulean_api import label_tier, model_class
from .georef import GridWindow

log = logging.getLogger(__name__)


def clean_geometry(geom: BaseGeometry) -> BaseGeometry | None:
    """Repair a geometry, or return None if nothing usable survives."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        from shapely import make_valid

        geom = make_valid(geom)
        if geom.is_empty:
            return None
    if geom.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
        return None
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        geom = unary_union(polys)
    if geom.is_empty or geom.area <= 0:
        return None
    return geom


def slick_geometry(feature: dict[str, Any]) -> BaseGeometry | None:
    """The cleaned shapely geometry of one Cerulean slick feature."""
    if feature.get("geometry") is None:
        return None
    return clean_geometry(shape(feature["geometry"]))


def rasterize_slicks(
    features: Iterable[dict[str, Any]],
    window: GridWindow,
) -> tuple[np.ndarray, dict[str, int]]:
    """Burn slick features into a label mask on ``window``.

    Returns ``(mask, stats)``. Mask values are ``{0,1,2,3,255}`` where 255 is
    ignore. Painting order matters, so it is fixed and explicit:

    1. background (the initial fill),
    2. positive classes 1-3,
    3. ignore (255) last, so an AMBIGUOUS/ANTHRO slick overlapping a confident
       one cannot silently downgrade it to ignore... and conversely a positive
       label never overwrites an ignore region it fully contains.

    Hard negatives (NOT_OIL / LAND / SEA_ICE / ARTEFACT) stay 0 but are counted.
    """
    mask = np.zeros(window.shape, dtype=np.uint8)
    stats = {"positive": 0, "ignore": 0, "hard_negative": 0, "dropped": 0, "T1": 0, "T2": 0}

    positive: list[tuple[BaseGeometry, int]] = []
    ignore: list[tuple[BaseGeometry, int]] = []

    for feat in features:
        props = feat.get("properties", {})
        geom = slick_geometry(feat)
        if geom is None:
            stats["dropped"] += 1
            continue

        tier = label_tier(props)
        cls = model_class(props)

        if cls == config.IGNORE_INDEX:
            ignore.append((geom, config.IGNORE_INDEX))
            stats["ignore"] += 1
            continue
        if cls == 0:
            # Hard negative: correctly background, but worth counting/forcing
            # into the tile sample.
            stats["hard_negative"] += 1
            continue
        if tier is None:
            # Not reviewed and not confident enough - neither a label nor a
            # trustworthy negative, so mark it ignore rather than background.
            ignore.append((geom, config.IGNORE_INDEX))
            stats["ignore"] += 1
            continue

        positive.append((geom, cls))
        stats["positive"] += 1
        stats[tier] += 1

    if positive:
        rasterize(
            positive, out=mask, transform=window.transform,
            all_touched=False, default_value=1,
        )
    if ignore:
        rasterize(
            ignore, out=mask, transform=window.transform,
            all_touched=False, default_value=config.IGNORE_INDEX,
        )
    return mask, stats


def tier1_mask(
    features: Iterable[dict[str, Any]],
    window: GridWindow,
) -> np.ndarray:
    """Mask built from Tier-1 (human-reviewed) slicks only, for evaluation.

    Everything that is not a reviewed positive becomes ignore, so unreviewed
    machine labels never count as either a hit or a false positive.
    """
    mask = np.full(window.shape, config.IGNORE_INDEX, dtype=np.uint8)
    feats = list(features)

    reviewed_pos: list[tuple[BaseGeometry, int]] = []
    for feat in feats:
        props = feat.get("properties", {})
        if label_tier(props) != "T1":
            continue
        cls = model_class(props)
        if cls in config.OIL_CLASSES:
            geom = slick_geometry(feat)
            if geom is not None:
                reviewed_pos.append((geom, cls))

    # Only scenes with at least one reviewed slick are usable for T1 metrics.
    if not reviewed_pos:
        return mask

    mask[:] = 0
    unreviewed: list[tuple[BaseGeometry, int]] = []
    for feat in feats:
        props = feat.get("properties", {})
        if label_tier(props) == "T1":
            continue
        geom = slick_geometry(feat)
        if geom is not None:
            unreviewed.append((geom, config.IGNORE_INDEX))

    if unreviewed:
        rasterize(
            unreviewed, out=mask, transform=window.transform, all_touched=False,
        )
    rasterize(
        reviewed_pos, out=mask, transform=window.transform, all_touched=False,
    )
    return mask
