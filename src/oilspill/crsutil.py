"""CRS transforms and metric geometry, routed through rasterio's bundled PROJ.

Why not pyproj or geopandas.to_crs? On this machine Windows Smart App Control
blocks pyproj's and pyogrio's unsigned native extensions
("An Application Control policy has blocked this file"), so ``import pyproj``
fails outright and ``GeoDataFrame.to_crs`` raises. rasterio's wheel bundles its
own GDAL 3.12 / PROJ 9.8 and loads fine, so every coordinate transform in this
package goes through :func:`rasterio.warp.transform`. See DECISIONS.md D0.6.

This keeps the package working with *no* pyproj dependency at all, which also
makes the Docker image smaller.
"""

from __future__ import annotations

import math

from rasterio.crs import CRS
from rasterio.warp import transform as _rio_transform
from rasterio.warp import transform_geom as _rio_transform_geom
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from . import config

WGS84 = CRS.from_epsg(config.CRS_EPSG)


def estimate_utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing a lon/lat point.

    Replaces ``GeoSeries.estimate_utm_crs``. Northern zones are 326xx, southern
    327xx. Longitude is normalised into [-180, 180) first so an antimeridian
    centroid cannot produce zone 61.
    """
    lon = ((lon + 180.0) % 360.0) - 180.0
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def utm_crs_for(geom: BaseGeometry) -> CRS:
    """A local UTM CRS appropriate for measuring ``geom``."""
    c = geom.centroid
    return CRS.from_epsg(estimate_utm_epsg(c.x, c.y))


def to_crs(geom: BaseGeometry, dst: CRS, src: CRS = WGS84) -> BaseGeometry:
    """Reproject a shapely geometry between two CRSs."""
    return shape(_rio_transform_geom(src, dst, mapping(geom)))


def transform_points(
    xs, ys, dst: CRS, src: CRS = WGS84
) -> tuple[list[float], list[float]]:
    """Reproject parallel coordinate sequences."""
    out_x, out_y = _rio_transform(src, dst, list(xs), list(ys))
    return list(out_x), list(out_y)


def metric_properties(geom: BaseGeometry) -> dict[str, float]:
    """Area (m2), perimeter (m), length (m) and Polsby-Popper for a lon/lat geometry.

    Measured in the geometry's local UTM zone, which is accurate to well under a
    percent for slick-sized features and is what Cerulean's own ``area`` /
    ``perimeter`` / ``polsby_popper`` fields are comparable against.

    ``length`` is the major-axis extent, taken as the longer side of the minimum
    rotated rectangle - the standard proxy for slick length, since a slick is a
    long thin feature and its skeleton length is expensive to compute.
    """
    utm = utm_crs_for(geom)
    g = to_crs(geom, utm)

    area = float(g.area)
    perimeter = float(g.length)

    try:
        mrr = g.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)[:5]
        sides = [
            math.dist(coords[i], coords[i + 1]) for i in range(min(4, len(coords) - 1))
        ]
        length = max(sides) if sides else 0.0
        width = min(sides) if sides else 0.0
    except Exception:  # noqa: BLE001 - degenerate geometries
        length = width = 0.0

    polsby_popper = (
        4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    )

    return {
        "area_m2": area,
        "perimeter_m": perimeter,
        "length_m": float(length),
        "width_m": float(width),
        "polsby_popper": float(polsby_popper),
        "utm_epsg": int(utm.to_epsg()),
    }


def buffer_metres(geom: BaseGeometry, metres: float) -> BaseGeometry:
    """Buffer a lon/lat geometry by a true ground distance.

    Done by going out to local UTM, buffering there, and coming back - a degree
    buffer would be anisotropic and wrong away from the equator.
    """
    utm = utm_crs_for(geom)
    return to_crs(to_crs(geom, utm).buffer(metres), WGS84, src=utm)


def metres_to_degrees(metres: float, lat: float) -> tuple[float, float]:
    """(dlon, dlat) spanning ``metres`` at latitude ``lat``.

    Cheap approximation for padding windows, where isotropy matters more than
    exactness. Use :func:`buffer_metres` when the result feeds a measurement.
    """
    dlat = metres / 111_320.0
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    dlon = metres / (111_320.0 * cos_lat)
    return dlon, dlat
