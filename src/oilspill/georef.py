"""Georeferencing: the Cerulean grid, and GCP-based warping of Sentinel-1 GRD.

Planetary Computer's ``sentinel-1-grd`` COGs carry no affine geotransform - only
a set of ground control points (~210 tie points). The STAC ``proj:transform`` is
a bbox approximation and must never be used for pixel georeferencing. Everything
here goes through ``rasterio.warp.reproject`` with explicit GCPs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.warp import reproject

from . import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridWindow:
    """A rectangular, grid-aligned window in EPSG:4326.

    ``col_off``/``row_off`` are integer offsets from the global grid origin at
    (GRID_ORIGIN_LON, GRID_ORIGIN_LAT), so two windows built independently are
    always pixel-consistent with each other.
    """

    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        px = config.PIXEL_DEG
        return Affine(
            px, 0.0, config.GRID_ORIGIN_LON + self.col_off * px,
            0.0, -px, config.GRID_ORIGIN_LAT - self.row_off * px,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) of the window's outer edges."""
        px = config.PIXEL_DEG
        west = config.GRID_ORIGIN_LON + self.col_off * px
        north = config.GRID_ORIGIN_LAT - self.row_off * px
        return (west, north - self.height * px, west + self.width * px, north)


def lon_to_col(lon: float) -> float:
    """Fractional global grid column of a longitude."""
    return (lon - config.GRID_ORIGIN_LON) / config.PIXEL_DEG


def lat_to_row(lat: float) -> float:
    """Fractional global grid row of a latitude."""
    return (config.GRID_ORIGIN_LAT - lat) / config.PIXEL_DEG


def grid_window_from_bounds(
    bounds: tuple[float, float, float, float],
    pad_m: float = 0.0,
) -> GridWindow:
    """Snap a lon/lat bbox outward onto the grid, optionally padded.

    ``pad_m`` is converted to degrees using the latitude of the bbox centre for
    longitude, so the padding is roughly isotropic on the ground.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    if pad_m:
        lat_mid = 0.5 * (min_lat + max_lat)
        pad_lat = pad_m / 111_320.0
        # Guard against the cosine collapsing near the poles.
        cos_lat = max(np.cos(np.radians(lat_mid)), 0.01)
        pad_lon = pad_m / (111_320.0 * cos_lat)
        min_lon -= pad_lon
        max_lon += pad_lon
        min_lat -= pad_lat
        max_lat += pad_lat

    min_lat = max(min_lat, -90.0)
    max_lat = min(max_lat, 90.0)

    col0 = int(np.floor(lon_to_col(min_lon)))
    col1 = int(np.ceil(lon_to_col(max_lon)))
    # Rows increase southward, so max_lat gives the smaller row index.
    row0 = int(np.floor(lat_to_row(max_lat)))
    row1 = int(np.ceil(lat_to_row(min_lat)))
    return GridWindow(col0, row0, max(col1 - col0, 1), max(row1 - row0, 1))


def assert_on_grid(transform: Affine, crs, tol: float = 1e-9) -> None:
    """Fail loudly unless a raster sits exactly on the Cerulean grid.

    Called at every stage boundary so a CRS or half-pixel mistake cannot
    silently propagate into the dataset or the detections.
    """
    if crs is not None:
        epsg = rasterio.crs.CRS.from_user_input(crs).to_epsg()
        if epsg != config.CRS_EPSG:
            raise AssertionError(f"expected EPSG:{config.CRS_EPSG}, got EPSG:{epsg}")

    px = config.PIXEL_DEG
    if abs(transform.a - px) > tol or abs(transform.e + px) > tol:
        raise AssertionError(
            f"pixel size {transform.a}, {transform.e} != +/-{px}"
        )
    if abs(transform.b) > tol or abs(transform.d) > tol:
        raise AssertionError(f"transform is rotated/sheared: {transform}")

    col = (transform.c - config.GRID_ORIGIN_LON) / px
    row = (config.GRID_ORIGIN_LAT - transform.f) / px
    if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
        raise AssertionError(
            f"origin ({transform.c}, {transform.f}) is not grid-aligned "
            f"(fractional col={col}, row={row})"
        )


# ---------------------------------------------------------------------------
# GCP handling
# ---------------------------------------------------------------------------

def scale_gcps(
    gcps: list[GroundControlPoint], factor: float
) -> list[GroundControlPoint]:
    """Rescale GCP pixel coordinates for a decimated read.

    A GCP's ``(col, row)`` refers to a *pixel corner* in the full-resolution
    grid (GDAL convention). Under decimation by ``factor`` the corner at
    full-res coordinate ``c`` lands at ``c / factor``, so the mapping is a plain
    division - no +/-0.5 pixel-centre correction. See DECISIONS.md; the
    alternative was tested in Phase 1 and produced a visible half-pixel shift.
    """
    if factor == 1:
        return list(gcps)
    return [
        GroundControlPoint(
            row=g.row / factor, col=g.col / factor,
            x=g.x, y=g.y, z=g.z, id=g.id, info=g.info,
        )
        for g in gcps
    ]


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------

def read_decimated(
    ds: rasterio.DatasetReader,
    factor: int = config.READ_DECIMATION,
    band: int = 1,
) -> tuple[np.ndarray, list[GroundControlPoint], rasterio.crs.CRS]:
    """Decimated whole-band read plus GCPs rescaled to match.

    Uses ``out_shape`` so GDAL can serve the request from the COG's internal
    overviews instead of pulling every full-resolution block.
    """
    out_h = max(ds.height // factor, 1)
    out_w = max(ds.width // factor, 1)
    arr = ds.read(
        band,
        out_shape=(out_h, out_w),
        resampling=Resampling.average,
    )
    gcps, gcp_crs = ds.gcps
    if not gcps:
        raise ValueError("dataset has no GCPs; cannot georeference")
    # Effective factor differs from `factor` when height/width are not exact
    # multiples, and the two axes can differ. Scale each axis by its own ratio.
    row_factor = ds.height / out_h
    col_factor = ds.width / out_w
    scaled = [
        GroundControlPoint(
            row=g.row / row_factor, col=g.col / col_factor,
            x=g.x, y=g.y, z=g.z, id=g.id, info=g.info,
        )
        for g in gcps
    ]
    return arr, scaled, gcp_crs


def warp_to_grid(
    src: np.ndarray,
    gcps: list[GroundControlPoint],
    gcp_crs,
    window: GridWindow,
    resampling: Resampling = Resampling.average,
    src_nodata: float = config.NODATA,
    dst_nodata: float = config.NODATA,
    dtype: str = "float32",
) -> np.ndarray:
    """Reproject a GCP-referenced array onto a Cerulean-grid window."""
    dst = np.full(window.shape, dst_nodata, dtype=dtype)
    reproject(
        source=src,
        destination=dst,
        gcps=gcps,
        src_crs=gcp_crs,
        src_nodata=src_nodata,
        dst_transform=window.transform,
        dst_crs=config.CRS,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    assert_on_grid(window.transform, config.CRS)
    return dst


def db_to_uint8(db: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """Scale pseudo-dB to uint8 0-255 with a frozen clip range.

    0 is reserved for nodata, matching Cerulean's convention that a zero input
    pixel means "no data" [SRC]. Valid data is therefore mapped into 1-255.
    """
    lo, hi = clip
    if not hi > lo:
        raise ValueError(f"invalid dB clip range {clip}")
    valid = db > 0.0  # dn_to_db maps DN==0 to exactly 0.0
    scaled = (db - lo) / (hi - lo)
    out = np.clip(np.round(scaled * 254.0) + 1.0, 1, 255).astype(np.uint8)
    out[~valid] = config.NODATA
    return out


def write_grid_geotiff(path, arr: np.ndarray, window: GridWindow, dtype=None, nodata=None):
    """Write a single-band GeoTIFF carrying the exact grid transform."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_on_grid(window.transform, config.CRS)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype or arr.dtype.name,
        "crs": config.CRS,
        "transform": window.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path
