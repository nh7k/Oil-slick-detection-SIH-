"""Whole-scene preparation: PC GRD item -> pseudo-dB raster on the Cerulean grid.

This is the ONE code path that turns imagery into model input. The dataset
builder, the evaluation script, the API and the batch job all call
:func:`prepare_item`, so training and inference can never drift apart.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from . import config
from .georef import GridWindow, grid_window_from_bounds, read_decimated, warp_to_grid
from .s1_source import open_asset

log = logging.getLogger(__name__)

#: Storage encoding of pseudo-dB rasters: uint16 = round(dB * DB_SCALE); 0 = nodata.
DB_SCALE = 100.0


@dataclass
class SceneRaster:
    scene_id: str
    db: np.ndarray            # float32 pseudo-dB, 0.0 == nodata
    window: GridWindow
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return self.db > 0.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.window.bounds


def item_bounds(item: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = item.get("bbox")
    if bbox and len(bbox) == 4:
        return tuple(float(v) for v in bbox)  # type: ignore[return-value]
    from shapely.geometry import shape

    return tuple(shape(item["geometry"]).bounds)  # type: ignore[return-value]


def prepare_item(item: dict[str, Any], decimation: int = config.READ_DECIMATION) -> SceneRaster:
    """Read a whole PC ``sentinel-1-grd`` VV asset and warp it onto the grid."""
    t0 = time.perf_counter()
    ds = open_asset(item)
    try:
        arr, gcps, gcp_crs = read_decimated(ds, factor=decimation)
        native = (ds.height, ds.width)
    finally:
        ds.close()
    t_read = time.perf_counter() - t0

    window = grid_window_from_bounds(item_bounds(item), pad_m=2_000.0)
    t1 = time.perf_counter()
    dn = warp_to_grid(arr.astype("float32"), gcps, gcp_crs, window)
    db = config.dn_to_db(dn)
    db[dn <= 0] = 0.0
    t_warp = time.perf_counter() - t1

    props = item.get("properties", {})
    meta = {
        "native_shape": native,
        "read_s": round(t_read, 2),
        "warp_s": round(t_warp, 2),
        "datetime": props.get("datetime") or props.get("start_datetime"),
        "orbit_direction": props.get("sat:orbit_state"),
        "platform": props.get("platform"),
    }
    log.info("prepared %s shape=%s read=%.1fs warp=%.1fs", item.get("id"), db.shape, t_read, t_warp)
    return SceneRaster(scene_id=item["id"], db=db.astype("float32"), window=window, meta=meta)


def prepare_geotiff(path: str | Path) -> SceneRaster:
    """Prepare a user-supplied Sentinel-1 VV GeoTIFF.

    Accepted inputs (anything else raises ``ValueError``):

    * raw GRD measurement TIFF / COG (integer DN) georeferenced by GCPs
      (e.g. ``measurement/*-vv-*.tiff`` from a SAFE, or a PC ``vv`` asset);
    * integer DN GeoTIFF with an affine transform in any CRS;
    * float pseudo-dB GeoTIFF produced by this project (values ~20-60, 0 = nodata).

    Calibrated sigma0 in dB (negative values, e.g. Earth Engine exports) is
    rejected because the model was trained on uncalibrated pseudo-dB.
    """
    path = Path(path)
    with rasterio.open(path) as ds:
        if ds.count < 1:
            raise ValueError("GeoTIFF has no bands")
        gcps, gcp_crs = ds.gcps
        has_transform = ds.crs is not None and not ds.transform.is_identity
        if not gcps and not has_transform:
            raise ValueError("GeoTIFF is not georeferenced (no CRS/transform and no GCPs)")
        is_int = np.issubdtype(np.dtype(ds.dtypes[0]), np.integer)

        if gcps and not has_transform:
            # Full-resolution GRD (~10 m) -> decimate like the PC path; small files as-is.
            factor = config.READ_DECIMATION if max(ds.width, ds.height) > 8000 else 1
            arr, sgcps, gcrs = read_decimated(ds, factor=factor)
            from shapely.geometry import MultiPoint

            b = MultiPoint([(g.x, g.y) for g in gcps]).bounds
            window = grid_window_from_bounds(b, pad_m=1_000.0)
            src = warp_to_grid(arr.astype("float32"), sgcps, gcrs, window)
        else:
            b = transform_bounds(ds.crs, config.CRS, *ds.bounds)
            window = grid_window_from_bounds(b)
            nodata = ds.nodata if ds.nodata is not None else 0
            src = np.zeros(window.shape, dtype="float32")
            reproject(
                source=rasterio.band(ds, 1), destination=src,
                src_nodata=nodata, dst_nodata=0,
                dst_transform=window.transform, dst_crs=config.CRS,
                resampling=Resampling.average,
            )

    if is_int:
        db = config.dn_to_db(src)
        db[src <= 0] = 0.0
    else:
        valid = src[src != 0]
        if valid.size == 0:
            raise ValueError("GeoTIFF contains only nodata")
        med = float(np.median(valid))
        if med < 5.0:
            raise ValueError(
                f"float values look like calibrated sigma0 dB (median {med:.1f}); "
                "upload raw GRD DN (integer) or this project's pseudo-dB GeoTIFF"
            )
        db = src
        db[~np.isfinite(db)] = 0.0
    if not (db > 0).any():
        raise ValueError("GeoTIFF footprint contains no valid pixels")
    return SceneRaster(scene_id=path.stem, db=db.astype("float32"), window=window,
                       meta={"source": "upload"})


def save_scene(scene: SceneRaster, path: str | Path) -> Path:
    """Store pseudo-dB as compact uint16 GeoTIFF (dB*100, 0=nodata)."""
    from .georef import write_grid_geotiff

    enc = np.clip(np.round(scene.db * DB_SCALE), 0, 65535).astype("uint16")
    return write_grid_geotiff(path, enc, scene.window, dtype="uint16", nodata=0)


def load_scene(path: str | Path) -> SceneRaster:
    with rasterio.open(path) as ds:
        enc = ds.read(1)
        t = ds.transform
    col = round((t.c - config.GRID_ORIGIN_LON) / config.PIXEL_DEG)
    row = round((config.GRID_ORIGIN_LAT - t.f) / config.PIXEL_DEG)
    window = GridWindow(col, row, enc.shape[1], enc.shape[0])
    return SceneRaster(Path(path).stem, enc.astype("float32") / DB_SCALE, window)


def to_uint8(db: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    from .georef import db_to_uint8

    return db_to_uint8(db, clip)
