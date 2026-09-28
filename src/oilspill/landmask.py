"""Natural Earth 10 m land polygons, read without OGR.

``geopandas.read_file`` needs pyogrio or fiona, and on this machine Smart App
Control blocks pyogrio's native extension (DECISIONS.md D0.6). Natural Earth
ships plain shapefiles, so we read them with pyshp - pure Python, no native code
and no GDAL/OGR involved at all.

Natural Earth is public domain.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from . import config

log = logging.getLogger(__name__)

_LAND_ZIP = "ne_10m_land.zip"
_cache: tuple[list[BaseGeometry], STRtree] | None = None


def land_zip_path() -> Path:
    return config.CACHE_DIR / _LAND_ZIP


def download_land(force: bool = False) -> Path:
    """Fetch ne_10m_land.zip into the cache. Idempotent."""
    path = land_zip_path()
    if path.exists() and not force and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import httpx

    log.info("downloading Natural Earth 10m land -> %s", path)
    with httpx.Client(timeout=300, follow_redirects=True) as c:
        r = c.get(config.NATURAL_EARTH_LAND_URL)
        r.raise_for_status()
        path.write_bytes(r.content)
    log.info("downloaded %.1f MB", path.stat().st_size / 1e6)
    return path


def _read_shapefile_from_zip(zip_path: Path) -> list[BaseGeometry]:
    """All polygon geometries from the single shapefile inside a Natural Earth zip."""
    import shapefile  # pyshp

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        shp = next((n for n in names if n.lower().endswith(".shp")), None)
        if shp is None:
            raise ValueError(f"no .shp inside {zip_path}")
        stem = shp[: -len(".shp")]
        dbf = next((n for n in names if n.lower() == f"{stem.lower()}.dbf"), None)
        shx = next((n for n in names if n.lower() == f"{stem.lower()}.shx"), None)

        kwargs = {"shp": io.BytesIO(zf.read(shp))}
        if dbf:
            kwargs["dbf"] = io.BytesIO(zf.read(dbf))
        if shx:
            kwargs["shx"] = io.BytesIO(zf.read(shx))

        geoms: list[BaseGeometry] = []
        with shapefile.Reader(**kwargs) as reader:
            for sh in reader.iterShapes():
                try:
                    g = shape(sh.__geo_interface__)
                except Exception as exc:  # noqa: BLE001
                    log.debug("skipping unreadable shape: %s", exc)
                    continue
                if g.is_empty:
                    continue
                if not g.is_valid:
                    from shapely import make_valid

                    g = make_valid(g)
                    if g.is_empty:
                        continue
                geoms.append(g)
    log.info("read %d land polygons from %s", len(geoms), zip_path.name)
    return geoms


def load_land() -> tuple[list[BaseGeometry], STRtree]:
    """Load (and memoise) land polygons plus a spatial index."""
    global _cache
    if _cache is None:
        geoms = _read_shapefile_from_zip(download_land())
        _cache = (geoms, STRtree(geoms))
    return _cache


def land_in_bounds(bounds: tuple[float, float, float, float]) -> list[BaseGeometry]:
    """Land polygons intersecting a lon/lat bbox."""
    geoms, tree = load_land()
    query = box(*bounds)
    return [geoms[i] for i in tree.query(query) if geoms[i].intersects(query)]


def land_mask(window, bounds_only: bool = False) -> np.ndarray:
    """Rasterise Natural Earth land onto a :class:`~oilspill.georef.GridWindow`."""
    from rasterio.features import rasterize

    geoms = land_in_bounds(window.bounds)
    mask = np.zeros(window.shape, dtype=bool)
    if not geoms:
        return mask
    burned = rasterize(
        [(g, 1) for g in geoms],
        out_shape=window.shape,
        transform=window.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return burned.astype(bool)


def buffered_land(
    bounds: tuple[float, float, float, float],
    buffer_m: float = config.POSTPROCESS.land_buffer_m,
) -> BaseGeometry | None:
    """Union of land in ``bounds``, buffered seaward by ``buffer_m``.

    Used to drop detections that are really coastline artefacts. Cerulean applies
    a 1 km land buffer for the same reason [SRC].
    """
    from shapely.ops import unary_union

    from .crsutil import buffer_metres

    geoms = land_in_bounds(bounds)
    if not geoms:
        return None
    # Clip to the (padded) area of interest first: buffering whole continents in
    # one UTM zone fails with "point outside of projection domain".
    pad = 0.5
    clip = box(bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    union = unary_union([g.intersection(clip) for g in geoms])
    if union.is_empty:
        return None
    try:
        return buffer_metres(union, buffer_m)
    except Exception as exc:  # noqa: BLE001
        log.warning("metric land buffer failed (%s); falling back to degrees", exc)
        lat_mid = 0.5 * (bounds[1] + bounds[3])
        from .crsutil import metres_to_degrees

        dlon, _ = metres_to_degrees(buffer_m, lat_mid)
        return union.buffer(dlon)
