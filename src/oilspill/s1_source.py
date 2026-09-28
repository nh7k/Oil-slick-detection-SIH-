"""Sentinel-1 GRD imagery from Microsoft Planetary Computer.

Facts this module is built on (verified 2026-09-27):

* STAC root ``https://planetarycomputer.microsoft.com/api/stac/v1``, collection
  ``sentinel-1-grd``, VV band in asset ``vv``.
* Signing is anonymous via ``planetary_computer.sign``; SAS tokens last about
  24 h, so a 403 means "re-sign and retry", not "gone".
* Assets are COGs of **raw DN** georeferenced **only by GCPs**. The STAC
  ``proj:transform`` is a bbox approximation - see :mod:`oilspill.georef`.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import planetary_computer
import pystac_client
import rasterio
from rasterio.errors import RasterioIOError

from . import config

log = logging.getLogger(__name__)

#: GDAL settings that make remote COG reads sane: no sidecar probing, a
#: generous block cache, and retries on transient HTTP failures.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    "GDAL_CACHEMAX": "512",
}


def apply_gdal_env() -> None:
    """Set the GDAL tuning variables in this process, without clobbering overrides."""
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Scene identifiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneKey:
    """The parts of a SAFE product name that identify an acquisition.

    ``S1A_IW_GRDH_1SDV_20200912T032519_20200912T032544_034320_03FD6B_6C6C``
      platform=S1A mode=IW product=GRDH polarisation=1SDV
      start=20200912T032519 stop=20200912T032544
      absolute_orbit=034320 mission_datatake=03FD6B unique_id=6C6C
    """

    platform: str
    mode: str
    product: str
    polarisation: str
    start: str
    stop: str

    @property
    def prefix(self) -> str:
        """The part that is stable across id spellings."""
        return "_".join(
            (self.platform, self.mode, self.product, self.polarisation,
             self.start, self.stop)
        )


def parse_scene_id(scene_id: str) -> SceneKey | None:
    """Parse a SAFE product name. Returns None if it does not look like one."""
    parts = scene_id.strip().rstrip("/").removesuffix(".SAFE").split("_")
    # Sentinel-1 product names use a doubled separator for the mode field in
    # some spellings (e.g. S1A_IW_GRDH vs S1A_IW__GRDH), producing an empty part.
    parts = [p for p in parts if p]
    if len(parts) < 6 or not parts[0].startswith("S1"):
        return None
    try:
        return SceneKey(
            platform=parts[0], mode=parts[1], product=parts[2],
            polarisation=parts[3], start=parts[4], stop=parts[5],
        )
    except IndexError:
        return None


# ---------------------------------------------------------------------------
# STAC
# ---------------------------------------------------------------------------

_catalog: pystac_client.Client | None = None


def catalog() -> pystac_client.Client:
    """Open (and memoise) the Planetary Computer STAC catalog."""
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(config.PC_STAC)
    return _catalog


def _stac_cache_path(scene_id: str) -> Path:
    config.STAC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.STAC_CACHE_DIR / f"{scene_id}.json"


def find_item(
    scene_id: str,
    intersects: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    """Locate the PC ``sentinel-1-grd`` item for a Cerulean ``s1_scene_id``.

    Strategy, in order:

    1. Exact item id lookup.
    2. Exact lookup of the id with its trailing product-unique-id token removed.
       **This is the case that actually fires.** Measured on 2026-09-27, all
       three Phase-1 scenes showed Cerulean carrying one more token than PC::

           Cerulean  S1A_..._050360_061036_7DC5
           PC        S1A_..._050360_061036

       so stripping the final ``_XXXX`` turns a 100-item search into one GET.
    3. A datetime +/-2 min search (optionally constrained by ``intersects``),
       matched on the product-id prefix, as a backstop.

    Returns the item as a dict, or None.
    """
    cache = _stac_cache_path(scene_id)
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache.unlink(missing_ok=True)

    cat = catalog()
    item_dict: dict[str, Any] | None = None

    try:
        collection = cat.get_collection(config.PC_COLLECTION)
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot open collection %s: %s", config.PC_COLLECTION, exc)
        collection = None

    # 1/2. exact id, then id minus the trailing unique-id token
    candidates_ids = [scene_id]
    head, sep, tail = scene_id.rpartition("_")
    if sep and len(tail) == 4:
        candidates_ids.append(head)

    if collection is not None:
        for cand_id in candidates_ids:
            try:
                item = collection.get_item(cand_id)
            except Exception as exc:  # noqa: BLE001 - STAC raises various errors
                log.debug("id lookup failed for %s: %s", cand_id, exc)
                continue
            if item is not None:
                item_dict = item.to_dict()
                how = "exact" if cand_id == scene_id else "id_minus_suffix"
                log.info("matched %s -> %s (%s)", scene_id, item.id, how)
                break

    # 2. datetime window + prefix match
    if item_dict is None:
        key = parse_scene_id(scene_id)
        if key is None:
            log.warning("cannot parse scene id %s", scene_id)
            return None
        start = _iso(key.start)
        lo = _shift_iso(start, -120)
        hi = _shift_iso(start, +120)
        search_kwargs: dict[str, Any] = {
            "collections": [config.PC_COLLECTION],
            "datetime": f"{lo}/{hi}",
            "limit": 100,
        }
        if intersects is not None:
            search_kwargs["intersects"] = intersects
        try:
            candidates = list(cat.search(**search_kwargs).items())
        except Exception as exc:  # noqa: BLE001
            log.warning("STAC search failed for %s: %s", scene_id, exc)
            candidates = []

        log.info(
            "exact id missed for %s; %d candidates in +/-2 min window",
            scene_id, len(candidates),
        )
        for cand in candidates:
            cand_key = parse_scene_id(cand.id)
            if cand_key is not None and cand_key.prefix == key.prefix:
                item_dict = cand.to_dict()
                log.info("matched %s -> %s by product-id prefix", scene_id, cand.id)
                break
        else:
            for cand in candidates:
                log.info("   candidate id: %s", cand.id)

    if item_dict is not None:
        cache.write_text(json.dumps(item_dict), encoding="utf-8")
    return item_dict


def _iso(compact: str) -> str:
    """``20200912T032519`` -> ``2020-09-12T03:25:19Z``."""
    d, t = compact.split("T")
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z"


def _shift_iso(iso: str, seconds: int) -> str:
    from datetime import datetime, timedelta

    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (dt + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def sign_asset(item: dict[str, Any], asset: str = config.PC_ASSET) -> str:
    """Return a freshly signed href for one asset of an item."""
    assets = item.get("assets", {})
    if asset not in assets:
        raise KeyError(
            f"item {item.get('id')} has no asset {asset!r}; "
            f"available: {sorted(assets)}"
        )
    return planetary_computer.sign(assets[asset]["href"])


def open_asset(
    item: dict[str, Any],
    asset: str = config.PC_ASSET,
    max_retries: int = 4,
) -> rasterio.DatasetReader:
    """Open a signed asset, re-signing on 403 (expired SAS token).

    The caller owns the returned dataset and must close it.
    """
    apply_gdal_env()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        href = sign_asset(item, asset)
        try:
            return rasterio.open(href)
        except RasterioIOError as exc:
            last_exc = exc
            msg = str(exc)
            transient = "403" in msg or "404" in msg or "timed out" in msg.lower()
            log.warning(
                "open failed (attempt %d/%d, transient=%s): %s",
                attempt + 1, max_retries, transient, msg.splitlines()[0],
            )
            if not transient:
                raise
            time.sleep(min(2.0**attempt, 20.0) + random.uniform(0, 1))
    raise RuntimeError(f"could not open asset {asset}: {last_exc}") from last_exc


def search_scenes(
    bbox: tuple[float, float, float, float],
    datetime_range: str,
    mode: str = "IW",
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """List ``sentinel-1-grd`` items over a bbox and time range.

    Filters to the requested acquisition mode and to items that actually carry a
    ``vv`` asset (so SH/single-pol products are dropped).
    """
    search = catalog().search(
        collections=[config.PC_COLLECTION],
        bbox=list(bbox),
        datetime=datetime_range,
        limit=500,
    )
    out: list[dict[str, Any]] = []
    for item in search.items():
        key = parse_scene_id(item.id)
        if key is not None and key.mode != mode:
            continue
        d = item.to_dict()
        if config.PC_ASSET not in d.get("assets", {}):
            continue
        out.append(d)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def item_footprint(item: dict[str, Any]) -> dict[str, Any]:
    """The item's geometry, as a GeoJSON geometry dict."""
    geom = item.get("geometry")
    if geom is None:
        raise ValueError(f"item {item.get('id')} has no geometry")
    return geom
