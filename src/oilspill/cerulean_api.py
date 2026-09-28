"""Client for the Cerulean OGC API Features endpoint (tipg).

Politeness: <= ``config.CERULEAN_MAX_RPS`` requests/second, exponential backoff
with jitter on 429/5xx, and every response cached to disk so a re-run costs
nothing.

Verified against the live service on 2026-09-27:

* ``/conformance`` advertises ``ogcapi-features-3/1.0/conf/filter``, so CQL2 is
  supported. ``filter-lang=cql2-text`` works.
* **``hitl_cls IS NOT NULL`` is silently ignored** - it and ``IS NULL`` both
  return the same ``numberMatched`` (1,326,855 of 1,334,040). Comparison
  operators do work: ``hitl_cls > 0`` returns 7,185 and every row really does
  carry a hitl value. So ``hitl_cls > 0`` is our Tier-1 predicate. See
  DECISIONS.md.
* ``s1_scene_id`` is the full SAFE product name, e.g.
  ``S1A_IW_GRDH_1SDV_20200912T032519_20200912T032544_034320_03FD6B_6C6C``.
* ``area`` is in square metres.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from . import config

log = logging.getLogger(__name__)

#: Cerulean cls ids whose slicks we treat as positive labels, as a CQL2 list.
_POSITIVE_CLS_SQL = ", ".join(str(c) for c in config.POSITIVE_CLS)


class _RateLimiter:
    """Minimum-interval limiter, safe across threads."""

    def __init__(self, max_rps: float):
        self._min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


_limiter = _RateLimiter(config.CERULEAN_MAX_RPS)


class CeruleanClient:
    """Cached, rate-limited reader for the Cerulean API."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        use_cache: bool = True,
    ):
        self.cache_dir = Path(cache_dir or config.CERULEAN_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "oilspill/0.1 (open-source Cerulean reproduction)"},
        )

    # -- plumbing ----------------------------------------------------------

    def _cache_path(self, url: str, params: dict[str, Any]) -> Path:
        key = f"{url}?{urlencode(sorted(params.items()))}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with disk cache, rate limiting and backoff."""
        cache_path = self._cache_path(url, params)
        if self.use_cache and cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("corrupt cache entry %s; refetching", cache_path.name)
                cache_path.unlink(missing_ok=True)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            _limiter.wait()
            try:
                resp = self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                # The service resets connections on some queries; treat a
                # transport failure exactly like a 5xx and back off.
                last_exc = exc
                log.warning("transport error (attempt %d): %s", attempt + 1, exc)
            else:
                if resp.status_code == 200:
                    payload = resp.json()
                    cache_path.write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                    return payload
                if resp.status_code in (429,) or resp.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                    log.warning(
                        "HTTP %d from Cerulean (attempt %d)",
                        resp.status_code, attempt + 1,
                    )
                else:
                    resp.raise_for_status()

            backoff = min(2.0**attempt, 60.0) + random.uniform(0.0, 1.0)
            log.info("backing off %.1fs", backoff)
            time.sleep(backoff)

        raise RuntimeError(
            f"Cerulean API failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    # -- queries -----------------------------------------------------------

    def iter_slicks(
        self,
        cql2: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        datetime_range: str | None = None,
        sortby: str | None = None,
        limit: int = config.CERULEAN_PAGE_LIMIT,
        max_features: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield slick GeoJSON features, paging with limit/offset.

        ``cql2`` is CQL2-text. Remember that ``IS NULL`` / ``IS NOT NULL`` do not
        work on this deployment - use ``hitl_cls > 0`` for reviewed slicks.
        """
        offset = 0
        yielded = 0
        while True:
            page_limit = limit
            if max_features is not None:
                page_limit = min(limit, max_features - yielded)
                if page_limit <= 0:
                    return

            params: dict[str, Any] = {"limit": page_limit, "offset": offset}
            if cql2:
                params["filter-lang"] = "cql2-text"
                params["filter"] = cql2
            if bbox:
                params["bbox"] = ",".join(str(v) for v in bbox)
            if datetime_range:
                params["datetime"] = datetime_range
            if sortby:
                params["sortby"] = sortby

            payload = self.get(config.CERULEAN_SLICKS, params)
            features = payload.get("features", [])
            if not features:
                return
            for feat in features:
                yield feat
                yielded += 1
                if max_features is not None and yielded >= max_features:
                    return
            # The server may cap the page size below what we asked for (tipg does),
            # so follow its "next" link rather than comparing against page_limit.
            has_next = any(link.get("rel") == "next" for link in payload.get("links", []))
            if not has_next:
                return
            offset += len(features)

    def slicks_in_scene(self, scene_id: str) -> list[dict[str, Any]]:
        """Every slick of any class in one scene.

        Used so that an unlabelled Cerulean slick is never silently treated as
        background when building masks.
        """
        escaped = scene_id.replace("'", "''")
        return list(self.iter_slicks(cql2=f"s1_scene_id = '{escaped}'"))

    def reviewed_slicks(
        self,
        min_area_m2: float = 5.0e6,
        min_confidence: float = 0.9,
        bbox: tuple[float, float, float, float] | None = None,
        datetime_range: str | None = None,
        max_features: int | None = None,
        sortby: str | None = None,
    ) -> list[dict[str, Any]]:
        """Tier-1 slicks: human-reviewed (``hitl_cls > 0``) and a positive class."""
        cql2 = (
            f"hitl_cls IN ({_POSITIVE_CLS_SQL}) "
            f"AND area > {min_area_m2} "
            f"AND machine_confidence > {min_confidence}"
        )
        return list(
            self.iter_slicks(
                cql2=cql2, bbox=bbox, datetime_range=datetime_range,
                max_features=max_features, sortby=sortby,
            )
        )

    def unreviewed_slicks(
        self,
        min_confidence: float = config.T2_MIN_CONFIDENCE,
        bbox: tuple[float, float, float, float] | None = None,
        datetime_range: str | None = None,
        max_features: int | None = None,
    ) -> list[dict[str, Any]]:
        """Tier-2 slicks: high machine confidence, positive class.

        ``hitl_cls`` cannot be tested for NULL server-side, so the reviewed rows
        are filtered out client-side.
        """
        cql2 = (
            f"cls IN ({_POSITIVE_CLS_SQL}) "
            f"AND machine_confidence >= {min_confidence}"
        )
        feats = self.iter_slicks(
            cql2=cql2, bbox=bbox, datetime_range=datetime_range,
            max_features=max_features,
        )
        return [f for f in feats if not f["properties"].get("hitl_cls")]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CeruleanClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def effective_cls(props: dict[str, Any]) -> int:
    """Cerulean class actually used for labelling: ``hitl_cls`` if reviewed."""
    hitl = props.get("hitl_cls")
    return int(hitl) if hitl else int(props["cls"])


def label_tier(props: dict[str, Any]) -> str | None:
    """``"T1"`` (reviewed), ``"T2"`` (confident machine label), or None to discard."""
    if props.get("hitl_cls"):
        return "T1"
    if (props.get("machine_confidence") or 0.0) >= config.T2_MIN_CONFIDENCE:
        return "T2"
    return None


def model_class(props: dict[str, Any]) -> int:
    """Our 0-3 model class (or ``config.IGNORE_INDEX``) for a slick."""
    return config.CLS_TO_MODEL.get(effective_cls(props), config.IGNORE_INDEX)
