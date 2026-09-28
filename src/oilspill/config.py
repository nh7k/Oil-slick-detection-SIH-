"""Single source of truth for grid constants, class maps, thresholds and paths.

Nothing in this package may hardcode a grid constant or threshold; import it
from here. Values tagged [SRC] are reproduced from Cerulean's published
inference behaviour; [DERIVED] values are computed from those; [OURS] are our
own choices and are recorded in DECISIONS.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_dir(var: str, default: Path) -> Path:
    raw = os.getenv(var, "").strip()
    return Path(raw).expanduser() if raw else default


#: Root for bulky derived artifacts. Defaults to <repo>/data but should point
#: outside any cloud-synced folder (see .env.example).
DATA_DIR = _env_dir("OILSPILL_DATA_DIR", REPO_ROOT / "data")
MODELS_DIR = _env_dir("OILSPILL_MODELS_DIR", REPO_ROOT / "models")
REPORTS_DIR = _env_dir("OILSPILL_REPORTS_DIR", REPO_ROOT / "reports")

CACHE_DIR = DATA_DIR / "cache"
CERULEAN_CACHE_DIR = CACHE_DIR / "cerulean"
STAC_CACHE_DIR = CACHE_DIR / "stac"
SCENE_CACHE_DIR = CACHE_DIR / "scenes"
TILES_DIR = DATA_DIR / "tiles"
MASKS_DIR = DATA_DIR / "masks"
STATUS_DIR = DATA_DIR / "status"
SPLITS_DIR = REPO_ROOT / "data" / "splits"  # small, tracked in git
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"


def ensure_dirs() -> None:
    """Create every directory this package writes to. Idempotent."""
    for d in (
        DATA_DIR, MODELS_DIR, REPORTS_DIR, CACHE_DIR, CERULEAN_CACHE_DIR,
        STAC_CACHE_DIR, SCENE_CACHE_DIR, TILES_DIR, MASKS_DIR, STATUS_DIR,
        SPLITS_DIR, CHECKPOINTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Grid - morecantile WorldCRS84Quad, zoom 9, tile scale 2  [SRC]
# ---------------------------------------------------------------------------

GRID_ZOOM = 9
GRID_TILE_PX = 512
#: EPSG:4326 degrees per pixel: 180 / 2**9 / 512  [DERIVED from SRC]
PIXEL_DEG = 180.0 / (2**GRID_ZOOM) / GRID_TILE_PX  # 0.0006866455078125
#: Grid origin. WorldCRS84Quad is anchored at the antimeridian / north pole.
GRID_ORIGIN_LON = -180.0
GRID_ORIGIN_LAT = 90.0
#: Approximate ground sample distance at the equator, metres. Cerulean's docs
#: say ~80 m and tile_width_m 40844; PIXEL_DEG * 111320 gives ~76.4 m. The
#: discrepancy is theirs; we use the exact degree value throughout.
PIXEL_M_EQUATOR = PIXEL_DEG * 111_320.0

TILE_PX = 512
#: Sliding-window stride for inference; overlap = TILE_PX - INFER_STRIDE = 128.
INFER_STRIDE = 384

NODATA = 0
#: EPSG:4326 is lon/lat; every raster this package writes uses this CRS.
CRS_EPSG = 4326
CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Radiometry
# ---------------------------------------------------------------------------

def dn_to_db(dn):
    """Raw Sentinel-1 GRD digital number -> pseudo-dB.

    ``10*log10(DN**2 + 1)``. The +1 keeps DN==0 (nodata / outside swath) finite
    at exactly 0.0 dB instead of -inf. This is NOT calibrated sigma0 - Cerulean
    deliberately skips calibration [SRC].
    """
    import numpy as np

    dn = np.asarray(dn, dtype=np.float32)
    return 10.0 * np.log10(dn * dn + 1.0)


#: dB clip range used to scale to uint8. Computed once in Phase 2 from the
#: 1st/99th percentiles of training tiles and then FROZEN. Until that
#: measurement exists this is None and callers must fail loudly rather than
#: guess - see DECISIONS.md.
DB_CLIP: tuple[float, float] | None = None

#: Dataset normalisation (mean/std of uint8/255 tiles). Measured in Phase 2 and
#: written to model_card.json. None until measured.
NORM_MEAN: float | None = None
NORM_STD: float | None = None


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

#: Our model's 4 output classes  [SRC]
MODEL_CLASSES = {0: "BACKGROUND", 1: "INFRA", 2: "NATURAL", 3: "VESSEL"}
N_CLASSES = 4
OIL_CLASSES = (1, 2, 3)
IGNORE_INDEX = 255

#: Cerulean's public.cls ids  [DOC]
CERULEAN_CLS = {
    1: "NOT_OIL", 2: "ANTHRO", 3: "NATURAL", 4: "INFRA", 5: "VESSEL",
    6: "OLD_VESSEL", 7: "REC_VESSEL", 8: "COIN_VESSEL", 9: "AMBIGUOUS",
    10: "LAND", 11: "SEA_ICE", 12: "ARTEFACT",
}

#: Cerulean cls id -> our model class. 255 == ignore (excluded from loss).
CLS_TO_MODEL: dict[int, int] = {
    4: 1,                       # INFRA
    3: 2,                       # NATURAL
    5: 3, 6: 3, 7: 3, 8: 3,     # VESSEL family
    2: IGNORE_INDEX,            # ANTHRO - too vague to supervise
    9: IGNORE_INDEX,            # AMBIGUOUS
    1: 0, 10: 0, 11: 0, 12: 0,  # NOT_OIL / LAND / SEA_ICE / ARTEFACT
}
#: Cerulean cls ids that become background but are informative negatives.
HARD_NEGATIVE_CLS = (1, 10, 11, 12)
#: Cerulean cls ids that map to a positive model class.
POSITIVE_CLS = tuple(sorted(k for k, v in CLS_TO_MODEL.items() if v in OIL_CLASSES))


# ---------------------------------------------------------------------------
# Post-processing thresholds  [SRC unless noted]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostprocessConfig:
    #: Hysteresis low threshold - islands of P(oil) above this are candidates.
    p_low: float = 1e-4
    #: Hysteresis seed threshold - an island survives only if some pixel >= this.
    p_seed: float = 0.9
    #: Polygon trim threshold.
    p_trim: float = 0.5
    #: Drop polygons >edge_frac within this many degrees of the scene edge.
    edge_deg: float = 0.005
    edge_frac: float = 0.5
    #: Polygon NMS IoU.
    nms_iou: float = 0.2
    #: Minimum polygon area, m2. [OURS] - Cerulean's value is not published.
    min_area_m2: float = 100_000.0
    #: Simplify tolerance, in pixels.
    simplify_px: float = 0.5
    #: Fragments closer than this many pixels (~76 m each) form one slick. [OURS]
    group_px: int = 6
    #: Pinholes smaller than this (pixels) are filled before polygonising. [OURS]
    max_hole_px: int = 16
    #: Land mask buffer, metres.
    land_buffer_m: float = 1_000.0


POSTPROCESS = PostprocessConfig()

#: Instance-matching IoU for evaluation. [OURS] - documented in reports.
EVAL_INSTANCE_IOU = 0.1


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

#: Decimation factor for reads of the native ~10 m GRD onto our ~76 m grid.
#: 8 takes 10 m -> 80 m, slightly coarser than the target, so `average`
#: resampling in the warp does the final small step.
READ_DECIMATION = 8

#: Tier-1 (human-reviewed) label weight in the training loss.
T1_LOSS_WEIGHT = 3.0
#: Minimum machine_confidence for a Tier-2 (unreviewed) label.
T2_MIN_CONFIDENCE = 0.9
#: Discard tiles with more than this fraction of nodata.
MAX_NODATA_FRAC = 0.5
#: Empty-ocean tiles are sampled at this ratio to positive tiles.
NEGATIVE_RATIO = 1.0
#: Empty tiles must be at least this far from any labelled polygon, metres.
NEGATIVE_EXCLUSION_M = 5_000.0

#: Split grouping cell size, degrees.
SPLIT_CELL_DEG = 2.0
TEST_CELL_FRAC = 0.15
VAL_CELL_FRAC = 0.15
SPLIT_SEED = 20260927

#: Trailing months reserved as the temporal test set.
TEMPORAL_TEST_MONTHS = 3
DATASET_START_DATE = "2023-01-01"


@dataclass(frozen=True)
class AOI:
    name: str
    #: (min_lon, min_lat, max_lon, max_lat)
    bbox: tuple[float, float, float, float]


#: Diverse ocean AOIs for dataset construction. [OURS]
AOIS: tuple[AOI, ...] = (
    AOI("arabian_sea_west_india", (68.0, 6.0, 78.0, 20.0)),
    AOI("persian_gulf", (47.5, 23.5, 57.0, 30.5)),
    AOI("gulf_of_mexico", (-97.5, 18.0, -81.0, 30.5)),
    AOI("east_mediterranean", (25.0, 30.5, 36.5, 37.5)),
    AOI("gulf_of_guinea", (-2.0, -2.0, 9.0, 7.0)),
    AOI("malacca_south_china_sea", (99.0, -1.0, 115.0, 12.0)),
    AOI("north_sea", (-2.0, 51.0, 9.0, 60.0)),
)

#: Default AOI for the Phase 12 batch job.
BATCH_AOI = AOIS[0]

#: Indian seas for the live/batch job (bboxes; land-only scenes are skipped). [OURS]
INDIA_AOIS: tuple[AOI, ...] = (
    AOI("india_arabian_sea", (66.0, 5.0, 78.0, 24.5)),
    AOI("india_bay_of_bengal_andaman", (78.0, 4.0, 95.0, 23.0)),
)
#: Skip scenes whose footprint is less than this fraction sea.
MIN_SEA_FRACTION = 0.05


# ---------------------------------------------------------------------------
# External services
# ---------------------------------------------------------------------------

CERULEAN_BASE = "https://api.cerulean.skytruth.org"
CERULEAN_SLICKS = f"{CERULEAN_BASE}/collections/public.slick_plus/items"
CERULEAN_SCENES = f"{CERULEAN_BASE}/collections/public.sentinel1_grd/items"
#: Politeness budget for the Cerulean API.
CERULEAN_MAX_RPS = 2.0
CERULEAN_PAGE_LIMIT = 500

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_COLLECTION = "sentinel-1-grd"
PC_ASSET = "vv"

NATURAL_EARTH_LAND_URL = (
    "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
)


# ---------------------------------------------------------------------------
# Database / API
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./oilspill.db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


# ---------------------------------------------------------------------------
# Attribution - must appear in README and the UI footer.
# ---------------------------------------------------------------------------

ATTRIBUTION_CERULEAN = "SkyTruth Cerulean (cerulean.skytruth.org). 2026. CC BY-SA 4.0."
ATTRIBUTION_COPERNICUS = "Contains modified Copernicus Sentinel data 2026."
