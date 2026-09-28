"""Detection store. SQLite locally; any SQLAlchemy URL (e.g. Postgres) in production.

Geometry is stored as GeoJSON text plus bbox columns, which keeps SQLite free of
spatial extensions while still allowing indexed bbox filtering.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text,
                        create_engine, event, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from oilspill import config


class Base(DeclarativeBase):
    pass


class Scene(Base):
    __tablename__ = "scene"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scene_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="planetary_computer")
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    orbit_direction: Mapped[str | None] = mapped_column(String(16))
    minx: Mapped[float] = mapped_column(Float)
    miny: Mapped[float] = mapped_column(Float)
    maxx: Mapped[float] = mapped_column(Float)
    maxy: Mapped[float] = mapped_column(Float)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    runs: Mapped[list[Run]] = relationship(back_populates="scene")


class Run(Base):
    __tablename__ = "run"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scene_fk: Mapped[int] = mapped_column(ForeignKey("scene.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict] = mapped_column(JSON)
    timings: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scene: Mapped[Scene] = relationship(back_populates="runs")


class Detection(Base):
    __tablename__ = "detection"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_fk: Mapped[int] = mapped_column(ForeignKey("run.id"), index=True)
    scene_fk: Mapped[int] = mapped_column(ForeignKey("scene.id"), index=True)
    cls: Mapped[int] = mapped_column(Integer, index=True)
    confidence: Mapped[float] = mapped_column(Float, index=True)
    confidence_mean: Mapped[float] = mapped_column(Float)
    confidence_max: Mapped[float] = mapped_column(Float)
    area_km2: Mapped[float] = mapped_column(Float, index=True)
    length_km: Mapped[float] = mapped_column(Float)
    perimeter_km: Mapped[float] = mapped_column(Float)
    polsby_popper: Mapped[float] = mapped_column(Float)
    centroid_lon: Mapped[float] = mapped_column(Float)
    centroid_lat: Mapped[float] = mapped_column(Float)
    minx: Mapped[float] = mapped_column(Float, index=True)
    miny: Mapped[float] = mapped_column(Float, index=True)
    maxx: Mapped[float] = mapped_column(Float, index=True)
    maxy: Mapped[float] = mapped_column(Float, index=True)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    geometry: Mapped[str] = mapped_column(Text)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class OverlayImage(Base):
    """Scene quicklook / probability PNGs (Web Mercator) with their lon/lat bounds.

    Stored in the database (not on disk) so they survive on hosts with ephemeral disks
    and are shared between the processing job and the web API.
    """
    __tablename__ = "overlay_image"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)  # quicklook | prob
    png: Mapped[bytes] = mapped_column(LargeBinary)
    meta: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def safe_key(scene_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in scene_id)[:120]


def normalize_url(url: str) -> str:
    """Accept the plain postgres:// URLs that Neon/Render print, using the psycopg 3 driver."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def save_overlays(session: Session, key: str, images: dict[str, bytes], meta: dict) -> None:
    for kind, png in images.items():
        row = session.get(OverlayImage, (key, kind))
        if row is None:
            row = OverlayImage(key=key, kind=kind)
            session.add(row)
        row.png, row.meta, row.created_at = png, meta, now()


def load_overlay(session: Session, key: str, kind: str) -> OverlayImage | None:
    return session.get(OverlayImage, (key, kind))


def make_engine(url: str | None = None):
    url = normalize_url(url or config.DATABASE_URL)
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    eng = create_engine(url, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _pragma(conn, _):  # noqa: ANN001
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    Base.metadata.create_all(eng)
    return eng


def now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def detection_feature(d: Detection, scene: Scene, run: Run) -> dict:
    return {
        "type": "Feature",
        "id": d.id,
        "geometry": json.loads(d.geometry),
        "properties": {
            "id": d.id, "cls": d.cls, "cls_name": config.MODEL_CLASSES.get(d.cls, str(d.cls)),
            "confidence": d.confidence, "confidence_mean": d.confidence_mean,
            "confidence_max": d.confidence_max, "area_km2": d.area_km2, "length_km": d.length_km,
            "perimeter_km": d.perimeter_km, "polsby_popper": d.polsby_popper,
            "centroid_lon": d.centroid_lon, "centroid_lat": d.centroid_lat,
            "bbox": [d.minx, d.miny, d.maxx, d.maxy], "scene_id": scene.scene_id,
            "acquired_at": _iso(d.acquired_at), "model_version": run.model_version, "run_id": run.id,
        },
    }


def store_run(session: Session, scene_meta: dict, model_version: str, params: dict, timings: dict,
              features: list[dict]) -> Run:
    """Insert/replace one scene's detections atomically (idempotent per scene+model)."""
    from shapely.geometry import shape

    sc = session.scalar(select(Scene).where(Scene.scene_id == scene_meta["scene_id"]))
    b = scene_meta["bounds"]
    if sc is None:
        sc = Scene(scene_id=scene_meta["scene_id"])
        session.add(sc)
    sc.source = scene_meta.get("source", "planetary_computer")
    sc.acquired_at = scene_meta.get("acquired_at")
    sc.orbit_direction = scene_meta.get("orbit_direction")
    sc.minx, sc.miny, sc.maxx, sc.maxy = b
    sc.processed_at = now()
    session.flush()

    # replace earlier detections of the same model version for this scene
    for old in session.scalars(select(Run).where(Run.scene_fk == sc.id, Run.model_version == model_version)):
        session.query(Detection).filter(Detection.run_fk == old.id).delete()
        session.delete(old)
    session.flush()

    run = Run(scene_fk=sc.id, model_version=model_version, params=params, timings=timings,
              status="done", started_at=scene_meta.get("started_at", now()), finished_at=now())
    session.add(run)
    session.flush()
    for f in features:
        p = f["properties"]
        g = shape(f["geometry"])
        minx, miny, maxx, maxy = g.bounds
        session.add(Detection(
            run_fk=run.id, scene_fk=sc.id, cls=p["cls"], confidence=p["confidence"],
            confidence_mean=p["confidence_mean"], confidence_max=p["confidence_max"],
            area_km2=p["area_km2"], length_km=p["length_km"], perimeter_km=p["perimeter_km"],
            polsby_popper=p["polsby_popper"], centroid_lon=p["centroid_lon"],
            centroid_lat=p["centroid_lat"], minx=minx, miny=miny, maxx=maxx, maxy=maxy,
            acquired_at=scene_meta.get("acquired_at"), geometry=json.dumps(f["geometry"]),
            extra={"class_prob_share": p.get("class_prob_share"), "n_pixels": p.get("n_pixels")},
        ))
    session.commit()
    return run


SessionFactory = sessionmaker
