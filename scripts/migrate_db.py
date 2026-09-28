"""Copy all detections (and scene overlay images) from one database to another.

Typical use: move the local SQLite results into the cloud Postgres (Neon) once.

    python scripts/migrate_db.py --src sqlite:///C:/oilspill-data/oilspill.db --dst "postgresql://..." \
        --overlays C:/oilspill-data/app/scenes

Idempotent: rows that already exist in the destination (same primary key) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.database import Detection, OverlayImage, Run, Scene, make_engine  # noqa: E402


def copy_table(src: Session, dst: Session, model) -> int:
    have = {r for (r,) in dst.execute(select(model.id))}
    n = 0
    for row in src.scalars(select(model)):
        if row.id in have:
            continue
        data = {c.key: getattr(row, c.key) for c in model.__table__.columns}
        dst.add(model(**data))
        n += 1
    dst.flush()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--overlays", default=None, help="folder with <scene key>/quicklook.png, prob.png, overlay.json")
    a = ap.parse_args()

    src_eng, dst_eng = make_engine(a.src), make_engine(a.dst)
    with Session(src_eng) as src, Session(dst_eng) as dst:
        counts = {m.__tablename__: copy_table(src, dst, m) for m in (Scene, Run, Detection)}
        n_img = 0
        # overlays already stored in the source database
        for row in src.scalars(select(OverlayImage)):
            if dst.get(OverlayImage, (row.key, row.kind)) is None:
                dst.add(OverlayImage(key=row.key, kind=row.kind, png=row.png, meta=row.meta,
                                     created_at=row.created_at))
                n_img += 1
        # overlays that only exist as files (older local runs)
        if a.overlays:
            from PIL import Image  # noqa: F401  (pipeline dependency; keeps the check honest)

            for d in sorted(Path(a.overlays).iterdir()):
                meta_f = d / "overlay.json"
                if not meta_f.exists():
                    continue
                meta = json.loads(meta_f.read_text())
                for kind in ("quicklook", "prob"):
                    f = d / f"{kind}.png"
                    if f.exists() and dst.get(OverlayImage, (d.name, kind)) is None:
                        dst.add(OverlayImage(key=d.name, kind=kind, png=_shrink(f), meta=meta,
                                             created_at=datetime.now(UTC)))
                        n_img += 1
        dst.commit()
    # Postgres sequences must continue after the copied ids.
    if dst_eng.dialect.name == "postgresql":
        with dst_eng.begin() as c:
            for t in ("scene", "run", "detection"):
                c.exec_driver_sql(
                    f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), COALESCE((SELECT MAX(id) FROM {t}), 1))")
    print(json.dumps({"copied_rows": counts, "overlay_images": n_img}))
    return 0


def _shrink(path: Path) -> bytes:
    """Re-encode older full-size PNGs at the size the pipeline now stores."""
    import io

    from PIL import Image

    img = Image.open(path)
    if max(img.size) > 1600:
        f = 1600 / max(img.size)
        img = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
