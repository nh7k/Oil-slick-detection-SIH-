"""End-to-end plumbing tests: real Phase-1 Sentinel-1 raster -> ONNX -> polygons -> DB -> API.

The ONNX model here is a TEST FIXTURE ONLY: a hand-built graph that turns
"darker than the scene mean" into oil logits. It exists to exercise the
pipeline wiring without a trained network, and is written to a temp dir -
never to models/, so the API can never serve it.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE1_TIF = ROOT / "reports" / "phase1" / "4823849_sar.tif"


def make_dark_spot_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    # logits: bg = 4*x ; infra = -100 ; natural = -100 ; vessel = -4*x  (x is the normalised input)
    w = np.array([4.0, 0.0, 0.0, -4.0], np.float32).reshape(4, 1, 1, 1)
    b = np.array([0.0, -100.0, -100.0, 0.0], np.float32)
    node = helper.make_node("Conv", ["input", "W", "B"], ["logits"])
    graph = helper.make_graph(
        [node], "dark_spot_fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 512, 512])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 4, 512, 512])],
        [helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.ravel()),
         helper.make_tensor("B", TensorProto.FLOAT, b.shape, b)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture()
def fixture_model(tmp_path):
    mdir = tmp_path / "models"
    mdir.mkdir()
    make_dark_spot_onnx(mdir / "model.onnx")
    card = {"model_version": "TEST-FIXTURE", "architecture": "dark-spot conv fixture (tests only)",
            "db_clip": [28.5, 45.0], "norm_mean": 0.55, "norm_std": 0.2,
            "classes": {"1": "INFRA", "2": "NATURAL", "3": "VESSEL"},
            "thresholds": {"p_seed": 0.9, "p_trim": 0.5}}
    (mdir / "model_card.json").write_text(json.dumps(card))
    return mdir


@pytest.fixture()
def client(tmp_path, fixture_model, monkeypatch):
    monkeypatch.setenv("OILSPILL_MODELS_DIR", str(fixture_model))
    monkeypatch.setenv("OILSPILL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    sys.path.insert(0, str(ROOT / "backend"))
    for m in [m for m in sys.modules if m.startswith(("oilspill", "app"))]:
        del sys.modules[m]
    importlib.import_module("oilspill.config")
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c, main


def test_postprocess_on_real_scene(fixture_model):
    from oilspill.infer import ModelCard, OnnxPredictor, predict_scene
    from oilspill.postprocess import extract_slicks
    from oilspill.scene import prepare_geotiff

    scene = prepare_geotiff(PHASE1_TIF)
    card = ModelCard.load(fixture_model / "model_card.json")
    probs, t = predict_scene(scene, OnnxPredictor(fixture_model / "model.onnx"), card)
    assert probs.shape == (4, *scene.db.shape)
    assert np.allclose(probs.sum(0), 1.0, atol=1e-4)
    feats = extract_slicks(probs, scene.window, scene.valid)
    assert feats, "dark-spot fixture should find the known dark vessel slick"
    for f in feats:
        p = f["properties"]
        assert 0 <= p["confidence"] <= p["confidence_max"] <= 1
        assert p["area_km2"] > 0
        minx, miny, maxx, maxy = p["bbox"]
        b = scene.bounds
        assert b[0] - 1e-6 <= minx and maxx <= b[2] + 1e-6 and b[1] - 1e-6 <= miny and maxy <= b[3] + 1e-6


def test_api_flow(client):
    c, _ = client
    h = c.get("/api/health").json()
    assert h["model_loaded"] and h["model_version"] == "TEST-FIXTURE"
    assert c.get("/api/model-info").json()["loaded"]

    # invalid inputs
    assert c.post("/api/predict-geotiff", files={"file": ("x.txt", b"hello")}).status_code == 415
    assert c.post("/api/predict-geotiff", files={"file": ("x.tif", b"notatiff")}).status_code == 415
    assert c.get("/api/detections", params={"bbox": "1,2,3"}).status_code == 422
    assert c.get("/api/detections/999999").status_code == 404
    assert c.get("/api/jobs/nope").status_code == 404

    # real GeoTIFF upload
    r = c.post("/api/predict-geotiff", files={"file": (PHASE1_TIF.name, PHASE1_TIF.read_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    fc = r.json()["detections"]
    assert fc["type"] == "FeatureCollection" and fc["features"]
    g = fc["features"][0]["geometry"]
    assert g["type"] == "MultiPolygon"
    lon, lat = g["coordinates"][0][0][0]
    assert 80 < lon < 95 and 15 < lat < 25  # Bay of Bengal scene: lon/lat order

    # empty DB -> empty collection, never fake data
    d = c.get("/api/detections", params={"min_confidence": 0.99}).json()
    assert d["type"] == "FeatureCollection"


def test_store_and_query(client):
    c, main = client
    from datetime import UTC, datetime

    from app.database import store_run
    from shapely.geometry import MultiPolygon, box, mapping

    geom = mapping(MultiPolygon([box(75.4, 10.2, 75.5, 10.3)]))
    feat = {"type": "Feature", "geometry": geom, "properties": {
        "cls": 3, "confidence": 0.93, "confidence_mean": 0.9, "confidence_max": 0.99, "area_km2": 120.0,
        "length_km": 11.0, "perimeter_km": 44.0, "polsby_popper": 0.78, "centroid_lon": 75.45,
        "centroid_lat": 10.25, "class_prob_share": [0, 0, 1], "n_pixels": 10}}
    meta = {"scene_id": "S1_TEST_SCENE", "bounds": [74, 9, 77, 12],
            "acquired_at": datetime(2026, 6, 21, tzinfo=UTC), "orbit_direction": "descending"}
    with main.SessionLocal() as s:
        store_run(s, meta, "TEST-FIXTURE", {}, {}, [feat])
        store_run(s, meta, "TEST-FIXTURE", {}, {}, [feat])  # idempotent re-run
    r = c.get("/api/detections", params={"bbox": "75,10,76,11", "min_confidence": 0.7, "classes": "3",
                                          "start": "2026-06-01", "end": "2026-06-30"}).json()
    assert len(r["features"]) == 1
    fid = r["features"][0]["id"]
    assert c.get(f"/api/detections/{fid}").json()["properties"]["scene_id"] == "S1_TEST_SCENE"
    assert c.get("/api/detections", params={"bbox": "0,0,1,1"}).json()["features"] == []
    assert c.get("/api/detections", params={"min_confidence": 0.95}).json()["features"] == []
    assert c.get("/api/detections", params={"classes": "1,2"}).json()["features"] == []
    assert c.get("/api/scenes").json()["scenes"][0]["n_detections"] == 1
