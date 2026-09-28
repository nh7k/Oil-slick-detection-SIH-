"""Grid and georeferencing invariants.

These guard the thing that, if wrong, silently corrupts the whole dataset: the
exact Cerulean grid alignment. Synthetic arrays only - confined to tests/ per
the project rules.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from oilspill import config
from oilspill.crsutil import estimate_utm_epsg, metric_properties
from oilspill.georef import (
    GridWindow,
    assert_on_grid,
    db_to_uint8,
    grid_window_from_bounds,
    lat_to_row,
    lon_to_col,
)


class TestGridConstants:
    def test_pixel_size_matches_cerulean_formula(self):
        # 180 / 2**9 / 512, the WorldCRS84Quad z9 scale-2 pixel.
        assert config.PIXEL_DEG == pytest.approx(0.0006866455078125, abs=1e-15)

    def test_pixel_size_divides_the_world_evenly(self):
        # A full 360 degrees must be an exact integer number of pixels, or the
        # global grid does not close.
        n = 360.0 / config.PIXEL_DEG
        assert n == pytest.approx(round(n), abs=1e-6)
        assert round(n) == 2 * (2**config.GRID_ZOOM) * config.GRID_TILE_PX

    def test_ground_sample_distance_is_about_76m(self):
        assert 76.0 < config.PIXEL_M_EQUATOR < 77.0


class TestGridWindow:
    def test_transform_is_grid_aligned(self):
        w = GridWindow(col_off=100_000, row_off=50_000, width=512, height=512)
        assert_on_grid(w.transform, config.CRS)

    def test_origin_window_starts_at_the_grid_origin(self):
        w = GridWindow(0, 0, 10, 10)
        assert w.transform.c == pytest.approx(config.GRID_ORIGIN_LON)
        assert w.transform.f == pytest.approx(config.GRID_ORIGIN_LAT)

    def test_bounds_round_trip_through_col_row(self):
        w = GridWindow(123_456, 78_901, 512, 256)
        west, south, east, north = w.bounds
        assert lon_to_col(west) == pytest.approx(w.col_off, abs=1e-6)
        assert lat_to_row(north) == pytest.approx(w.row_off, abs=1e-6)
        assert lon_to_col(east) == pytest.approx(w.col_off + w.width, abs=1e-6)
        assert lat_to_row(south) == pytest.approx(w.row_off + w.height, abs=1e-6)

    def test_window_from_bounds_contains_the_request(self):
        req = (75.0, 10.0, 75.5, 10.4)
        w = grid_window_from_bounds(req)
        west, south, east, north = w.bounds
        assert west <= req[0] and south <= req[1]
        assert east >= req[2] and north >= req[3]

    def test_padding_grows_the_window_both_ways(self):
        req = (75.0, 10.0, 75.5, 10.4)
        plain = grid_window_from_bounds(req)
        padded = grid_window_from_bounds(req, pad_m=20_000.0)
        assert padded.width > plain.width
        assert padded.height > plain.height
        assert padded.bounds[0] < plain.bounds[0]
        assert padded.bounds[3] > plain.bounds[3]

    def test_windows_built_separately_share_one_global_grid(self):
        # Two overlapping requests must agree on pixel edges, or tiles built in
        # different runs will not line up.
        a = grid_window_from_bounds((75.0, 10.0, 75.5, 10.4))
        b = grid_window_from_bounds((75.2, 10.1, 75.9, 10.8))
        offset = (b.transform.c - a.transform.c) / config.PIXEL_DEG
        assert offset == pytest.approx(round(offset), abs=1e-6)

    def test_high_latitude_padding_does_not_explode(self):
        w = grid_window_from_bounds((-2.0, 78.0, 1.0, 79.0), pad_m=20_000.0)
        assert w.bounds[3] <= 90.0
        assert w.width > 0 and w.height > 0


class TestAssertOnGrid:
    def test_rejects_wrong_crs(self):
        w = GridWindow(0, 0, 10, 10)
        with pytest.raises(AssertionError, match="EPSG"):
            assert_on_grid(w.transform, "EPSG:3857")

    def test_rejects_wrong_pixel_size(self):
        bad = Affine(0.001, 0, -180, 0, -0.001, 90)
        with pytest.raises(AssertionError, match="pixel size"):
            assert_on_grid(bad, config.CRS)

    def test_rejects_rotation(self):
        px = config.PIXEL_DEG
        bad = Affine(px, 1e-6, -180, 1e-6, -px, 90)
        with pytest.raises(AssertionError, match="rotated"):
            assert_on_grid(bad, config.CRS)

    def test_rejects_half_pixel_offset(self):
        # The classic silent corruption: a half-pixel shift.
        px = config.PIXEL_DEG
        bad = Affine(px, 0, -180 + px / 2, 0, -px, 90)
        with pytest.raises(AssertionError, match="grid-aligned"):
            assert_on_grid(bad, config.CRS)


class TestDbToUint8:
    def test_nodata_stays_zero_and_data_never_collides_with_it(self):
        db = np.array([[0.0, 10.0, 25.0, 40.0]], dtype=np.float32)
        out = db_to_uint8(db, (5.0, 35.0))
        assert out[0, 0] == config.NODATA == 0
        assert (out[0, 1:] >= 1).all(), "valid data must not be mapped onto nodata"

    def test_clipping_saturates_at_the_ends(self):
        db = np.array([[1.0, 100.0]], dtype=np.float32)
        out = db_to_uint8(db, (5.0, 35.0))
        assert out[0, 0] == 1
        assert out[0, 1] == 255

    def test_monotonic(self):
        db = np.linspace(5.0, 35.0, 64, dtype=np.float32)[None, :]
        out = db_to_uint8(db, (5.0, 35.0)).astype(int)
        assert (np.diff(out[0]) >= 0).all()

    def test_rejects_degenerate_range(self):
        with pytest.raises(ValueError):
            db_to_uint8(np.zeros((2, 2), dtype=np.float32), (10.0, 10.0))


class TestDnToDb:
    def test_zero_dn_maps_to_exactly_zero(self):
        # dn_to_db must keep nodata finite and distinguishable.
        assert config.dn_to_db(np.array([0.0]))[0] == 0.0

    def test_monotonic_in_dn(self):
        db = config.dn_to_db(np.array([0, 1, 10, 100, 1000, 10000]))
        assert (np.diff(db) > 0).all()


class TestCrsUtil:
    @pytest.mark.parametrize(
        ("lon", "lat", "epsg"),
        [
            (75.0, 10.0, 32643),    # Kerala, north
            (-90.0, 27.0, 32616),   # Gulf of Mexico; -90 is the 15/16 boundary
            (11.6, -5.9, 32732),    # Congo basin, south; zone 32 spans 6-12E
            (-179.5, 10.0, 32601),  # just east of the antimeridian
            (179.5, 10.0, 32660),   # just west of it
        ],
    )
    def test_utm_zone_selection(self, lon, lat, epsg):
        assert estimate_utm_epsg(lon, lat) == epsg

    def test_longitude_wrapping_never_yields_zone_61(self):
        for lon in (180.0, 180.5, 540.0, -180.0):
            epsg = estimate_utm_epsg(lon, 0.0)
            zone = epsg % 100
            assert 1 <= zone <= 60, f"lon={lon} gave zone {zone}"

    def test_metric_properties_of_a_known_square(self):
        # ~1 km square at the equator, where a degree is well behaved.
        from shapely.geometry import box

        d = 1000.0 / 111_320.0
        props = metric_properties(box(0.0, 0.0, d, d))
        assert props["area_m2"] == pytest.approx(1_000_000.0, rel=0.02)
        assert props["perimeter_m"] == pytest.approx(4_000.0, rel=0.02)
        # Polsby-Popper of a square is pi/4.
        assert props["polsby_popper"] == pytest.approx(np.pi / 4, rel=0.02)

    def test_length_is_the_major_axis(self):
        from shapely.geometry import box

        dx = 5000.0 / 111_320.0
        dy = 500.0 / 111_320.0
        props = metric_properties(box(0.0, 0.0, dx, dy))
        assert props["length_m"] == pytest.approx(5000.0, rel=0.03)
        assert props["width_m"] == pytest.approx(500.0, rel=0.03)


class TestClassMaps:
    def test_every_cerulean_class_maps_somewhere(self):
        for cls_id in config.CERULEAN_CLS:
            assert cls_id in config.CLS_TO_MODEL, f"cls {cls_id} unmapped"

    def test_positive_classes_are_the_vessel_infra_natural_family(self):
        assert config.POSITIVE_CLS == (3, 4, 5, 6, 7, 8)

    def test_model_classes_are_contiguous(self):
        assert sorted(config.MODEL_CLASSES) == list(range(config.N_CLASSES))
