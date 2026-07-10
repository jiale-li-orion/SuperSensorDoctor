"""Tests for BaselineProvider — DB-grounded personal vital sign baselines."""

import os
import pytest
from datetime import datetime, timedelta
from storage.db import get_db, init_db, DB_PATH
from storage.models import insert_sensing_window
from agent_layer.baseline_provider import BaselineProvider

RESIDENT = "resident_01"


@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    yield
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


class TestBaselineProvider:

    def setup_method(self):
        """Seed 20 data points: values [70, 71, ..., 89] → mean=79.5, std≈5.92."""
        for i, val in enumerate(range(70, 90)):
            ts = datetime.now() - timedelta(hours=i)
            insert_sensing_window(
                window_id=f"bp_test_{i}", timestamp=ts,
                resident_id=RESIDENT,
                heart_rate=float(val),
                wifi_confidence=0.9, mmwave_confidence=0.85,
            )

    # ── compute() ────────────────────────────────────────────────────────

    def test_compute_returns_baseline(self):
        bp = BaselineProvider(days=30, min_points=5)
        result = bp.compute(RESIDENT, "hr", at_timestamp=datetime.now())
        assert result is not None
        assert "mean" in result
        assert "std" in result
        assert "count" in result
        assert result["count"] >= 20
        assert 75 < result["mean"] < 84

    def test_compute_with_insufficient_data(self):
        bp = BaselineProvider(days=30, min_points=100)
        result = bp.compute(RESIDENT, "hr", at_timestamp=datetime.now())
        assert result is None  # need 100, have 20

    def test_compute_empty_db(self):
        with get_db() as conn:
            conn.execute("DELETE FROM sensing_windows")
        bp = BaselineProvider()
        result = bp.compute(RESIDENT, "hr", at_timestamp=datetime.now())
        assert result is None

    def test_compute_min_floor_prevents_zerodiv(self):
        """Single-value data should not crash from std=0."""
        with get_db() as conn:
            conn.execute("DELETE FROM sensing_windows")
        insert_sensing_window(
            "single_val", datetime.now(), resident_id=RESIDENT,
            heart_rate=75.0, wifi_confidence=0.9, mmwave_confidence=0.85,
        )
        bp = BaselineProvider()
        result = bp.compute(RESIDENT, "hr", at_timestamp=datetime.now())
        if result:
            assert result["std"] > 0  # floor should ensure this

    def test_compute_respects_timestamp(self):
        """Data after the at_timestamp window should be excluded."""
        old_ts = datetime.now() - timedelta(days=100)
        bp = BaselineProvider(days=30)
        result = bp.compute(RESIDENT, "hr", at_timestamp=old_ts)
        assert result is None  # seeded data is all recent, not in old window

    def test_compute_invalid_metric_returns_none(self):
        bp = BaselineProvider()
        result = bp.compute(RESIDENT, "invalid_metric", at_timestamp=datetime.now())
        assert result is None

    def test_compute_default_lookback_includes_recent_data(self):
        """Default 21-day lookback should include data seeded within hours."""
        bp = BaselineProvider()  # days=21
        result = bp.compute(RESIDENT, "hr", at_timestamp=datetime.now())
        assert result is not None
        assert result["count"] == 20

    # ── compute_metric() ─────────────────────────────────────────────────

    def test_compute_metric_returns_z_score(self):
        bp = BaselineProvider()
        result = bp.compute_metric(RESIDENT, "hr", 85.0, at_timestamp=datetime.now())
        assert result is not None
        assert "z_score" in result
        assert isinstance(result["z_score"], float)

    def test_compute_metric_returns_zscore_2sigma(self):
        bp = BaselineProvider()
        result = bp.compute_metric(RESIDENT, "hr", 100.0, at_timestamp=datetime.now())
        assert result is not None
        assert "z_score_2sigma" in result
        assert isinstance(result["z_score_2sigma"], float)

    def test_compute_metric_value_near_mean_low_z(self):
        """A value very close to mean (~79.5) should have |z_score| < 1."""
        bp = BaselineProvider()
        result = bp.compute_metric(RESIDENT, "hr", 79.0, at_timestamp=datetime.now())
        assert result is not None
        assert abs(result["z_score"]) < 1.0

    def test_compute_metric_none_on_no_baseline(self):
        """compute_metric returns None when baseline cannot be computed."""
        bp = BaselineProvider(min_points=1000)
        result = bp.compute_metric(RESIDENT, "hr", 75.0, at_timestamp=datetime.now())
        assert result is None

    def test_compute_metric_invalid_metric_returns_none(self):
        bp = BaselineProvider()
        result = bp.compute_metric(RESIDENT, "invalid", 75.0, at_timestamp=datetime.now())
        assert result is None

    def test_compute_metric_with_old_timestamp_no_data(self):
        """Timestamp far in the past should yield no baseline data."""
        old_ts = datetime.now() - timedelta(days=365)
        bp = BaselineProvider(days=30)
        result = bp.compute_metric(RESIDENT, "hr", 75.0, at_timestamp=old_ts)
        assert result is None

    # ── RR metric ────────────────────────────────────────────────────────

    def test_compute_rr_metric(self):
        with get_db() as conn:
            conn.execute("DELETE FROM sensing_windows")
        for i in range(10):
            ts = datetime.now() - timedelta(hours=i)
            insert_sensing_window(
                window_id=f"rr_test_{i}", timestamp=ts,
                resident_id=RESIDENT,
                respiration_rate=16.0 + (i % 5 - 2),
                wifi_confidence=0.9, mmwave_confidence=0.85,
            )
        bp = BaselineProvider()
        result = bp.compute(RESIDENT, "rr", at_timestamp=datetime.now())
        assert result is not None
        assert 14 < result["mean"] < 18

    # ── body_temp metric ────────────────────────────────────────────────

    def test_compute_temp_metric(self):
        with get_db() as conn:
            conn.execute("DELETE FROM sensing_windows")
        for i in range(10):
            ts = datetime.now() - timedelta(hours=i)
            insert_sensing_window(
                window_id=f"temp_test_{i}", timestamp=ts,
                resident_id=RESIDENT,
                body_temp=36.5 + (i % 5 - 2) * 0.1,
                wifi_confidence=0.9, mmwave_confidence=0.85,
            )
        bp = BaselineProvider()
        result = bp.compute(RESIDENT, "temp", at_timestamp=datetime.now())
        assert result is not None
        assert 36.3 < result["mean"] < 36.7
