"""Tests for the Agent layer's report data access.

The point of this module is that a report window is selected by **time
range**, never by "the newest N rows", so a busy week cannot be silently
truncated. These tests are read-only: they must not delete the shared
database (see the Known issues section of the README).
"""

from datetime import datetime

import pytest

from agent_layer.report_data import (
    ReportWindow,
    load_report_window,
    resolve_reference_ts,
)
from storage.db import init_db, get_db
from storage.models import query_episodes_in_range


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _db_span():
    with get_db() as conn:
        row = conn.execute(
            "SELECT MIN(start_time) AS lo, MAX(start_time) AS hi FROM episode_logs"
        ).fetchone()
    return row["lo"], row["hi"]


class TestReferenceTimestamp:
    def test_explicit_reference_is_honoured(self):
        given = datetime(2026, 1, 2, 3, 4, 5)
        assert resolve_reference_ts("resident_01", given) == given

    def test_reference_falls_back_to_newest_evidence(self):
        lo, hi = _db_span()
        if not hi:
            pytest.skip("no episodes in the database")
        ref = resolve_reference_ts("resident_01")
        # Anchored on real evidence, not on wall-clock time.
        assert ref >= datetime.fromisoformat(lo)
        assert abs((ref - datetime.fromisoformat(hi)).total_seconds()) < 7 * 86400


class TestWindowSelection:
    def test_window_returns_episodes_oldest_first(self):
        window = load_report_window("resident_01")
        assert isinstance(window, ReportWindow)
        times = [e["start_time"] for e in window.episodes]
        assert times == sorted(times)

    def test_window_has_no_row_cap(self):
        """A wide window must return everything, not a page of it."""
        lo, hi = _db_span()
        if not lo:
            pytest.skip("no episodes in the database")
        everything = query_episodes_in_range("resident_01", lo, hi)
        assert everything, "expected episodes in the database"

        # Anchor at the end of history so the whole span falls in-window, then
        # compare against the same bounds fetched without any limit.
        ref = datetime.fromisoformat(hi)
        window = load_report_window("resident_01", reference_ts=ref, window_days=3650)
        assert window.count == len(everything)

    def test_window_is_bounded_by_window_days(self):
        ref = datetime.fromisoformat(_db_span()[1])
        wide = load_report_window("resident_01", reference_ts=ref, window_days=3650)
        narrow = load_report_window("resident_01", reference_ts=ref, window_days=1)
        assert narrow.count <= wide.count
        assert datetime.fromisoformat(narrow.start) < narrow.reference_ts

    def test_empty_resident_yields_empty_window(self):
        window = load_report_window("nobody_here")
        assert window.count == 0
        assert isinstance(window.reference_ts, datetime)


class TestJsonDecoding:
    def test_json_columns_are_decoded_to_dicts(self):
        window = load_report_window(
            "resident_01",
            reference_ts=datetime.fromisoformat(_db_span()[1]),
            window_days=3650,
        )
        if not window.episodes:
            pytest.skip("no episodes in the database")
        for ep in window.episodes:
            for key in ("decision", "evidence", "action", "audit"):
                assert isinstance(ep.get(key), dict), f"{key} not decoded"

    def test_context_accepts_a_loaded_window(self):
        from agent_layer.report_agent import build_report_context
        window = load_report_window(
            "resident_01",
            reference_ts=datetime.fromisoformat(_db_span()[1]),
            window_days=3650,
        )
        ctx = build_report_context(window.episodes, None, window.reference_ts)
        assert ctx["summary"]["total_records"] == window.count
