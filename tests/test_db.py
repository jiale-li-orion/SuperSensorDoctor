import pytest
from datetime import datetime
from storage.db import get_db, init_db, DB_PATH
from storage.models import (
    insert_sensing_window, insert_health_event, insert_episode_log,
    query_recent_windows, query_pending_events, query_episodes_by_resident,
)


@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    yield
    import os
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


def test_insert_and_query_window():
    init_db()
    row = insert_sensing_window(
        window_id="w001", timestamp=datetime.now(),
        heart_rate=72.0, respiration_rate=16.0,
        wifi_confidence=0.9,
    )
    assert row is not None

    windows = query_recent_windows("resident_01", metric="heart_rate", minutes=60)
    assert len(windows) >= 1


def test_insert_and_query_event():
    init_db()
    insert_sensing_window(window_id="w001", timestamp=datetime.now())
    evt = insert_health_event(
        event_id="e001", window_id="w001",
        event_type="hr_abnormal", timestamp=datetime.now(),
        trigger_reason="test",
    )
    assert evt is not None

    pending = query_pending_events()
    assert len(pending) >= 1


def test_episode_log():
    init_db()
    insert_sensing_window(window_id="w001", timestamp=datetime.now())
    insert_health_event(
        event_id="e001", window_id="w001",
        event_type="test", timestamp=datetime.now(),
        trigger_reason="test",
    )
    log = insert_episode_log(
        episode_id="ep001", event_id="e001",
        resident_id="resident_01",
        start_time=datetime.now(),
    )
    assert log is not None

    logs = query_episodes_by_resident("resident_01")
    assert len(logs) >= 1
