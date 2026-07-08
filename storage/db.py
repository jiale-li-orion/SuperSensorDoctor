"""SQLite 连接与建表"""

import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = str(DATA_DIR / "supersense.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sensing_windows (
    window_id     TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    resident_id   TEXT DEFAULT 'resident_01',
    rr            REAL,
    hr            REAL,
    body_temp     REAL,
    wifi_conf     REAL DEFAULT 1.0,
    mmwave_conf   REAL DEFAULT 1.0,
    thermal_conf  REAL DEFAULT 1.0,
    nlos_flag     INTEGER DEFAULT 0,
    missing_mods  TEXT DEFAULT '[]',
    activity_state TEXT DEFAULT 'unknown',
    posture       TEXT,
    fall_status   TEXT,
    sensor_contact INTEGER,
    source        TEXT DEFAULT 'replay'
);
CREATE INDEX IF NOT EXISTS idx_windows_time ON sensing_windows(resident_id, timestamp);

CREATE TABLE IF NOT EXISTS health_events (
    event_id      TEXT PRIMARY KEY,
    window_id     TEXT NOT NULL REFERENCES sensing_windows(window_id),
    event_type    TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    trigger_reason TEXT,
    rule_markers  TEXT DEFAULT '{}',
    handled       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_pending ON health_events(handled, timestamp);

CREATE TABLE IF NOT EXISTS episode_logs (
    episode_id    TEXT PRIMARY KEY,
    event_id      TEXT NOT NULL REFERENCES health_events(event_id),
    resident_id   TEXT DEFAULT 'resident_01',
    start_time    TEXT NOT NULL,
    end_time      TEXT,
    evidence      TEXT DEFAULT '{}',
    decision      TEXT DEFAULT '{}',
    action        TEXT DEFAULT '{}',
    audit         TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_episodes_time ON episode_logs(resident_id, start_time);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA_SQL)
