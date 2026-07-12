"""SQLite 连接与建表"""

import sqlite3
from pathlib import Path
from threading import Lock

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
    modalities_json TEXT,
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

_init_lock = Lock()
_init_done = False


def get_db() -> sqlite3.Connection:
    global _init_done
    if not _init_done:
        with _init_lock:
            if not _init_done:
                # Use a temporary non-WAL connection for init to avoid lock conflicts
                conn = sqlite3.connect(DB_PATH)
                conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript(SCHEMA_SQL)
                # Column migrations
                for col in [
                    "modalities_json TEXT",
                    "rr_wifi REAL", "rr_mm REAL", "hr_wifi REAL", "hr_mm REAL",
                    "rr_conf REAL", "hr_conf REAL",
                    "quality_event INTEGER DEFAULT 0",
                    "rr_source TEXT", "hr_source TEXT",
                    "rr_truth REAL", "hr_truth REAL",
                ]:
                    try:
                        conn.execute(f"ALTER TABLE sensing_windows ADD COLUMN {col}")
                    except sqlite3.OperationalError:
                        pass
                conn.close()
                _init_done = True
                # Seed demo data on first init (lazy import to avoid cycle)
                try:
                    from storage.models import seed_demo_data
                    seed_demo_data()
                except Exception:
                    pass  # seeding is best-effort

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Force re-initialization (used for testing)."""
    global _init_done
    _init_done = False
    get_db().close()
