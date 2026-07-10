"""数据访问层 — 封装常用查询"""

import json
from datetime import datetime, timedelta
from typing import Optional
from storage.db import get_db


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


# ── Sensing Windows ──

def insert_sensing_window(
    window_id: str, timestamp: datetime,
    resident_id: str = "resident_01",
    heart_rate=None, respiration_rate=None, body_temp=None,
    wifi_confidence=1.0, mmwave_confidence=1.0, thermal_confidence=1.0,
    nlos_flag=False, missing_modalities=None, modalities_json=None, activity_state="unknown",
    posture=None, fall_status=None, sensor_contact=None, source="replay",
) -> dict:
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sensing_windows
            (window_id, timestamp, resident_id, rr, hr, body_temp,
             wifi_conf, mmwave_conf, thermal_conf, nlos_flag, missing_mods,
             modalities_json,
             activity_state, posture, fall_status, sensor_contact, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            window_id, timestamp.isoformat(),
            resident_id,
            respiration_rate, heart_rate, body_temp,
            wifi_confidence, mmwave_confidence, thermal_confidence,
            int(nlos_flag), json.dumps(missing_modalities or []),
            modalities_json,
            activity_state, posture, fall_status, sensor_contact, source,
        ))
        row = conn.execute(
            "SELECT * FROM sensing_windows WHERE window_id=?", (window_id,)
        ).fetchone()
        return _row_to_dict(row)


VALID_METRIC_COLS = {"heart_rate": "hr", "respiration_rate": "rr", "body_temp": "body_temp"}

def query_recent_windows(resident_id: str, metric: str, minutes: int = 60) -> list[dict]:
    col = VALID_METRIC_COLS[metric]  # KeyError 兜底, 防 SQL 注入
    threshold = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    with get_db() as conn:
        rows = conn.execute(f"""
            SELECT timestamp, {col} as value, wifi_conf, mmwave_conf, modalities_json
            FROM sensing_windows
            WHERE resident_id=? AND timestamp >= ?
            ORDER BY timestamp
        """, (resident_id, threshold)).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── Health Events ──

def insert_health_event(
    event_id: str, window_id: str, event_type: str,
    timestamp: datetime, trigger_reason: str,
    rule_markers: Optional[dict] = None,
) -> dict:
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO health_events
            (event_id, window_id, event_type, timestamp, trigger_reason, rule_markers)
            VALUES (?,?,?,?,?,?)
        """, (
            event_id, window_id, event_type,
            timestamp.isoformat(), trigger_reason,
            json.dumps(rule_markers or {}),
        ))
        row = conn.execute(
            "SELECT * FROM health_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return _row_to_dict(row)


def query_pending_events() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM health_events
            WHERE handled=0
            ORDER BY timestamp
        """).fetchall()
        return [_row_to_dict(r) for r in rows]


def mark_event_handled(event_id: str):
    with get_db() as conn:
        conn.execute(
            "UPDATE health_events SET handled=1 WHERE event_id=?", (event_id,)
        )


# ── Episode Logs ──

def insert_episode_log(
    episode_id: str, event_id: str, resident_id: str,
    start_time: datetime, end_time=None,
    evidence=None, decision=None, action=None, audit=None,
) -> dict:
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO episode_logs
            (episode_id, event_id, resident_id, start_time, end_time,
             evidence, decision, action, audit)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            episode_id, event_id, resident_id, start_time.isoformat(),
            end_time.isoformat() if end_time else None,
            json.dumps(evidence or {}), json.dumps(decision or {}),
            json.dumps(action or {}), json.dumps(audit or {}),
        ))
        row = conn.execute(
            "SELECT * FROM episode_logs WHERE episode_id=?", (episode_id,)
        ).fetchone()
        return _row_to_dict(row)


def query_episodes_by_resident(resident_id: str, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM episode_logs
            WHERE resident_id=?
            ORDER BY start_time DESC
            LIMIT ?
        """, (resident_id, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── Latest Sensing Window ──

def query_latest_sensing_window(resident_id: str) -> dict:
    """返回该居民最新的完整传感窗口 (含所有字段)"""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM sensing_windows
            WHERE resident_id=?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (resident_id,)).fetchone()
        return _row_to_dict(row)
