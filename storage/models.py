"""数据访问层 — 封装常用查询"""

import json
import math
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
    wifi_confidence=None, mmwave_confidence=None, thermal_confidence=None,
    nlos_flag=False, missing_modalities=None, modalities_json=None, activity_state="unknown",
    posture=None, fall_status=None, sensor_contact=None, source="replay",
    # portable_v2 parameters
    rr_wifi=None, rr_mm=None, hr_wifi=None, hr_mm=None,
    rr_conf=None, hr_conf=None,
    quality_event=False,
    rr_source=None, hr_source=None,
    rr_truth=None, hr_truth=None,
) -> dict:
    nlos_int = int(bool(nlos_flag))
    quality_int = int(quality_event or 0)
    with get_db() as conn:
        conn.execute("""
            INSERT INTO sensing_windows
            (window_id, timestamp, resident_id, rr, hr, body_temp,
             wifi_conf, mmwave_conf, thermal_conf, nlos_flag, missing_mods,
             modalities_json, activity_state, posture, fall_status, sensor_contact, source,
             rr_wifi, rr_mm, hr_wifi, hr_mm, rr_conf, hr_conf,
             quality_event, rr_source, hr_source, rr_truth, hr_truth)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(window_id) DO UPDATE SET
                timestamp=excluded.timestamp,
                resident_id=excluded.resident_id,
                rr=excluded.rr,
                hr=excluded.hr,
                body_temp=excluded.body_temp,
                wifi_conf=excluded.wifi_conf,
                mmwave_conf=excluded.mmwave_conf,
                thermal_conf=excluded.thermal_conf,
                nlos_flag=excluded.nlos_flag,
                missing_mods=excluded.missing_mods,
                modalities_json=excluded.modalities_json,
                activity_state=excluded.activity_state,
                posture=excluded.posture,
                fall_status=excluded.fall_status,
                sensor_contact=excluded.sensor_contact,
                source=excluded.source,
                rr_wifi=excluded.rr_wifi,
                rr_mm=excluded.rr_mm,
                hr_wifi=excluded.hr_wifi,
                hr_mm=excluded.hr_mm,
                rr_conf=excluded.rr_conf,
                hr_conf=excluded.hr_conf,
                quality_event=excluded.quality_event,
                rr_source=excluded.rr_source,
                hr_source=excluded.hr_source,
                rr_truth=excluded.rr_truth,
                hr_truth=excluded.hr_truth
        """, (
            window_id, timestamp.isoformat(),
            resident_id,
            respiration_rate, heart_rate, body_temp,
            wifi_confidence, mmwave_confidence, thermal_confidence,
            nlos_int, json.dumps(missing_modalities or []),
            modalities_json,
            activity_state, posture, fall_status, sensor_contact, source,
            rr_wifi, rr_mm, hr_wifi, hr_mm,
            float(rr_conf) if rr_conf is not None else None,
            float(hr_conf) if hr_conf is not None else None,
            quality_int, rr_source, hr_source, rr_truth, hr_truth,
        ))
        row = conn.execute(
            "SELECT * FROM sensing_windows WHERE window_id=?", (window_id,)
        ).fetchone()
        return _row_to_dict(row)


VALID_METRIC_COLS = {"heart_rate": "hr", "respiration_rate": "rr", "body_temp": "body_temp"}

def query_recent_windows(
    resident_id: str,
    metric: str,
    minutes: int = 60,
    reference_timestamp: Optional[datetime] = None,
) -> list[dict]:
    col = VALID_METRIC_COLS[metric]  # KeyError 兜底, 防 SQL 注入
    ref = reference_timestamp or datetime.now()
    if ref.tzinfo is not None:
        ref = ref.replace(tzinfo=None)
    threshold = (ref - timedelta(minutes=minutes)).isoformat()
    upper = ref.isoformat()
    with get_db() as conn:
        rows = conn.execute(f"""
            SELECT timestamp, {col} as value, wifi_conf, mmwave_conf, modalities_json,
                   rr_wifi, rr_mm, hr_wifi, hr_mm, rr_conf, hr_conf,
                   quality_event, rr_source, hr_source
            FROM sensing_windows
            WHERE resident_id=? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, (resident_id, threshold, upper)).fetchall()
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


def query_portable_v2_windows(resident_id: str = "resident_01", limit: int = 100) -> list[dict]:
    """返回包含 portable_v2 扩展字段的传感窗口。"""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM sensing_windows
            WHERE resident_id=? AND rr_wifi IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (resident_id, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── Trends ──

def query_trend_range(
    resident_id: str,
    reference: Optional[datetime] = None,
    minutes: Optional[int] = 60,
    max_points: int = 1200,
) -> list[dict]:
    """Return sensing windows for trend charting, with forced sampling.

    Uses the data timeline (not wall-clock) so historical replay works.
    When row count exceeds max_points, uniformly sample but always
    include rows flagged with quality_event, modality_conflict, or nlos_flag.
    """
    ref = reference or datetime.now()
    if ref.tzinfo is not None:
        ref = ref.replace(tzinfo=None)
    upper = ref.isoformat()

    with get_db() as conn:
        if minutes is None:
            rows = conn.execute("""
                SELECT * FROM sensing_windows
                WHERE resident_id=? AND timestamp <= ?
                ORDER BY timestamp
            """, (resident_id, upper)).fetchall()
        else:
            threshold = (ref - timedelta(minutes=minutes)).isoformat()
            rows = conn.execute("""
                SELECT * FROM sensing_windows
                WHERE resident_id=? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (resident_id, threshold, upper)).fetchall()

    result = [_row_to_dict(r) for r in rows]

    if len(result) <= max_points:
        return result

    # Forced sampling: keep flagged rows, then uniformly sample the rest
    flagged = [r for r in result if r.get("quality_event") or r.get("nlos_flag")]
    flagged_ids = {r["window_id"] for r in flagged}
    normal = [r for r in result if r["window_id"] not in flagged_ids]

    # Always keep first and last
    keep_ids = {result[0]["window_id"], result[-1]["window_id"]}
    keep_ids.update(flagged_ids)

    budget = max_points - len(keep_ids)
    if budget > 0 and normal:
        step = max(1, math.ceil(len(normal) / budget))
        keep_ids.update(normal[i]["window_id"] for i in range(0, len(normal), step))

    sampled = [r for r in result if r["window_id"] in keep_ids]
    if len(sampled) > max_points:
        # Flagged windows have priority; cap deterministically if they alone
        # exceed the rendering budget.
        sample_step = math.ceil(len(sampled) / max_points)
        sampled = sampled[::sample_step]
        if sampled[-1]["window_id"] != result[-1]["window_id"]:
            sampled[-1] = result[-1]
    return sampled


# ── Episodes (filtered, paginated) ──

def query_filtered_episodes(
    resident_id: str,
    level: Optional[str] = None,
    event_type: Optional[str] = None,
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return paginated episodes with health_event enrichment.

    Returns (rows, total_count).
    """
    conditions = ["e.resident_id = ?"]
    params: list = [resident_id]

    if level:
        conditions.append("json_extract(e.decision, '$.level') = ?")
        params.append(level)
    if event_type:
        conditions.append("h.event_type = ?")
        params.append(event_type)
    if from_dt:
        conditions.append("e.start_time >= ?")
        params.append(from_dt.isoformat())
    if to_dt:
        conditions.append("e.start_time < ?")
        params.append(to_dt.isoformat())
    if search:
        conditions.append(
            "(e.event_id LIKE ? OR json_extract(e.decision, '$.label') LIKE ? OR json_extract(e.evidence, '$.event.trigger_reason') LIKE ?)"
        )
        p = f"%{search}%"
        params.extend([p, p, p])

    where = " AND ".join(conditions)

    with get_db() as conn:
        count_row = conn.execute(f"""
            SELECT COUNT(*) as cnt
            FROM episode_logs e
            LEFT JOIN health_events h ON e.event_id = h.event_id
            WHERE {where}
        """, params).fetchone()
        total = count_row["cnt"] if count_row else 0

        rows = conn.execute(f"""
            SELECT e.*, h.event_type as health_event_type,
                   h.trigger_reason as health_trigger, h.handled
            FROM episode_logs e
            LEFT JOIN health_events h ON e.event_id = h.event_id
            WHERE {where}
            ORDER BY e.start_time DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()

    return [_row_to_dict(r) for r in rows], total


def seed_demo_data():
    """If DB is empty, insert demo sensing windows + episode_logs so the
    report and dashboard show data on first start without manual loading."""
    from storage.db import get_db, DB_PATH
    from pathlib import Path
    import uuid

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) as c FROM sensing_windows").fetchone()["c"]
        if count > 0:
            return False  # already has data

        # ── 30 sensing windows ──
        base_ts = datetime.now().replace(second=0, microsecond=0) - timedelta(hours=1)
        window_ids = []
        for i in range(30):
            wid = f"demo_{uuid.uuid4().hex[:6]}"
            window_ids.append(wid)
            ts = (base_ts + timedelta(minutes=i * 2)).isoformat()
            conn.execute("""INSERT INTO sensing_windows
                (window_id, timestamp, resident_id, hr, rr, body_temp,
                 wifi_conf, mmwave_conf, activity_state, nlos_flag, source, quality_event)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (wid, ts, "resident_01",
                 70 + (i % 20), 16 + (i % 6), 36.5 + (i % 5) * 0.2,
                 0.85, 0.75 if i % 5 else 0.3, "rest" if i % 3 else "walking",
                 1 if i % 7 == 0 else 0, "seed", 1 if i % 11 == 0 else 0))

        # ── Health events + episode_logs ──
        demo_decisions = [
            {
                "level": "L4", "label": "emergency",
                "event_interpretation": "HR=155 extreme, reflex arc triggered",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "HR=155 in marked_high band", "source": "RCP_NEWS2_2017_REFERENCE"}
                ],
            },
            {
                "level": "L2", "label": "resident_alert",
                "event_interpretation": "RR=24 vs personal baseline 19±2, z=2.5, resting position",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "RR=24 in elevated 21-24 band", "source": "RCP_NEWS2_2017_REFERENCE"},
                    {"type": "personal_baseline", "finding": "RR=24 vs baseline 19±2, z=2.5", "source": "RESIDENT_HISTORY"},
                    {"type": "activity_context", "finding": "No activity explanation for elevated RR", "source": "ACTIVITY_CONTEXT"},
                ],
            },
            {
                "level": "L1", "label": "continuous_observation",
                "event_interpretation": "HR deviation z=1.8, transient 60s, resolved",
                "clinical_basis": [
                    {"type": "personal_baseline", "finding": "HR z=1.8 from personal baseline", "source": "RESIDENT_HISTORY"},
                    {"type": "persistence", "finding": "Only 60s; sustained deterioration not established", "source": "PROJECT_POLICY"},
                ],
            },
            {
                "level": "L3", "label": "family_notification",
                "event_interpretation": "Sustained HR elevation z=3.2 over 12 min, activity context does not explain",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "HR=135 in marked_high band", "source": "RCP_NEWS2_2017_REFERENCE"},
                    {"type": "personal_baseline", "finding": "HR=135 vs baseline 72±3, z=3.2", "source": "RESIDENT_HISTORY"},
                    {"type": "persistence", "finding": "Sustained 720s, persistent deterioration", "source": "PROJECT_POLICY"},
                    {"type": "sensing_quality", "finding": "WiFi+mmWave consistent, reliable", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L0", "label": "record_only",
                "event_interpretation": "Slight HR rise during walking, normalized",
                "clinical_basis": [
                    {"type": "activity_context", "finding": "Walking explains transient HR rise", "source": "ACTIVITY_CONTEXT"},
                ],
            },
        ]

        event_ts = base_ts
        for d in demo_decisions:
            eid = f"seed_{uuid.uuid4().hex[:8]}"
            et = {"L4": "hr_abnormal", "L2": "rr_baseline_deviation", "L1": "hr_abnormal",
                  "L3": "hr_abnormal", "L0": "hr_abnormal"}[d["level"]]
            conn.execute("""INSERT OR IGNORE INTO health_events
                (event_id, window_id, event_type, timestamp, trigger_reason, handled)
                VALUES (?,?,?,?,?,?)""",
                (eid, window_ids[0], et, event_ts.isoformat(), d["event_interpretation"][:80], 1))
            conn.execute("""INSERT INTO episode_logs
                (episode_id, event_id, resident_id, start_time, end_time,
                 evidence, decision, action, audit)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (f"ep_{uuid.uuid4().hex[:8]}", eid, "resident_01",
                 event_ts.isoformat(), (event_ts + timedelta(seconds=30)).isoformat(),
                 json.dumps({"sensing_summary": {
                     "heart_rate": 70 + int(d["level"][1]) * 10,
                     "respiration_rate": 16 + int(d["level"][1]) * 2,
                     "body_temp": 36.5, "wifi_confidence": 0.85, "mmwave_confidence": 0.75,
                     "hr_wifi": 68 + int(d["level"][1]) * 10 if d["level"] in ("L3","L4") else None,
                     "hr_mm": 72 + int(d["level"][1]) * 10 if d["level"] in ("L3","L4") else None,
                     "nlos_flag": False, "activity_state": "rest",
                 }}),
                 json.dumps(d), json.dumps({"channel": "none" if d["level"] in ("L0","L1") else
                                              "screen" if d["level"] == "L2" else
                                              "family_push" if d["level"] == "L3" else "emergency"}),
                 json.dumps({"tools_called": ["nurse_rule:seed"], "step_count": 0})))
            event_ts += timedelta(minutes=5)

        return True  # data seeded
