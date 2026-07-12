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
    report and dashboard show data on first start without manual loading.

    Generates ~240 windows across 1 hour with varied patterns:
    normal, walking, NLOS, noisy, extreme vitals, modality conflict."""
    from storage.db import get_db
    import uuid, math

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) as c FROM sensing_windows").fetchone()["c"]
        if count > 0:
            return False

        base_ts = datetime.now().replace(second=0, microsecond=0) - timedelta(hours=1)
        window_ids = []

        # ── Generate 240 windows with varied patterns ──
        def _v(mid, pattern):
            """Generate per-modality values with realistic variation."""
            base, spread = pattern
            wifi = base + spread * (math.sin(mid * 0.5) + (hash(f"w{mid}") % 10 - 5) / 10)
            mm = base + spread * (math.sin(mid * 0.7 + 1) + (hash(f"m{mid}") % 10 - 5) / 10)
            return max(0, round(wifi, 1)), max(0, round(mm, 1))

        for i in range(240):
            wid = f"demo_{uuid.uuid4().hex[:6]}"
            window_ids.append(wid)
            ts = (base_ts + timedelta(seconds=i * 15)).isoformat()

            # Base vitals (normal resting)
            hr_base, rr_base, temp_base = 72, 16, 36.5

            # Segment 0-30: normal resting
            # Segment 30-45: walking (elevated HR/RR)
            # Segment 50-70: NLOS occlusion (mmwave degraded)
            # Segment 80-100: post-walk recovery
            # Segment 100-120: noisy / low confidence
            # Segment 120-130: extreme HR spike
            # Segment 130-150: normal
            # Segment 150-170: modality conflict (wifi/mm disagree)
            # Segment 170-200: normal with short NLOS bursts
            # Segment 200-220: elevated RR
            # Segment 220-240: normal

            seg = i // 10  # each segment = 10 windows = 2.5 min

            activity = "rest"
            nlos = 0
            wifi_conf = 0.85
            mmwave_conf = 0.75
            extra = {}  # per-modality extras

            if 30 <= i < 50:
                # Walking
                hr_base = 95 + (i - 30) // 2
                rr_base = 22 + (i - 30) // 5
                activity = "walking"
                wifi_conf = 0.80
                mmwave_conf = 0.75
            elif 50 <= i < 70:
                # NLOS occlusion
                hr_base = 78 + (i - 50) % 5
                rr_base = 18
                nlos = 1
                wifi_conf = 0.70
                mmwave_conf = 0.15  # mmwave heavily degraded in NLOS
            elif 80 <= i < 100:
                # Post-walk recovery
                hr_base = 85 - (i - 80) // 2
                rr_base = 20 - (i - 80) // 5
                activity = "rest"
                wifi_conf = 0.78
                mmwave_conf = 0.72
            elif 100 <= i < 120:
                # Noisy/low confidence
                hr_base = 75 + (i % 5) * 3
                rr_base = 17 + (i % 4)
                wifi_conf = 0.35 + (i % 10) * 0.05
                mmwave_conf = 0.25 + (i % 8) * 0.05
                activity = "rest"
            elif 120 <= i < 130:
                # Extreme HR spike
                hr_base = 140 + (i - 120) * 3
                rr_base = 26 + (i - 120)
                activity = "rest"
                wifi_conf = 0.80
                mmwave_conf = 0.75
                extra = {"quality_event": 1, "hr_source": "wifi_main"}
            elif 150 <= i < 170:
                # Modality conflict: WiFi and mmWave disagree on HR
                hr_base = 78
                rr_base = 17
                activity = "rest"
                wifi_conf = 0.82
                mmwave_conf = 0.78
                extra = {"hr_wifi": 72.0, "hr_mm": 92.0, "hr_source": "fused_consistent",
                         "rr_wifi": 16.5, "rr_mm": 17.0,
                         "hr_conf": 0.75, "rr_conf": 0.80}
            elif 200 <= i < 220:
                # Elevated RR (baseline deviation)
                hr_base = 76
                rr_base = 22 + (i - 200) // 5
                activity = "rest"
                wifi_conf = 0.85
                mmwave_conf = 0.80
                extra = {"rr_source": "mmwave_main"}
            elif i >= 220:
                # Back to normal
                hr_base = 70 + (i % 5)
                rr_base = 16
                activity = "sleep"
                wifi_conf = 0.88
                mmwave_conf = 0.80

            hr_wifi_v = extra.get("hr_wifi", hr_base + (hash(f"hw{i}") % 6 - 3))
            hr_mm_v = extra.get("hr_mm", hr_base + (hash(f"hm{i}") % 6 - 3))
            rr_wifi_v = extra.get("rr_wifi", rr_base + (hash(f"rw{i}") % 4 - 2))
            rr_mm_v = extra.get("rr_mm", rr_base + (hash(f"rm{i}") % 4 - 2))

            conn.execute("""INSERT INTO sensing_windows
                (window_id, timestamp, resident_id, hr, rr, body_temp,
                 wifi_conf, mmwave_conf, thermal_conf, activity_state, nlos_flag, source,
                 quality_event, hr_wifi, hr_mm, rr_wifi, rr_mm, hr_conf, rr_conf,
                 hr_source, rr_source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (wid, ts, "resident_01",
                 hr_base, rr_base, temp_base + (hash(f"t{i}") % 3) * 0.1,
                 round(wifi_conf, 2), round(mmwave_conf, 2), 0.85,
                 activity, nlos, "seed",
                 extra.get("quality_event", 0),
                 hr_wifi_v, hr_mm_v, rr_wifi_v, rr_mm_v,
                 extra.get("hr_conf", 0.70 + (hash(f"hc{i}") % 30) / 100),
                 extra.get("rr_conf", 0.70 + (hash(f"rc{i}") % 30) / 100),
                 extra.get("hr_source"), extra.get("rr_source")))

        # ── Episode logs for various event types ──
        demo_decisions = [
            {
                "level": "L4", "label": "emergency",
                "event_type": "hr_abnormal",
                "event_interpretation": "HR=155 extreme, reflex arc — immediate escalation required",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "HR=155 in marked_high band (score 3)", "source": "RCP_NEWS2_2017_REFERENCE"},
                    {"type": "sensing_quality", "finding": "WiFi+mmWave consistent, reliable", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L3", "label": "family_notification",
                "event_type": "hr_abnormal",
                "event_interpretation": "Sustained HR=135 for 12 min, z=3.2, resting, requires family notification",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "HR=135 in high band (score 2)", "source": "RCP_NEWS2_2017_REFERENCE"},
                    {"type": "personal_baseline", "finding": "HR=135 vs baseline 72±3, z=3.2", "source": "RESIDENT_HISTORY"},
                    {"type": "persistence", "finding": "Sustained 720s, persistent deterioration established", "source": "PROJECT_POLICY"},
                    {"type": "sensing_quality", "finding": "Multi-modal agreement, reliable measurement", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L2", "label": "resident_alert",
                "event_type": "rr_baseline_deviation",
                "event_interpretation": "RR=26 elevated from baseline 16±2, z=5.0, resting",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "RR=26 in marked_high band (score 3)", "source": "RCP_NEWS2_2017_REFERENCE"},
                    {"type": "personal_baseline", "finding": "RR=26 vs personal baseline 16±2, z=5.0", "source": "RESIDENT_HISTORY"},
                    {"type": "activity_context", "finding": "Resting; no activity explanation", "source": "ACTIVITY_CONTEXT"},
                    {"type": "sensing_quality", "finding": "WiFi+mmWave agreement, reliable", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L1", "label": "continuous_observation",
                "event_type": "hr_abnormal",
                "event_interpretation": "HR z=1.8 during walking, normalized quickly",
                "clinical_basis": [
                    {"type": "personal_baseline", "finding": "HR z=1.8 deviation from personal baseline", "source": "RESIDENT_HISTORY"},
                    {"type": "activity_context", "finding": "Walking explains transient elevation", "source": "ACTIVITY_CONTEXT"},
                    {"type": "persistence", "finding": "Only 60s, not sustained", "source": "PROJECT_POLICY"},
                ],
            },
            {
                "level": "L2", "label": "resident_alert",
                "event_type": "modality_conflict",
                "event_interpretation": "WiFi HR=72 vs mmWave HR=92; delta=20 exceeds consistency threshold",
                "clinical_basis": [
                    {"type": "sensing_quality", "finding": "HR modality conflict delta=20", "source": "SENSOR_FUSION"},
                    {"type": "project_policy", "finding": "No reliable dominant modality, needs recheck", "source": "PROJECT_POLICY"},
                ],
            },
            {
                "level": "L0", "label": "record_only",
                "event_type": "low_confidence",
                "event_interpretation": "WiFi=0.40, mmWave=0.30, both below 0.5; recording only",
                "clinical_basis": [
                    {"type": "sensing_quality", "finding": "Dual low confidence, data unreliable", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L1", "label": "continuous_observation",
                "event_type": "nlos_occlusion",
                "event_interpretation": "mmWave confidence=0.15 due to NLOS, WiFi maintains 0.70",
                "clinical_basis": [
                    {"type": "sensing_quality", "finding": "mmWave degraded by NLOS; WiFi still reliable", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L3", "label": "family_notification",
                "event_type": "rr_bradypnea",
                "event_interpretation": "RR=6, below NEWS2 threshold, requires family notification",
                "clinical_basis": [
                    {"type": "absolute_reference", "finding": "RR=6 in marked_low band (score 3)", "source": "RCP_NEWS2_2017_REFERENCE"},
                    {"type": "sensing_quality", "finding": "WiFi+mmWave agreement, reliable", "source": "SENSOR_FUSION"},
                ],
            },
            {
                "level": "L4", "label": "emergency",
                "event_type": "fall_detected",
                "event_interpretation": "Fall detected with HR=105, needs immediate check",
                "clinical_basis": [
                    {"type": "fall_context", "finding": "Fall detected; injury status unknown", "source": "NICE_NG249_2025"},
                    {"type": "absolute_reference", "finding": "HR=105 in elevated band", "source": "RCP_NEWS2_2017_REFERENCE"},
                ],
            },
        ]

        event_ts = base_ts + timedelta(minutes=2)
        level_channel = {"L0": "none", "L1": "none", "L2": "screen", "L3": "family_push", "L4": "emergency"}
        for d in demo_decisions:
            eid = f"seed_{uuid.uuid4().hex[:8]}"
            conn.execute("""INSERT OR IGNORE INTO health_events
                (event_id, window_id, event_type, timestamp, trigger_reason, handled)
                VALUES (?,?,?,?,?,?)""",
                (eid, window_ids[0], d["event_type"],
                 event_ts.isoformat(), d["event_interpretation"][:80], 1))
            conn.execute("""INSERT INTO episode_logs
                (episode_id, event_id, resident_id, start_time, end_time,
                 evidence, decision, action, audit)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (f"ep_{uuid.uuid4().hex[:8]}", eid, "resident_01",
                 event_ts.isoformat(), (event_ts + timedelta(seconds=30)).isoformat(),
                 json.dumps({"sensing_summary": {
                     "heart_rate": 70 + int(d["level"][1]) * 10,
                     "respiration_rate": 16 + int(d["level"][1]) * 3,
                     "body_temp": 36.5,
                     "wifi_confidence": 0.85, "mmwave_confidence": 0.75,
                     "hr_wifi": 68 + int(d["level"][1]) * 10 if d["level"] in ("L3","L4") else None,
                     "hr_mm": 72 + int(d["level"][1]) * 10 if d["level"] in ("L3","L4") else None,
                     "nlos_flag": 0, "activity_state": "rest",
                 }}),
                 json.dumps(d), json.dumps({"channel": level_channel.get(d["level"], "none")}),
                 json.dumps({"tools_called": ["nurse_rule:seed"], "step_count": 0})))
            event_ts += timedelta(minutes=6)

        return True
