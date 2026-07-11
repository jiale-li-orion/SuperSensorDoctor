"""FastAPI Web 入口 — 医生工作站"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from storage.models import (
    query_recent_windows,
    query_pending_events,
    query_episodes_by_resident,
    query_latest_sensing_window,
    query_trend_range,
    query_filtered_episodes,
    get_db,
)
from storage.db import DB_PATH
from sensing_simulator.replay_engine import ReplayEngine
from sensing_simulator.sensor_hub import SensorHub
from scripts.load_portable_v2 import load_portable_v2_csv
from agent_layer.baseline_provider import BaselineProvider
from agent_layer.fusion_engine import FusionEngine
from agent_layer.state_objects import StateObject
from agent_layer.report_agent import ReportAgent
from agent_layer.clinical_policy import news2_reference

app = FastAPI(title="SuperSenseDoctor", version="0.1.0")

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """医生工作站首页"""
    windows = query_recent_windows("resident_01", "heart_rate", 60)
    pending = query_pending_events()
    episodes = query_episodes_by_resident("resident_01", 20)

    import json
    for ep in episodes:
        for key in ("decision", "evidence", "action", "audit"):
            if isinstance(ep.get(key), str):
                ep[key] = json.loads(ep[key])

    hr_values = [float(w["value"]) for w in windows if w.get("value") is not None]
    avg_hr = round(sum(hr_values) / len(hr_values), 1) if hr_values else None

    import json
    recent_episodes_json = json.dumps(episodes[:10], ensure_ascii=False, default=str)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_windows": len(windows),
        "pending_events": len(pending),
        "total_episodes": len(episodes),
        "avg_heart_rate": avg_hr,
        "recent_episodes": episodes[:10],
        "recent_episodes_json": recent_episodes_json,
        "pending_events_list": pending[:10],
        "db_path": str(DB_PATH),
        "now": datetime.now,
        "replay_running": getattr(getattr(request.app.state, "replay_engine", None), "_running", False),
        "replay_count": getattr(getattr(request.app.state, "replay_engine", None), "_windows_created", 0),
        "pv_loading": getattr(request.app.state, "pv_loading", False),
        "pv_count": getattr(request.app.state, "pv_count", 0),
        "pv_status": getattr(request.app.state, "pv_status", "idle"),
        "pv_error": getattr(request.app.state, "pv_error", None),
    })


@app.post("/api/replay/start")
async def start_replay(request: Request):
    if getattr(request.app.state, "replay_running", False):
        return {"status": "already_running", "windows_created": getattr(request.app.state, "replay_count", 0)}

    hub = getattr(request.app.state, "sensor_hub", None)
    if hub is None:
        return {"status": "error", "message": "No sensor_hub configured"}

    async def _replay_loop():
        request.app.state.replay_running = True
        request.app.state.replay_count = 0
        try:
            for state in ReplayEngine.generate_synthetic(duration_sec=30):
                if not getattr(request.app.state, "replay_running", True):
                    break
                await hub.compose(
                    window_id=state.window_id,
                    timestamp=datetime.now(),
                    data={
                        "heart_rate": state.heart_rate,
                        "respiration_rate": state.respiration_rate,
                        "body_temp": state.body_temp,
                        "wifi_confidence": state.wifi_confidence,
                        "mmwave_confidence": state.mmwave_confidence,
                        "thermal_confidence": state.thermal_confidence,
                        "nlos_flag": state.nlos_flag,
                        "fall_status": state.fall_status,
                        "activity_state": state.activity_state,
                        "source": state.source,
                    },
                )
                request.app.state.replay_count += 1
                await asyncio.sleep(0)
        finally:
            request.app.state.replay_running = False

    request.app.state.replay_task = asyncio.create_task(_replay_loop())
    return {"status": "started"}


@app.post("/api/replay/stop")
async def stop_replay(request: Request):
    if getattr(request.app.state, "replay_running", False):
        request.app.state.replay_running = False
        task = getattr(request.app.state, "replay_task", None)
        if task:
            task.cancel()
        return {"status": "stopped", "windows_created": getattr(request.app.state, "replay_count", 0)}
    return {"status": "not_running"}


@app.get("/api/replay/status")
async def replay_status(request: Request):
    return {
        "running": getattr(request.app.state, "replay_running", False),
        "windows_created": getattr(request.app.state, "replay_count", 0),
    }


@app.post("/api/data/load-team-data")
async def load_team_data(request: Request):
    """Load team data CSV/XLSX into DB through sensor_hub."""
    import pandas as pd
    from pathlib import Path

    hub = getattr(request.app.state, "sensor_hub", None)
    if hub is None:
        return {"status": "error", "message": "No sensor_hub configured"}

    project_root = Path(__file__).parent.parent
    csv_path = project_root / "team_data" / "fused_vital_signs_timestamp_rr_hr.csv"
    xlsx_path = project_root / "team_data" / "fall_status.xlsx"

    if not csv_path.exists():
        return {"status": "error", "message": f"CSV not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    fall_df = pd.read_excel(xlsx_path) if xlsx_path.exists() else pd.DataFrame()

    count = 0
    fall_windows = set()
    if not fall_df.empty and "window_id" in fall_df.columns:
        fall_windows = set(fall_df["window_id"].dropna().astype(str).tolist())

    for _, row in df.iterrows():
        window_id = str(row.get("window_id", f"team_{count}"))
        await hub.compose(
            window_id=window_id,
            timestamp=pd.to_datetime(row.get("timestamp", pd.Timestamp.now())).to_pydatetime(),
            data={
                "heart_rate": float(row["hr"]) if pd.notna(row.get("hr")) else None,
                "respiration_rate": float(row["rr"]) if pd.notna(row.get("rr")) else None,
                "body_temp": float(row.get("body_temp", row.get("temp", 36.5))) if pd.notna(row.get("body_temp", row.get("temp", None))) else None,
                "wifi_confidence": float(row.get("wifi_conf", 0.8)) if pd.notna(row.get("wifi_conf", None)) else 0.8,
                "mmwave_confidence": float(row.get("mmwave_conf", 0.7)) if pd.notna(row.get("mmwave_conf", None)) else 0.7,
                "thermal_confidence": float(row.get("thermal_conf", 0.75)) if pd.notna(row.get("thermal_conf", None)) else 0.75,
                "nlos_flag": bool(row.get("nlos_flag", False)),
                "fall_status": "fall" if window_id in fall_windows else "no_fall",
                "activity_state": str(row.get("activity_state", "unknown")),
                "source": "csv",
            },
        )
        count += 1

    return {"status": "ok", "windows_loaded": count}


@app.post("/api/data/load-portable-v2")
async def load_portable_v2_endpoint(request: Request):
    """Load 2,349 real per-modality windows from portable_v2 CSV through sensor_hub."""
    hub = getattr(request.app.state, "sensor_hub", None)
    if hub is None:
        return {"status": "error", "message": "No sensor_hub configured"}
    if getattr(request.app.state, "pv_loading", False):
        return {
            "status": "already_running",
            "windows_loaded": getattr(request.app.state, "pv_count", 0),
            "windows_target": 2349,
        }

    async def _load_loop():
        request.app.state.pv_loading = True
        request.app.state.pv_status = "loading"
        request.app.state.pv_error = None
        request.app.state.pv_count = 0
        try:
            cnt = await load_portable_v2_csv(
                hub,
                progress_callback=lambda c: setattr(request.app.state, "pv_count", c),
                evaluate=False,
            )
            request.app.state.pv_count = cnt
            request.app.state.pv_status = "completed"
        except Exception as exc:
            request.app.state.pv_status = "failed"
            request.app.state.pv_error = f"{type(exc).__name__}: {exc}"
        finally:
            request.app.state.pv_loading = False

    request.app.state.pv_task = asyncio.create_task(_load_loop())
    return {"status": "started", "windows_target": 2349}


@app.get("/api/data/portable-v2-status")
async def portable_v2_status(request: Request):
    return {
        "loading": getattr(request.app.state, "pv_loading", False),
        "status": getattr(request.app.state, "pv_status", "idle"),
        "error": getattr(request.app.state, "pv_error", None),
        "windows_loaded": getattr(request.app.state, "pv_count", 0),
        "windows_target": 2349,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "db": str(DB_PATH)}


RANGE_MAP = {"15m": 15, "1h": 60, "6h": 360, "24h": 1440}


@app.get("/api/trends")
async def api_trends(
    resident_id: str = Query("resident_01"),
    range: str = Query("1h"),
    metrics: str = Query("hr,rr"),
    max_points: int = Query(600, le=1200),
):
    """时间序列趋势数据，含融合值、每模态估计、置信度、事件标记。"""
    minutes = RANGE_MAP.get(range, 60)

    # Reference timestamp = latest window (supports historical replay)
    ref = None
    latest = query_latest_sensing_window(resident_id)
    if latest and latest.get("timestamp"):
        try:
            ref = datetime.fromisoformat(str(latest["timestamp"]))
        except (ValueError, TypeError):
            pass

    rows = query_trend_range(resident_id, reference=ref, minutes=minutes, max_points=max_points)
    if not rows:
        return {"status": "no_data"}

    req_metrics = [m.strip() for m in metrics.split(",")]
    points = []
    for r in rows:
        pt = {"timestamp": r.get("timestamp")}
        for m in req_metrics:
            col_map = {"hr": ("hr", "hr_wifi", "hr_mm", "hr_conf"),
                       "rr": ("rr", "rr_wifi", "rr_mm", "rr_conf"),
                       "temp": ("body_temp", None, None, None)}
            fused, wifi, mm, conf = col_map.get(m, (None,)*4)
            pt[f"{m}_fused"] = r.get(fused)
            if wifi:
                pt[f"{m}_wifi"] = r.get(wifi)
            if mm:
                pt[f"{m}_mm"] = r.get(mm)
            if conf:
                pt[f"{m}_conf"] = r.get(conf)
        pt["nlos_flag"] = bool(r.get("nlos_flag", False))
        pt["quality_event"] = bool(r.get("quality_event", 0))
        pt["source"] = r.get("source", "unknown")
        points.append(pt)

    return {
        "status": "ok",
        "range": range,
        "count": len(points),
        "reference_timestamp": ref.isoformat() if ref else None,
        "points": points,
    }


@app.get("/api/episodes")
async def api_episodes(
    resident_id: str = Query("resident_01"),
    level: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    from_dt: Optional[str] = Query(None, alias="from"),
    to_dt: Optional[str] = Query(None, alias="to"),
    search: Optional[str] = Query(None, alias="query"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
):
    """分页查询诊断事件，支持级别、类型、时间范围和关键词筛选。"""
    from_dt_parsed = None
    to_dt_parsed = None
    if from_dt:
        try:
            from_dt_parsed = datetime.fromisoformat(from_dt)
        except (ValueError, TypeError):
            pass
    if to_dt:
        try:
            to_dt_parsed = datetime.fromisoformat(to_dt)
        except (ValueError, TypeError):
            pass

    rows, total = query_filtered_episodes(
        resident_id, level=level, event_type=event_type,
        from_dt=from_dt_parsed, to_dt=to_dt_parsed,
        search=search, limit=limit, offset=offset,
    )

    # Parse JSON fields
    for r in rows:
        for key in ("decision", "evidence", "action", "audit"):
            if isinstance(r.get(key), str):
                try:
                    r[key] = json.loads(r[key])
                except (json.JSONDecodeError, TypeError):
                    pass

    return {
        "status": "ok",
        "total": total,
        "limit": limit,
        "offset": offset,
        "episodes": rows,
    }


@app.get("/api/vitals/latest")
async def api_vitals_latest():
    """最新体征快照 + 基线 + 绝对参考 + 完整融合 + 质量"""
    row = query_latest_sensing_window("resident_01")
    if not row:
        return {"status": "no_data"}

    ref_ts = datetime.now()
    try:
        if row.get("timestamp"):
            parsed = datetime.fromisoformat(str(row["timestamp"]))
            ref_ts = parsed
    except (ValueError, TypeError):
        pass

    # Baseline
    bp = BaselineProvider()
    baseline = {}
    for metric, value, col in [
        ("hr", row.get("hr"), "hr"),
        ("rr", row.get("rr"), "rr"),
        ("temp", row.get("body_temp"), "temp"),
    ]:
        if value is not None:
            try:
                bl = bp.compute_metric("resident_01", col, value, ref_ts)
                if bl:
                    baseline[metric] = {
                        "value": value,
                        "mean": bl["mean"],
                        "std": bl["std"],
                        "z_score": bl["z_score"],
                        "source": "RESIDENT_HISTORY",
                    }
            except Exception:
                pass

    # Absolute reference
    absolute_reference = {
        "heart_rate": news2_reference("hr", row.get("hr")),
        "respiration_rate": news2_reference("rr", row.get("rr")),
        "body_temp": news2_reference("temp", row.get("body_temp")),
    }

    # Fusion
    state = StateObject(
        window_id=row.get("window_id", ""),
        timestamp=ref_ts,
        heart_rate=row.get("hr"),
        respiration_rate=row.get("rr"),
        body_temp=row.get("body_temp"),
        wifi_confidence=row.get("wifi_conf"),
        mmwave_confidence=row.get("mmwave_conf"),
        thermal_confidence=row.get("thermal_conf"),
        nlos_flag=bool(row.get("nlos_flag", False)),
        fall_status=row.get("fall_status"),
        activity_state=row.get("activity_state", "unknown"),
        hr_wifi=row.get("hr_wifi"),
        hr_mm=row.get("hr_mm"),
        rr_wifi=row.get("rr_wifi"),
        rr_mm=row.get("rr_mm"),
        rr_conf=row.get("rr_conf"),
        hr_conf=row.get("hr_conf"),
        rr_source=row.get("rr_source"),
        hr_source=row.get("hr_source"),
    )
    engine = FusionEngine()
    fusion_hr = engine.fuse(state, "hr")

    has_pm = state.hr_wifi is not None or state.rr_wifi is not None

    # Latest action
    latest_action = None
    episodes = query_episodes_by_resident("resident_01", 1)
    if episodes:
        ep = episodes[0]
        if isinstance(ep.get("decision"), str):
            ep["decision"] = json.loads(ep["decision"])
        if isinstance(ep.get("action"), str):
            ep["action"] = json.loads(ep["action"])
        lvl = ep.get("decision", {}).get("level", "L0")
        ch = ep.get("action", {}).get("channel", "none")
        latest_action = {"level": lvl, "channel": ch, "episode_id": ep.get("episode_id")}

    # Compute data freshness (strip tz for naive comparison)
    now_naive = datetime.now()
    if ref_ts.tzinfo is not None:
        ref_ts_naive = ref_ts.replace(tzinfo=None)
    else:
        ref_ts_naive = ref_ts

    return {
        "status": "ok",
        "timestamp": row.get("timestamp"),
        "reference_timestamp": ref_ts.isoformat(),
        "source": row.get("source", "unknown"),
        "data_freshness_sec": int((now_naive - ref_ts_naive).total_seconds()),

        "heart_rate": row.get("hr"),
        "respiration_rate": row.get("rr"),
        "body_temp": row.get("body_temp"),

        "baseline": baseline or None,
        "absolute_reference": absolute_reference,

        "has_per_modality": has_pm,
        "hr_wifi": state.hr_wifi,
        "hr_mm": state.hr_mm,
        "rr_wifi": state.rr_wifi,
        "rr_mm": state.rr_mm,
        "hr_conf": state.hr_conf,
        "rr_conf": state.rr_conf,
        "hr_source": state.hr_source,
        "rr_source": state.rr_source,

        "quality": {
            "wifi_confidence": row.get("wifi_conf"),
            "mmwave_confidence": row.get("mmwave_conf"),
            "thermal_confidence": row.get("thermal_conf"),
            "hr_conf": row.get("hr_conf"),
            "rr_conf": row.get("rr_conf"),
            "nlos_flag": bool(row.get("nlos_flag", False)),
            "quality_event": bool(row.get("quality_event", 0)),
            "missing_modalities": json.loads(row.get("missing_mods", "[]")) if isinstance(row.get("missing_mods"), str) else (row.get("missing_mods") or []),
            "has_per_modality": has_pm,
        },

        "activity_state": row.get("activity_state", "unknown"),
        "fall_status": row.get("fall_status"),

        "fusion": {
            "metric": fusion_hr.metric,
            "estimates": fusion_hr.estimates,
            "checks": fusion_hr.checks,
            "verdict": fusion_hr.verdict,
        } if fusion_hr else None,

        "latest_action": latest_action,
    }


@app.get("/api/episode/{episode_id}")
async def api_episode_detail(episode_id: str):
    """单条诊断记录完整 JSON"""
    from storage.models import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM episode_logs WHERE episode_id=?", (episode_id,)
        ).fetchone()
    if not row:
        return {"status": "not_found"}
    row = dict(row)
    import json
    for key in ("decision", "evidence", "action", "audit"):
        if isinstance(row.get(key), str):
            row[key] = json.loads(row[key])
    # Attach the original health_event for richer context
    event_row = None
    if row.get("event_id"):
        with get_db() as conn2:
            event_row = conn2.execute(
                "SELECT * FROM health_events WHERE event_id=?", (row["event_id"],)
            ).fetchone()
        if event_row:
            event_row = dict(event_row)
            if isinstance(event_row.get("rule_markers"), str):
                event_row["rule_markers"] = json.loads(event_row["rule_markers"])
    return {
        "status": "ok",
        "episode": row,
        "health_event": event_row,
    }


@app.get("/episodes", response_class=HTMLResponse)
async def episodes_page(request: Request):
    """诊断事件列表页面"""
    return templates.TemplateResponse("episodes.html", {
        "request": request,
    })


@app.get("/episode/{episode_id}", response_class=HTMLResponse)
async def episode_detail_page(request: Request, episode_id: str):
    """诊断详情 HTML 页"""
    import json

    # Fetch episode
    from storage.models import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM episode_logs WHERE episode_id=?", (episode_id,)
        ).fetchone()
    if not row:
        return HTMLResponse("<h1>Not Found</h1>", status_code=404)
    ep = dict(row)
    for key in ("decision", "evidence", "action", "audit"):
        if isinstance(ep.get(key), str):
            ep[key] = json.loads(ep[key])

    # Fetch health_event
    health_event = None
    if ep.get("event_id"):
        with get_db() as conn2:
            he = conn2.execute(
                "SELECT * FROM health_events WHERE event_id=?", (ep["event_id"],)
            ).fetchone()
        if he:
            health_event = dict(he)
            if isinstance(health_event.get("rule_markers"), str):
                health_event["rule_markers"] = json.loads(health_event["rule_markers"])

    return templates.TemplateResponse("episode_detail.html", {
        "request": request,
        "episode": ep,
        "health_event": health_event,
    })


@app.get("/report", response_class=HTMLResponse)
async def report_page(request: Request):
    """周报 — 真实 7 天窗口，区分最高风险级别与最高频级别。"""
    from datetime import timedelta

    episodes = query_episodes_by_resident("resident_01", 200)
    for ep in episodes:
        for key in ("decision", "evidence", "action", "audit"):
            if isinstance(ep.get(key), str):
                try:
                    ep[key] = json.loads(ep[key])
                except (json.JSONDecodeError, TypeError):
                    pass

    # Shared reference timestamp: latest episode start_time or now
    ref_ts = datetime.now()
    if episodes and episodes[0].get("start_time"):
        try:
            ref_ts = datetime.fromisoformat(str(episodes[0]["start_time"]))
        except (ValueError, TypeError):
            pass
    week_ago = (ref_ts - timedelta(days=7)).isoformat()
    ref_iso = ref_ts.isoformat()

    # Filter episodes to 7-day window
    week_episodes = [ep for ep in episodes
                     if str(ep.get("start_time", "")) >= week_ago]

    # Levels
    level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
    highest_risk = "L0"
    label_by_level = {"L0": "记录", "L1": "观察", "L2": "提醒", "L3": "通知", "L4": "紧急"}
    for ep in week_episodes:
        lv = str(ep.get("decision", {}).get("level", "L0"))[:2]
        if lv in level_counts:
            level_counts[lv] += 1
            # Track highest (L4 > L3 > ...)
            if int(lv[1]) > int(highest_risk[1]):
                highest_risk = lv

    most_frequent_level = max(level_counts, key=level_counts.get) if any(level_counts.values()) else "L0"

    # Health events in same window
    events = []
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM health_events WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp DESC",
            (week_ago, ref_iso),
        ).fetchall()
    for r in rows:
        d = dict(r)
        if isinstance(d.get("rule_markers"), str):
            try:
                d["rule_markers"] = json.loads(d["rule_markers"])
            except (json.JSONDecodeError, TypeError):
                pass
        events.append(d)

    # Enrich with confidence from episode evidence
    event_dicts = []
    for e in events:
        ed = dict(e)
        ed["wifi_confidence"] = None
        ed["mmwave_confidence"] = None
        for ep in week_episodes:
            if ep.get("event_id") == ed.get("event_id"):
                evid = ep.get("evidence", {})
                if isinstance(evid, str):
                    try:
                        evid = json.loads(evid)
                    except (json.JSONDecodeError, TypeError):
                        evid = {}
                ss = evid.get("sensing_summary", {}) if isinstance(evid, dict) else {}
                ed["wifi_confidence"] = ss.get("wifi_confidence")
                ed["mmwave_confidence"] = ss.get("mmwave_confidence")
                break
        event_dicts.append(ed)

    report_text = ReportAgent().generate_weekly_report(
        [type('Ep', (object,), {'start_time': ref_ts, 'decision': ep.get('decision', {})})() for ep in week_episodes],
        events=event_dicts,
    )

    nlos_count = sum(1 for e in events if e.get("nlos_flag") or e.get("event_type") == "nlos_occlusion")
    low_conf_count = sum(1 for e in events if e.get("event_type") == "low_confidence")
    conflict_count = sum(1 for e in events if e.get("event_type") == "modality_conflict")

    # Event type breakdown
    type_counts = {}
    for e in events:
        et = e.get("event_type", "unknown")
        type_counts[et] = type_counts.get(et, 0) + 1
    event_type_breakdown = [{"type": k, "count": v} for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]

    # Action channel breakdown
    channel_counts = {}
    for ep in week_episodes:
        ch = ep.get("action", {}).get("channel", "unknown") if isinstance(ep.get("action"), dict) else "unknown"
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    action_breakdown = [{"channel": k, "count": v} for k, v in sorted(channel_counts.items(), key=lambda x: -x[1])]

    return templates.TemplateResponse("report.html", {
        "request": request,
        "report_text": report_text,
        "total_episodes": len(week_episodes),
        "level_counts": level_counts,
        "highest_risk_level": highest_risk,
        "highest_risk_label": label_by_level.get(highest_risk, ""),
        "most_frequent_level": most_frequent_level,
        "most_frequent_label": label_by_level.get(most_frequent_level, ""),
        "nlos_count": nlos_count,
        "low_conf_count": low_conf_count,
        "conflict_count": conflict_count,
        "event_type_breakdown": event_type_breakdown,
        "action_breakdown": action_breakdown,
        "reference_date": ref_ts.strftime("%Y-%m-%d %H:%M"),
    })
