"""FastAPI Web 入口 — 医生工作站"""

from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from storage.models import (
    query_recent_windows,
    query_pending_events,
    query_episodes_by_resident,
    query_latest_sensing_window,
)
from storage.db import DB_PATH
from sensing_simulator.replay_engine import ReplayEngine
from sensing_simulator.sensor_hub import SensorHub
from agent_layer.baseline_provider import BaselineProvider
from agent_layer.fusion_engine import FusionEngine
from agent_layer.state_objects import StateObject
from agent_layer.report_agent import ReportAgent

app = FastAPI(title="SuperSenseDoctor", version="0.1.0")
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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
    })


@app.post("/api/replay/start")
async def start_replay(request: Request):
    """启动合成数据回放 — 通过 app.state.sensor_hub 进入完整 Agent 链路"""
    hub = getattr(request.app.state, "sensor_hub", None) or SensorHub()
    count = 0
    for state in ReplayEngine.generate_synthetic(duration_sec=30):
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
        count += 1
    return {"status": "ok", "windows_created": count}


@app.get("/api/health")
async def health():
    return {"status": "ok", "db": str(DB_PATH)}


@app.get("/api/vitals/latest")
async def api_vitals_latest():
    """最新体征快照 + 基线 + 融合信息"""
    from datetime import datetime
    row = query_latest_sensing_window("resident_01")
    if not row:
        return {"status": "no_data"}

    # Baseline z_scores
    bp = BaselineProvider()
    hr_bl = bp.compute_metric("resident_01", "hr", row.get("hr"), datetime.now()) if row.get("hr") else None
    temp_bl = bp.compute_metric("resident_01", "temp", row.get("body_temp"), datetime.now()) if row.get("body_temp") else None

    # Fusion
    state = StateObject(
        window_id=row.get("window_id", ""),
        timestamp=datetime.now(),
        heart_rate=row.get("hr"),
        respiration_rate=row.get("rr"),
        body_temp=row.get("body_temp"),
        wifi_confidence=row.get("wifi_conf", 0.0) or 0.0,
        mmwave_confidence=row.get("mmwave_conf", 0.0) or 0.0,
        thermal_confidence=row.get("thermal_conf", 0.0) or 0.0,
        nlos_flag=bool(row.get("nlos_flag", False)),
        fall_status=row.get("fall_status"),
        activity_state=row.get("activity_state", "unknown"),
    )
    engine = FusionEngine()
    fusion_hr = engine.fuse(state, "hr")

    return {
        "status": "ok",
        "timestamp": row.get("timestamp"),
        "heart_rate": row.get("hr"),
        "respiration_rate": row.get("rr"),
        "body_temp": row.get("body_temp"),
        "hr_z_score": round(hr_bl["z_score"], 2) if hr_bl else None,
        "temp_z_score": round(temp_bl["z_score"], 2) if temp_bl else None,
        "wifi_confidence": row.get("wifi_conf"),
        "mmwave_confidence": row.get("mmwave_conf"),
        "nlos_flag": bool(row.get("nlos_flag", False)),
        "activity_state": row.get("activity_state", "unknown"),
        "fall_status": row.get("fall_status"),
        "fusion": {
            "dominant": fusion_hr.verdict.get("dominant_modality"),
            "fused_value": fusion_hr.verdict.get("fused_value"),
            "rationale": fusion_hr.verdict.get("rationale"),
        } if fusion_hr else None,
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
    """周报 HTML 页"""
    from storage.models import get_db
    from datetime import datetime, timedelta

    episodes = query_episodes_by_resident("resident_01", 200)
    import json
    for ep in episodes:
        for key in ("decision", "evidence", "action", "audit"):
            if isinstance(ep.get(key), str):
                ep[key] = json.loads(ep[key])

    # Collect events for ReportAgent
    events = []
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM health_events WHERE handled=1 ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
    for r in rows:
        d = dict(r)
        if isinstance(d.get("rule_markers"), str):
            d["rule_markers"] = json.loads(d["rule_markers"])
        events.append(d)

    # Build event dicts from health_events DB rows + episode evidence
    event_dicts = []
    for e in events:
        ed = dict(e)
        # Already json-parsed rule_markers above
        # Inject confidence from matching episode's evidence
        ed["wifi_confidence"] = None
        ed["mmwave_confidence"] = None
        for ep in episodes:
            if ep.get("event_id") == ed.get("event_id"):
                evidence = ep.get("evidence", {})
                if isinstance(evidence, str):
                    evidence = json.loads(evidence)
                ss = evidence.get("sensing_summary", {}) if isinstance(evidence, dict) else {}
                ed["wifi_confidence"] = ss.get("wifi_confidence")
                ed["mmwave_confidence"] = ss.get("mmwave_confidence")
                break
        event_dicts.append(ed)

    report_text = ReportAgent().generate_weekly_report(
        [type('Ep', (object,), {'start_time': datetime.now(), 'decision': ep.get('decision', {})})() for ep in episodes],
        events=event_dicts,
    )

    # Count stats from events
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    nlos_count = sum(1 for e in events if e.get("nlos_flag") or e.get("event_type") == "nlos_occlusion")
    low_conf_count = sum(1 for e in events if e.get("event_type") == "low_confidence")
    conflict_count = sum(1 for e in events if e.get("event_type") == "modality_conflict")

    level_counts = {"L0":0,"L1":0,"L2":0,"L3":0,"L4":0}
    for ep in episodes:
        lv = ep.get("decision", {}).get("level", "L0")[:2]
        if lv in level_counts:
            level_counts[lv] += 1

    return templates.TemplateResponse("report.html", {
        "request": request,
        "report_text": report_text,
        "total_episodes": len(episodes),
        "level_counts": level_counts,
        "nlos_count": nlos_count,
        "low_conf_count": low_conf_count,
        "conflict_count": conflict_count,
    })
