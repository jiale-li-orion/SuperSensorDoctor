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
)
from storage.db import DB_PATH
from sensing_simulator.replay_engine import ReplayEngine
from sensing_simulator.sensor_hub import SensorHub

app = FastAPI(title="SuperSenseDoctor", version="0.1.0")
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """医生工作站首页"""
    windows = query_recent_windows("resident_01", "heart_rate", 60)
    pending = query_pending_events()
    episodes = query_episodes_by_resident("resident_01", 20)

    # 将数据库中 JSON 字符串反序列化为 dict, 供模板直接访问字段
    import json
    for ep in episodes:
        for key in ("decision", "evidence", "action", "audit"):
            if isinstance(ep.get(key), str):
                ep[key] = json.loads(ep[key])

    hr_values = [float(w["value"]) for w in windows if w.get("value") is not None]
    avg_hr = round(sum(hr_values) / len(hr_values), 1) if hr_values else None

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_windows": len(windows),
        "pending_events": len(pending),
        "total_episodes": len(episodes),
        "avg_heart_rate": avg_hr,
        "recent_episodes": episodes[:10],
        "pending_events_list": pending[:10],
        "db_path": str(DB_PATH),
    })


@app.post("/api/replay/start")
async def start_replay():
    """启动合成数据回放"""
    hub = SensorHub()
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
                "fall_status": state.fall_status,
                "source": "synthetic",
            },
        )
        count += 1
    return {"status": "ok", "windows_created": count}


@app.get("/api/health")
async def health():
    return {"status": "ok", "db": str(DB_PATH)}
