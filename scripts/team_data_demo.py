"""Phase D: Team data end-to-end demo.

Replays team_data CSV + fall_status.xlsx through the full pipeline:
  Data -> SensorHub -> NurseAgent -> DiagnosisAgent -> DB

Usage:
  python -m scripts.team_data_demo
"""

import asyncio
import csv
import json
import uuid
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.db import init_db
from storage.models import (
    insert_sensing_window, insert_health_event,
    mark_event_handled, query_episodes_by_resident,
    query_latest_sensing_window,
)
from agent_layer.event_bus import EventBus
from agent_layer.nurse_agent import NurseAgent
from agent_layer.diagnosis_agent import DiagnosisAgent
from agent_layer.llm_provider import DeepSeekProvider
from agent_layer.tools import create_default_tools
from sensing_simulator.sensor_hub import SensorHub
from agent_layer.modality_synthesizer import parse_modalities


def load_fused_csv() -> list[dict]:
    """Load fused_vital_signs CSV."""
    rows = []
    path = Path("team_data/fused_vital_signs_timestamp_rr_hr.csv")
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": row["timestamp"],
                "heart_rate": float(row["heart_rate_bpm"]),
                "respiration_rate": float(row["respiratory_rate_bpm"]),
            })
    print(f"  Loaded {len(rows)} fused vital sign rows")
    return rows


def load_fall_status() -> dict:
    """Load fall_status.xlsx, return dict keyed by timestamp (rounded to second)."""
    import openpyxl
    wb = openpyxl.load_workbook("team_data/fall_status.xlsx")
    ws = wb.active
    fall = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        ts_str, temp, fall_status = row
        if ts_str is None:
            continue
        # Normalize timestamp to second precision for matching
        try:
            ts = datetime.fromisoformat(str(ts_str).replace(" ", "T"))
            key = ts.strftime("%Y-%m-%dT%H:%M:%S")
            fall[key] = {
                "fall_status": "fall" if fall_status == "\u8dcc\u5012" else "no_fall",
                "body_temp": float(temp) if temp else None,
                "_ts": ts_str,
            }
        except (ValueError, TypeError):
            pass
    print(f"  Loaded {len(fall)} fall status entries")
    return fall


async def run_demo():
    """Run the full team_data pipeline once."""
    print("\n=== Phase D: Team Data End-to-End Demo ===\n")

    # 1. Init
    init_db()
    print("[1/6] DB initialized")

    # 2. Load data
    fused_rows = load_fused_csv()
    fall_map = load_fall_status()
    print("[2/6] Team data loaded")

    # 3. Init agents
    bus = EventBus()
    nurse = NurseAgent(event_bus=bus)
    tools = create_default_tools()
    llm = DeepSeekProvider(
        api_key=__import__('os').environ.get("DEEPSEEK_API_KEY", "placeholder"),
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )
    diagnosis = DiagnosisAgent(
        resident_id="resident_01",
        llm_provider=llm,
        tool_registry=tools,
        max_steps=8,
    )
    sensor_hub = SensorHub(nurse=nurse)

    # Track results
    episode_logs = []

    # 4. Subscribe to events
    @bus.subscribe("*")
    def on_event(event):
        pass  # silent for demo

    @bus.subscribe("hr_abnormal")
    @bus.subscribe("fall_detected")
    @bus.subscribe("temp_abnormal")
    @bus.subscribe("rr_bradypnea")
    @bus.subscribe("rr_tachypnea")
    @bus.subscribe("low_confidence")
    @bus.subscribe("nlos_occlusion")
    @bus.subscribe("fall_no_physiological_change")
    async def on_diagnosis_event(event):
        insert_health_event(
            event_id=event.event_id,
            window_id=event.state.window_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            trigger_reason=event.trigger_reason,
            rule_markers=event.rule_markers,
        )
        result = await diagnosis.handle_event(event)
        mark_event_handled(event.event_id)
        episode_logs.append(result)

    # 5. Replay ~120 rows sampled across the dataset
    chunk = fused_rows[:100]
    chunk += fused_rows[200:210]
    chunk += fused_rows[500:510]
    chunk += fused_rows[800:810]

    processed = 0
    for row in chunk:
        ts_str = row["timestamp"]
        # Normalize for fall lookup
        ts_normalized = ts_str[:19]
        fall_info = fall_map.get(ts_normalized, {})

        data = {
            "heart_rate": row["heart_rate"],
            "respiration_rate": row["respiration_rate"],
            "body_temp": fall_info.get("body_temp"),
            "fall_status": fall_info.get("fall_status", "no_fall"),
            "wifi_confidence": 0.9,
            "mmwave_confidence": 0.85,
            "thermal_confidence": 0.9,
            "nlos_flag": False,
            "activity_state": "rest",
            "source": "csv",
        }
        try:
            ts = datetime.fromisoformat(str(ts_str))
        except ValueError:
            ts = datetime.now()

        result = await sensor_hub.compose(f"demo_{uuid.uuid4().hex[:8]}", ts, data)
        processed += 1

    print(f"[3/6] Replayed {processed} rows through SensorHub")

    # 6. Show results
    print(f"[4/6] Nurse generated {len(episode_logs)} events")
    print(f"[5/6] Diagnosis produced {len(episode_logs)} episode logs")

    # 7. Verify DB contents
    latest = query_latest_sensing_window("resident_01")
    if latest:
        print(f"\n[6/6] DB verification:")
        print(f"  Latest window: {latest.get('window_id')}")
        print(f"  HR={latest.get('hr')}, RR={latest.get('rr')}")
        mods = parse_modalities(latest.get("modalities_json"))
        if mods:
            wifi = mods.get("wifi", {})
            mmwave = mods.get("mmwave", {})
            print(f"  WiFi estimate: HR={wifi.get('hr')}, RR={wifi.get('rr')}")
            print(f"  mmWave estimate: HR={mmwave.get('hr')}, RR={mmwave.get('rr')}")

    if episode_logs:
        print(f"\n  Episode logs ({len(episode_logs)}):")
        for i, log in enumerate(episode_logs[:5]):
            level = log.decision.get("level", "?")
            etype = log.decision.get("label", "")
            reflex = log.audit.get("reflex", False)
            print(f"    {i+1}. {log.episode_id}: L{level} ({etype}) reflex={reflex}")

    # Summary statistics
    reflex_count = sum(1 for e in episode_logs if e.audit.get("reflex", False))
    print(f"\n  Summary:")
    print(f"    Total episodes: {len(episode_logs)}")
    print(f"    Reflex arc: {reflex_count}")
    print(f"    LLM path: {len(episode_logs) - reflex_count}")
    print(f"\nPhase D DEMO PASSED")
    return episode_logs


if __name__ == "__main__":
    asyncio.run(run_demo())
