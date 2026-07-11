"""SuperSenseDoctor — Agent Layer 启动入口"""

import asyncio
import yaml
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from storage.db import init_db
from storage.models import insert_sensing_window, insert_health_event, mark_event_handled
from agent_layer.event_bus import EventBus
from agent_layer.nurse_agent import NurseAgent
from agent_layer.diagnosis_agent import DiagnosisAgent
from agent_layer.report_agent import ReportAgent
from agent_layer.llm_provider import DeepSeekProvider
from agent_layer.tools import create_default_tools
from agent_layer.baseline_provider import BaselineProvider
from sensing_simulator.replay_engine import ReplayEngine
from sensing_simulator.sensor_hub import SensorHub

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def create_app(config: dict = None) -> FastAPI:
    """创建 FastAPI 应用并初始化 Agent 层"""
    if config is None:
        config = load_config()

    # 初始化数据库
    init_db()

    # 初始化 Event Bus
    bus = EventBus()

    # 初始化 Agent 层
    nurse = NurseAgent(
        event_bus=bus,
        hr_deviation=config["agents"]["nurse"]["threshold_hr_deviation"],
        temp_deviation=config["agents"]["nurse"]["threshold_temp_deviation"],
        baseline_provider=BaselineProvider(),
    )

    # LLM Provider
    llm_config = config["llm"]
    llm = DeepSeekProvider(
        api_key=os.getenv("DEEPSEEK_API_KEY", llm_config.get("api_key", "")),
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        temperature=llm_config["temperature"],
    )

    tools = create_default_tools()
    diagnosis = DiagnosisAgent(
        resident_id="resident_01",
        llm_provider=llm,
        tool_registry=tools,
        max_steps=config["agents"]["diagnosis"]["max_steps"],
    )

    sensor_hub = SensorHub(nurse=nurse)

    # 注册 Agent 到 Event Bus
    @bus.subscribe("*")
    def on_event(event):
        print(f"[EventBus] {event.event_type}: {event.trigger_reason}")

    @bus.subscribe("hr_abnormal")
    @bus.subscribe("fall_detected")
    @bus.subscribe("temp_abnormal")
    @bus.subscribe("rr_bradypnea")
    @bus.subscribe("rr_tachypnea")
    @bus.subscribe("low_confidence")
    @bus.subscribe("nlos_occlusion")
    @bus.subscribe("modality_conflict")
    @bus.subscribe("fall_no_physiological_change")
    async def on_diagnosis_event(event):
        """Nurse Agent 发布事件 → 持久化 health_event → Diagnosis Agent 处理"""
        # 先持久化 health_event 父行, 满足 episode_logs FK 约束
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
        print(f"[Diagnosis] {result.episode_id}: L{result.decision.get('level', '?')}")

    # Import and return FastAPI app
    from web.app import app as fastapi_app
    fastapi_app.state.sensor_hub = sensor_hub     # Inject sensor_hub
    fastapi_app.state.diagnosis_agent = diagnosis  # Inject diagnosis for health checks
    return fastapi_app


def main():
    config = load_config()
    print(f"SuperSenseDoctor Agent Layer starting...")
    print(f"  LLM: {config['llm']['provider']}/{config['llm']['model']}")
    print(f"  DB: {config['storage']['db_path']}")

    app = create_app(config)
    uvicorn.run(
        app,
        host=config["web"]["host"],
        port=config["web"]["port"],
    )


if __name__ == "__main__":
    main()
