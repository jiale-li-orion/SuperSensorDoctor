import os
from datetime import datetime

import pytest

from storage.db import DB_PATH
from storage.models import query_episodes_by_resident


@pytest.fixture(autouse=True)
def clean_db():
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass
    yield
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


@pytest.mark.asyncio
async def test_modality_conflict_reaches_diagnosis(monkeypatch):
    import main

    class MockLLM:
        def __init__(self, *args, **kwargs):
            pass

        async def chat(self, messages):
            return type("Resp", (), {"content": '{"level":"L1"}', "tool_calls": None})()

        async def chat_with_tools(self, messages, schema):
            return type("Resp", (), {"content": '{"level":"L1"}', "tool_calls": None})()

    monkeypatch.setattr(main, "DeepSeekProvider", MockLLM)

    config = main.load_config()
    app = main.create_app(config)
    hub = app.state.sensor_hub

    await hub.compose(
        "main_conflict",
        datetime.now(),
        {
            "heart_rate": 75.0,
            "hr_wifi": 70.0,
            "hr_mm": 82.0,
            "wifi_confidence": 0.9,
            "mmwave_confidence": 0.9,
            "hr_conf": 0.95,
        },
    )

    episodes = query_episodes_by_resident("resident_01")
    assert len(episodes) == 1
    assert episodes[0]["event_id"].startswith("modality_conflict_")
