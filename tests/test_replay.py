import pytest
from datetime import datetime
from sensing_simulator.replay_engine import ReplayEngine
from sensing_simulator.sensor_hub import SensorHub
from sensing_simulator.sensor_aligner import SensorAligner
from agent_layer.state_objects import StateObject
from storage.db import init_db, DB_PATH


@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    yield
    import os
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


class TestReplayEngine:
    def test_generates_correct_count(self):
        states = list(ReplayEngine.generate_synthetic(duration_sec=30))
        assert len(states) == 30

    def test_contains_abnormal_hr(self):
        states = list(ReplayEngine.generate_synthetic(duration_sec=45))
        hr_values = [s.heart_rate for s in states]
        assert any(h > 95 for h in hr_values), "should have elevated HR"


class TestSensorHub:
    @pytest.mark.asyncio
    async def test_compose_no_nurse(self):
        init_db()
        hub = SensorHub()
        await hub.compose(
            window_id="test_w1",
            timestamp=datetime.now(),
            data={"heart_rate": 72.0},
        )
        # 不抛异常即通过

    @pytest.mark.asyncio
    async def test_compose_with_nurse(self):
        init_db()
        from agent_layer.nurse_agent import NurseAgent
        from agent_layer.event_bus import EventBus

        bus = EventBus()
        nurse = NurseAgent(event_bus=bus)
        hub = SensorHub(nurse=nurse)

        events = []
        @bus.subscribe("hr_abnormal")
        def handler(e):
            events.append(e)

        await hub.compose(
            window_id="test_w2",
            timestamp=datetime.now(),
            data={"heart_rate": 130.0},
        )
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_compose_evaluate_false_only_persists(self):
        init_db()
        from agent_layer.nurse_agent import NurseAgent
        from agent_layer.event_bus import EventBus
        from storage.models import query_latest_sensing_window

        bus = EventBus()
        events = []
        bus.subscribe("*")(lambda e: events.append(e))
        hub = SensorHub(nurse=NurseAgent(event_bus=bus))

        await hub.compose(
            window_id="ingest_only",
            timestamp=datetime.now(),
            data={"heart_rate": 130.0},
            evaluate=False,
        )
        latest = query_latest_sensing_window("resident_01")
        assert latest["window_id"] == "ingest_only"
        assert latest["hr"] == 130.0
        assert events == []


class TestSensorAligner:
    def test_align_hr_only(self):
        aligner = SensorAligner(window_sec=1.0)
        now = datetime.now()
        hr_rows = [
            {"timestamp": now, "heart_rate": 72.0, "wifi_conf": 0.9},
        ]
        result = aligner.align(hr_rows=hr_rows)
        assert len(result) >= 1

    def test_align_merge(self):
        aligner = SensorAligner(window_sec=1.0)
        now = datetime.now()
        hr_rows = [
            {"timestamp": now, "heart_rate": 72.0, "wifi_conf": 0.9},
        ]
        fall_rows = [
            {"timestamp": now, "body_temp": 36.5, "fall_status": "no_fall"},
        ]
        result = aligner.align(hr_rows=hr_rows, fall_rows=fall_rows)
        assert len(result) >= 1
        merged = result[0]
        assert merged.heart_rate == 72.0
        assert merged.body_temp == 36.5
