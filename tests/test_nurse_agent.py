import pytest
from datetime import datetime, timedelta
from agent_layer.state_objects import StateObject, HealthEvent
from agent_layer.nurse_agent import NurseAgent
from agent_layer.event_bus import EventBus
from agent_layer.confidence import estimate_wifi_confidence


class TestConfidence:
    def test_missing_data_zero(self):
        s = StateObject(window_id="w1", timestamp=datetime.now())
        assert estimate_wifi_confidence(s) == 0.0

    def test_normal_data_high_confidence(self):
        s = StateObject(
            window_id="w1", timestamp=datetime.now(),
            heart_rate=72.0, respiration_rate=16.0,
        )
        c = estimate_wifi_confidence(s)
        assert c > 0.8

    def test_abnormal_hr_lowers_confidence(self):
        s = StateObject(
            window_id="w1", timestamp=datetime.now(),
            heart_rate=200.0,  # beyond physiological range
        )
        assert estimate_wifi_confidence(s) < 0.8


class TestNurseAgent:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def agent(self, bus):
        return NurseAgent(event_bus=bus)

    @pytest.mark.asyncio
    async def test_normal_state_no_event(self, agent, bus):
        events = []
        @bus.subscribe("*")
        def handler(e):
            events.append(e)

        s = StateObject(
            window_id="w001", timestamp=datetime.now(),
            heart_rate=72.0, body_temp=36.5,
        )
        await agent.evaluate(s)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_high_hr_triggers_event(self, agent, bus):
        events = []
        @bus.subscribe("hr_abnormal")
        def handler(e):
            events.append(e)

        s = StateObject(
            window_id="w002", timestamp=datetime.now(),
            heart_rate=120.0, body_temp=36.5,
        )
        await agent.evaluate(s)
        assert len(events) == 1
        assert events[0].event_type == "hr_abnormal"

    @pytest.mark.asyncio
    async def test_fall_triggers_l4_event(self, agent, bus):
        events = []
        @bus.subscribe("fall_detected")
        def handler(e):
            events.append(e)

        s = StateObject(
            window_id="w003", timestamp=datetime.now(),
            fall_status="fall", heart_rate=110.0,
        )
        await agent.evaluate(s)
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_fall_no_physiological_change(self, agent, bus):
        events = []
        @bus.subscribe("fall_no_physiological_change")
        def handler(e):
            events.append(e)

        s = StateObject(
            window_id="w004", timestamp=datetime.now(),
            fall_status="fall", heart_rate=72.0, body_temp=36.5,
        )
        await agent.evaluate(s)
        assert len(events) == 1
        assert events[0].event_type == "fall_no_physiological_change"

    @pytest.mark.asyncio
    async def test_temp_abnormal(self, agent, bus):
        events = []
        @bus.subscribe("temp_abnormal")
        def handler(e):
            events.append(e)

        s = StateObject(
            window_id="w005", timestamp=datetime.now(),
            body_temp=38.5,
        )
        await agent.evaluate(s)
        assert len(events) == 1
        assert events[0].event_type == "temp_abnormal"

    @pytest.mark.asyncio
    async def test_multiple_events(self, agent, bus):
        events = []
        @bus.subscribe("*")
        def handler(e):
            events.append(e)

        s = StateObject(
            window_id="w006", timestamp=datetime.now(),
            heart_rate=120.0, body_temp=39.0, fall_status="fall",
        )
        await agent.evaluate(s)
        assert len(events) >= 2  # at least fall + hr_abnormal
        event_types = {e.event_type for e in events}
        assert "fall_detected" in event_types
        assert "hr_abnormal" in event_types
