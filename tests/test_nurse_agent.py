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


# ---------------------------------------------------------------------------
# NurseAgent z_score path tests (Phase E)
# ---------------------------------------------------------------------------

class TestNurseAgentZScore:
    """Tests for the BaselineProvider z_score integration in NurseAgent."""

    def setup_method(self):
        """Seed baseline data with known statistics."""
        from storage.db import init_db, get_db
        from storage.models import insert_sensing_window
        init_db()
        # Clear any data from the autouse fixture in TestNurseAgent
        with get_db() as conn:
            conn.execute("DELETE FROM health_events")
            conn.execute("DELETE FROM sensing_windows")
        # 30 values centered around 75
        for i in range(30):
            ts = datetime.now() - timedelta(hours=i)
            hr = 75.0 + (i % 11 - 5) * 0.5  # range ~72.5-77.5
            temp = 36.5 + (i % 5 - 2) * 0.1  # range ~36.3-36.7
            insert_sensing_window(
                window_id=f"ztest_{i}", timestamp=ts,
                heart_rate=hr, body_temp=temp,
                wifi_confidence=0.9, mmwave_confidence=0.85,
            )

    @pytest.mark.asyncio
    async def test_hr_z_score_triggers_abnormal(self):
        """HR far from personal baseline (z > 2) should trigger hr_abnormal."""
        from agent_layer.baseline_provider import BaselineProvider
        bus = EventBus()
        bp = BaselineProvider()
        agent = NurseAgent(event_bus=bus, baseline_provider=bp, z_threshold=2.0)
        events = []
        bus.subscribe('hr_abnormal')(lambda e: events.append(e))

        s = StateObject('z1', datetime.now(), heart_rate=90.0)
        await agent.evaluate(s)
        assert len(events) == 1, f"Expected hr_abnormal event, got {len(events)}"
        assert 'hr_z_score' in events[0].rule_markers, "Missing hr_z_score in markers"

    @pytest.mark.asyncio
    async def test_hr_normal_does_not_trigger(self):
        """HR near personal baseline (z < 2) should NOT trigger."""
        from agent_layer.baseline_provider import BaselineProvider
        bus = EventBus()
        bp = BaselineProvider()
        agent = NurseAgent(event_bus=bus, baseline_provider=bp, z_threshold=2.0)
        events = []
        bus.subscribe('hr_abnormal')(lambda e: events.append(e))

        s = StateObject('z2', datetime.now(), heart_rate=75.5)
        await agent.evaluate(s)
        assert len(events) == 0, f"Expected 0 events for normal HR, got {len(events)}"

    @pytest.mark.asyncio
    async def test_temp_z_score_triggers_abnormal(self):
        """Temp far from personal baseline should trigger temp_abnormal."""
        from agent_layer.baseline_provider import BaselineProvider
        bus = EventBus()
        bp = BaselineProvider()
        agent = NurseAgent(event_bus=bus, baseline_provider=bp, z_threshold=2.0)
        events = []
        bus.subscribe('temp_abnormal')(lambda e: events.append(e))

        s = StateObject('z3', datetime.now(), body_temp=38.0)
        await agent.evaluate(s)
        assert len(events) == 1, f"Expected temp_abnormal event, got {len(events)}"
        assert 'temp_z_score' in events[0].rule_markers, "Missing temp_z_score in markers"

    @pytest.mark.asyncio
    async def test_z_score_fallback_to_deviation(self):
        """Without baseline_provider, rules still fire via hard deviation."""
        bus = EventBus()
        agent = NurseAgent(event_bus=bus, baseline_provider=None)
        events = []
        bus.subscribe('hr_abnormal')(lambda e: events.append(e))

        s = StateObject('z4', datetime.now(), heart_rate=120.0)
        await agent.evaluate(s)
        assert len(events) == 1, f"Expected fallback trigger, got {len(events)}"

    @pytest.mark.asyncio
    async def test_no_false_positive_when_z_score_below_threshold(self):
        """Value within normal range should not trigger even with baseline provider."""
        from agent_layer.baseline_provider import BaselineProvider
        bus = EventBus()
        bp = BaselineProvider()
        agent = NurseAgent(event_bus=bus, baseline_provider=bp, z_threshold=2.0)
        events = []
        bus.subscribe('*')(lambda e: events.append(e))

        s = StateObject('z5', datetime.now(), heart_rate=74.5)
        await agent.evaluate(s)
        hr_events = [e for e in events if e.event_type == 'hr_abnormal']
        assert len(hr_events) == 0, "Should not trigger for normal HR"

    @pytest.mark.asyncio
    async def test_duration_tracker(self):
        """Duration should accumulate across multiple evaluate() calls."""
        from agent_layer.baseline_provider import BaselineProvider
        bus = EventBus()
        bp = BaselineProvider()
        agent = NurseAgent(event_bus=bus, baseline_provider=bp, z_threshold=2.0)
        events = []
        bus.subscribe('hr_abnormal')(lambda e: events.append(e))

        now = datetime.now()
        s1 = StateObject('zd1', now, heart_rate=92.0)
        await agent.evaluate(s1)
        assert len(events) == 1
        dur1 = events[0].rule_markers.get('duration_sec', 0)

        s2 = StateObject('zd2', now + timedelta(seconds=60), heart_rate=95.0)
        await agent.evaluate(s2)
        assert len(events) == 2
        dur2 = events[-1].rule_markers.get('duration_sec', 0)
        assert dur2 >= dur1 or dur2 > 0, f"Duration should accumulate: {dur1} -> {dur2}"

    @pytest.mark.asyncio
    async def test_duration_tracker_uses_event_time_for_historical_replay(self):
        """Old replay timestamps should accumulate by event time, not wall-clock now."""
        bus = EventBus()
        agent = NurseAgent(event_bus=bus, baseline_provider=None)
        events = []
        bus.subscribe('hr_abnormal')(lambda e: events.append(e))

        historical = datetime(2026, 7, 9, 10, 0, 0)
        await agent.evaluate(StateObject('old1', historical, heart_rate=120.0))
        await agent.evaluate(StateObject('old2', historical + timedelta(seconds=60), heart_rate=121.0))

        assert len(events) == 2
        assert events[-1].rule_markers.get('duration_sec', 0) >= 60


# ---------------------------------------------------------------------------
# NurseAgent Rule 8: Modality conflict tests (Phase F)
# ---------------------------------------------------------------------------

class TestNurseAgentModalityConflict:
    """Tests for Rule 8 modality conflict detection."""

    @pytest.mark.asyncio
    async def test_rule_8_does_not_crash_with_normal_data(self):
        """Rule 8 should not crash even when no conflict exists."""
        from agent_layer.event_bus import EventBus
        bus = EventBus()
        agent = NurseAgent(event_bus=bus)
        events = []
        bus.subscribe("modality_conflict")(lambda e: events.append(e))

        state = StateObject("mct1", datetime.now(),
            heart_rate=72.0, respiration_rate=16.0,
            wifi_confidence=0.9, mmwave_confidence=0.85)
        await agent.evaluate(state)
        # Normal data should not trigger modality conflict
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_rule_8_no_vitals_no_crash(self):
        """Rule 8 should not crash when vitals are None."""
        from agent_layer.event_bus import EventBus
        bus = EventBus()
        agent = NurseAgent(event_bus=bus)
        events = []
        bus.subscribe("modality_conflict")(lambda e: events.append(e))

        state = StateObject("mct2", datetime.now(),
            heart_rate=None, respiration_rate=None)
        await agent.evaluate(state)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_rule_8_real_per_modality_conflict(self):
        """Real WiFi/mmWave portable_v2 estimates should trigger conflict."""
        bus = EventBus()
        agent = NurseAgent(event_bus=bus)
        events = []
        bus.subscribe("modality_conflict")(lambda e: events.append(e))

        state = StateObject(
            "mct3", datetime.now(),
            heart_rate=75.0,
            hr_wifi=70.0,
            hr_mm=82.0,
            wifi_confidence=0.9,
            mmwave_confidence=0.9,
            hr_conf=0.95,
        )
        await agent.evaluate(state)
        assert len(events) == 1
        assert events[0].event_type == "modality_conflict"
        assert events[0].rule_markers["hr_modality_delta"] == 12.0
