import pytest
from datetime import datetime
from agent_layer.event_bus import EventBus
from agent_layer.state_objects import HealthEvent, StateObject


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def sample_event():
    s = StateObject(window_id="w1", timestamp=datetime.now())
    return HealthEvent(
        event_id="evt_001", event_type="hr_abnormal",
        timestamp=datetime.now(), state=s,
        trigger_reason="test",
    )


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_wildcard(self, bus, sample_event):
        received = []

        @bus.subscribe("hr_abnormal")
        def handler(event):
            received.append(event)

        await bus.publish(sample_event)
        assert len(received) == 1
        assert received[0].event_id == "evt_001"

    @pytest.mark.asyncio
    async def test_subscribe_wildcard_all(self, bus, sample_event):
        received = []

        @bus.subscribe("*")
        def handler(event):
            received.append(event)

        await bus.publish(sample_event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_no_match_is_safe(self, bus, sample_event):
        received = []

        @bus.subscribe("temp_abnormal")
        def handler(event):
            received.append(event)

        await bus.publish(sample_event)
        assert len(received) == 0


def test_subscribe_too_many(bus):
    with pytest.raises(RuntimeError, match="Max 30 subscribers"):
        for i in range(35):
            @bus.subscribe(f"event_{i}")
            def h(event):
                pass
    assert len(bus._subscribers) == 30  # should cap at 30
