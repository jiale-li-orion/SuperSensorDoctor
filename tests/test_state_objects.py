import pytest
from agent_layer.state_objects import StateObject, HealthEvent, EpisodeLog
from datetime import datetime


class TestStateObject:
    def test_default_fields_are_none(self):
        s = StateObject(window_id="w1", timestamp=datetime.now())
        assert s.heart_rate is None
        assert s.respiration_rate is None
        assert s.body_temp is None
        assert s.wifi_confidence is None
        assert s.missing_modalities == []
        assert s.source == "replay"

    def test_full_state(self):
        s = StateObject(
            window_id="w1",
            timestamp=datetime.now(),
            heart_rate=72.0,
            respiration_rate=16.0,
            body_temp=36.5,
            wifi_confidence=0.9,
            mmwave_confidence=0.85,
            nlos_flag=False,
            activity_state="rest",
            posture="lying",
        )
        assert s.heart_rate == 72.0
        assert s.nlos_flag is False


class TestHealthEvent:
    def test_event_creation(self):
        s = StateObject(window_id="w1", timestamp=datetime.now(), heart_rate=95.0)
        e = HealthEvent(
            event_id="evt_001",
            event_type="hr_abnormal",
            timestamp=datetime.now(),
            state=s,
            trigger_reason="HR exceeds baseline by 15 bpm",
            rule_markers={"deviation": 15, "duration_sec": 60},
        )
        assert e.state.heart_rate == 95.0
        assert e.rule_markers["deviation"] == 15


class TestEpisodeLog:
    def test_serializable(self):
        from dataclasses import asdict
        log = EpisodeLog(
            episode_id="ep_001", resident_id="resident_01",
            start_time=datetime.now(),
            evidence={"baseline": {"mean": 72}},
            decision={"level": "L1", "label": "observe"},
            action={"channel": "none"},
            audit={"tools_called": ["query_history"]},
        )
        d = asdict(log)
        assert d["evidence"]["baseline"]["mean"] == 72
        assert d["audit"]["tools_called"] == ["query_history"]
