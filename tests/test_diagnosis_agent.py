"""Tests for DiagnosisAgent Think/Act ReAct loop"""

import os
import pytest
from datetime import datetime
from agent_layer.state_objects import StateObject, HealthEvent
from agent_layer.llm_provider import MockProvider, ChatMessage
from agent_layer.tools import ToolRegistry, tool
from agent_layer.diagnosis_agent import DiagnosisAgent
from storage.db import init_db, DB_PATH
from storage.models import insert_sensing_window, insert_health_event


@pytest.fixture(autouse=True)
def clean_db():
    """Initialize the DB and create FK parent rows before each test."""
    init_db()
    # ── parent rows needed by write_episode_tool FK constraints ──
    insert_sensing_window(
        window_id="w1", timestamp=datetime.now(),
        heart_rate=72.0,
    )
    insert_health_event(
        event_id="evt_001", window_id="w1",
        event_type="hr_abnormal", timestamp=datetime.now(),
        trigger_reason="test seed",
    )
    yield
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


@pytest.fixture
def registry():
    r = ToolRegistry()

    @tool(name="ping", description="Ping", parameters={"x": {"type": "string"}})
    def ping(x: str) -> str:
        return f"pong:{x}"

    r.register(ping)
    return r


def make_hr_event(hr: float = 120.0) -> HealthEvent:
    s = StateObject(
        window_id="w1", timestamp=datetime.now(),
        heart_rate=hr, body_temp=36.5,
        wifi_confidence=0.9, mmwave_confidence=0.85,
    )
    return HealthEvent(
        event_id="evt_001", event_type="hr_abnormal",
        timestamp=datetime.now(), state=s,
        trigger_reason="HR deviation 15 bpm",
    )


class TestDiagnosisAgent:
    @pytest.mark.asyncio
    async def test_think_mode_returns_decision(self):
        """LLM returns a JSON decision → agent returns EpisodeLog with parsed decision"""
        decision_json = (
            '{"level": "L2", "explanation": "HR elevated vs baseline", '
            '"action_message": "提醒休息"}'
        )
        provider = MockProvider(response=decision_json)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            max_steps=8,
        )
        event = make_hr_event()
        log = await agent.handle_event(event)

        assert log.decision["level"] == "L2"
        assert "提醒" in log.decision.get("action_message", "")

    @pytest.mark.asyncio
    async def test_reuses_tool_result(self):
        """Tool call from LLM is executed and result integrated into context"""
        decision_json = '{"level": "L1", "explanation": "checked"}'
        provider = MockProvider(response=decision_json)

        r = ToolRegistry()
        calls = []

        @tool(name="test_tool", description="test", parameters={"q": {"type": "string"}})
        def test_tool(q: str) -> str:
            calls.append(q)
            return f"result:{q}"

        r.register(test_tool)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=r,
            max_steps=8,
        )
        event = make_hr_event()

        # MockProvider will return tool_calls via chat_with_tools
        provider._tool_calls = [{
            "id": "call_1", "type": "function",
            "function": {"name": "test_tool", "arguments": '{"q": "hr_72"}'},
        }]
        await agent.handle_event(event)

        assert len(calls) == 1
        assert calls[0] == "hr_72"

    @pytest.mark.asyncio
    async def test_max_steps_fallback_to_L0(self):
        """When LLM never returns a decision, agent falls back to L0"""
        provider = MockProvider(response="I'm thinking...")
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            max_steps=2,
        )
        event = make_hr_event()
        log = await agent.handle_event(event)

        assert log.decision["level"] == "L0"
        assert "max steps" in log.decision.get("explanation", "").lower()

    @pytest.mark.asyncio
    async def test_audit_includes_tools_and_steps(self):
        """EpisodeLog.audit contains tool names and step count"""
        decision_json = '{"level": "L3", "explanation": "sustained abnormality"}'
        provider = MockProvider(response=decision_json)

        r = ToolRegistry()

        @tool(name="query_history", description="hist", parameters={"m": {"type": "string"}})
        def query_history(m: str) -> str:
            return f"history:{m}"

        r.register(query_history)
        agent = DiagnosisAgent(
            resident_id="r2",
            llm_provider=provider,
            tool_registry=r,
            max_steps=3,
        )
        provider._tool_calls = [{
            "id": "call_x", "type": "function",
            "function": {"name": "query_history", "arguments": '{"m": "hr"}'},
        }]
        log = await agent.handle_event(make_hr_event())

        assert log.audit["step_count"] >= 1
        assert "query_history" in log.audit["tools_called"]
        assert log.audit["event_id"] == "evt_001"


class TestEvidenceChain:
    """Phase 5+7: evidence collection during DiagnosisAgent loop"""

    @pytest.mark.asyncio
    async def test_episode_contains_evidence_chain(self):
        """EpisodeLog.evidence contains event, sensing_summary, tool_results"""
        decision_json = '{"level": "L1", "explanation": "checking evidence chain"}'
        provider = MockProvider(response=decision_json)

        r = ToolRegistry()
        @tool(name="query_history", description="hist", parameters={"m": {"type": "string"}})
        def query_history(m: str) -> str:
            return {"status": "ok", "mean": 72.0}

        r.register(query_history)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=r,
            max_steps=3,
        )
        provider._tool_calls = [{
            "id": "call_ev", "type": "function",
            "function": {"name": "query_history", "arguments": '{"m": "hr"}'},
        }]
        log = await agent.handle_event(make_hr_event())

        # evidence has top-level keys
        assert "event" in log.evidence
        assert "sensing_summary" in log.evidence
        assert "tool_results" in log.evidence
        assert log.evidence["event"]["event_type"] == "hr_abnormal"
        assert log.evidence["sensing_summary"]["heart_rate"] == 120.0

    @pytest.mark.asyncio
    async def test_evidence_tool_results_not_empty(self):
        """When agent calls tools, evidence.tool_results captures them"""
        decision_json = '{"level": "L1", "explanation": "checking tool capture"}'
        provider = MockProvider(response=decision_json)

        r = ToolRegistry()
        @tool(name="ping", description="ping", parameters={"x": {"type": "string"}})
        def ping(x: str) -> str:
            return f"pong:{x}"

        r.register(ping)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=r,
            max_steps=2,
        )
        provider._tool_calls = [{
            "id": "call_p", "type": "function",
            "function": {"name": "ping", "arguments": '{"x": "test"}'},
        }]
        log = await agent.handle_event(make_hr_event())

        assert len(log.evidence["tool_results"]) > 0
        assert log.evidence["tool_results"][0]["tool"] == "ping"
        assert "result" in log.evidence["tool_results"][0]


class TestTriageParser:
    """Phase 8: brace-counting parser for nested TriageDecision JSON"""

    def test_parse_nested_decision(self):
        """Brace-counting parser handles nested JSON objects"""
        agent = DiagnosisAgent(
            resident_id="resident",
            llm_provider=None,
            tool_registry=ToolRegistry(),
            max_steps=2,
        )
        content = (
            '{"level": "L2", "label": "resident_alert", '
            '"event_interpretation": "HR 15bpm above baseline", '
            '"evidence_used": ["query_history", "read_sensing_state"], '
            '"uncertainty": {"sensing_quality": "reliable", '
            '"missing_evidence": [], "needs_recheck": false}, '
            '"action": {"channel": "screen", "recheck_after_sec": 600}, '
            '"safety_boundary": "care_support_only"}'
        )
        d = agent._try_parse_decision(content)
        assert d is not None
        assert d["level"] == "L2"
        assert d["label"] == "resident_alert"
        assert d["uncertainty"]["sensing_quality"] == "reliable"
        assert d["action"]["channel"] == "screen"

    def test_parse_partial_fills_defaults(self):
        """Missing optional fields get default values"""
        # Only level provided — should still parse with defaults
        content = '{"level": "L1", "event_interpretation": "mild deviation"}'
        agent = DiagnosisAgent("r", None, ToolRegistry(), 2)
        d = agent._try_parse_decision(content)
        assert d is not None
        assert d["level"] == "L1"
        # Optional fields get defaults
        assert d["safety_boundary"] == "care_support_only"
        assert d["uncertainty"]["needs_recheck"] is True

    def test_parse_rejects_invalid_level(self):
        """Invalid level returns None"""
        content = '{"level": "L5", "explanation": "test"}'
        agent = DiagnosisAgent("r", None, ToolRegistry(), 2)
        d = agent._try_parse_decision(content)
        assert d is None

    def test_parse_no_json_returns_none(self):
        """No JSON at all returns None"""
        agent = DiagnosisAgent("r", None, ToolRegistry(), 2)
        d = agent._try_parse_decision("I'm thinking...")
        assert d is None

    def test_parse_malformed_json_returns_none(self):
        """Malformed JSON returns None"""
        agent = DiagnosisAgent("r", None, ToolRegistry(), 2)
        d = agent._try_parse_decision('{"level": "L1", broken')
        assert d is None

    def test_parse_flat_decision_still_works(self):
        """Backward compat: flat JSON still parses"""
        content = '{"level": "L0", "explanation": "normal"}'
        agent = DiagnosisAgent("r", None, ToolRegistry(), 2)
        d = agent._try_parse_decision(content)
        assert d is not None
        assert d["level"] == "L0"


# ---------------------------------------------------------------------------
# DiagnosisAgent context upgrade tests (Phase F)
# ---------------------------------------------------------------------------

class TestDiagnosisAgentContext:
    """Tests for duration_sec and fusion context in DiagnosisAgent."""

    def test_duration_in_context(self):
        from agent_layer.diagnosis_agent import DiagnosisAgent
        from agent_layer.state_objects import HealthEvent, StateObject
        from agent_layer.tools import ToolRegistry
        from datetime import datetime

        agent = DiagnosisAgent("r1", None, ToolRegistry())
        state = StateObject("ctx1", datetime.now(), heart_rate=85.0)
        event = HealthEvent("e1", "hr_abnormal", datetime.now(), state, "HR high",
                          rule_markers={"duration_sec": 300, "hr_z_score": 3.5})

        # Override llm_provider to avoid None crash
        class MockLLM:
            async def chat(self, msgs): return type('obj', (object,), {'content': '{"level":"L0"}', 'tool_calls': None})
            async def chat_with_tools(self, msgs, schema): return type('obj', (object,), {'content': '{"level":"L0"}', 'tool_calls': None})
        agent.llm = MockLLM()

        msgs = agent._build_context(event)
        ctx = msgs[1].content
        import json
        parsed = json.loads(ctx)
        persistence = parsed.get("clinical_summary", {}).get("persistence", {})
        assert persistence.get("duration_sec") == 300, f"Missing persistence.duration_sec in context:\n{ctx}"
        assert persistence.get("is_persistent_5min") is True, f"Missing is_persistent_5min in context:\n{ctx}"

    def test_fusion_context_included(self):
        from agent_layer.diagnosis_agent import DiagnosisAgent
        from agent_layer.state_objects import HealthEvent, StateObject
        from agent_layer.tools import ToolRegistry
        from datetime import datetime

        class MockLLM:
            async def chat(self, msgs): return type('obj', (object,), {'content': '{"level":"L0"}', 'tool_calls': None})
            async def chat_with_tools(self, msgs, schema): return type('obj', (object,), {'content': '{"level":"L0"}', 'tool_calls': None})

        agent = DiagnosisAgent("r1", MockLLM(), ToolRegistry())
        state = StateObject("ctx2", datetime.now(), heart_rate=85.0)
        event = HealthEvent("e2", "modality_conflict", datetime.now(), state, "HR conflict",
                          rule_markers={"hr_modality_delta": 8.5, "hr_dominant": "wifi"})

        msgs = agent._build_context(event)
        ctx = msgs[1].content
        assert "模态差异" in ctx or "hr_modality_delta" in ctx, f"Missing fusion context:\n{ctx}"

    def test_evidence_includes_rule_markers(self):
        from agent_layer.diagnosis_agent import DiagnosisAgent
        from agent_layer.state_objects import HealthEvent, StateObject
        from agent_layer.tools import ToolRegistry
        from datetime import datetime

        agent = DiagnosisAgent("r1", None, ToolRegistry())
        state = StateObject("ctx3", datetime.now(), heart_rate=85.0)
        event = HealthEvent("e3", "hr_abnormal", datetime.now(), state, "HR high",
                          rule_markers={"duration_sec": 300})

        ev = agent._build_evidence(event, [])
        assert "rule_markers" in ev, "Missing rule_markers in evidence"
        assert ev["rule_markers"]["duration_sec"] == 300

    def test_evidence_includes_portable_fields_without_truth(self):
        from agent_layer.diagnosis_agent import DiagnosisAgent
        from agent_layer.state_objects import HealthEvent, StateObject
        from agent_layer.tools import ToolRegistry
        from datetime import datetime

        agent = DiagnosisAgent("r1", None, ToolRegistry())
        state = StateObject(
            "ctx4", datetime.now(), heart_rate=75.0,
            hr_wifi=70.0, hr_mm=80.0, hr_conf=0.9,
            rr_wifi=17.0, rr_mm=19.0, rr_conf=0.7,
            hr_truth=73.0, rr_truth=18.0,
        )
        event = HealthEvent("e4", "modality_conflict", datetime.now(), state, "conflict")

        ev = agent._build_evidence(event, [])
        summary = ev["sensing_summary"]
        assert summary["hr_wifi"] == 70.0
        assert summary["hr_mm"] == 80.0
        assert summary["rr_conf"] == 0.7
        assert "hr_truth" not in summary
        assert "rr_truth" not in summary
