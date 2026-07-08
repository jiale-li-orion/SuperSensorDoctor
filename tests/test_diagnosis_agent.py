"""Tests for DiagnosisAgent Think/Act ReAct loop"""

import pytest
from datetime import datetime
from agent_layer.state_objects import StateObject, HealthEvent
from agent_layer.llm_provider import MockProvider, ChatMessage
from agent_layer.tools import ToolRegistry, tool
from agent_layer.diagnosis_agent import DiagnosisAgent


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
