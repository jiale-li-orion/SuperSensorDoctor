"""Tests for TriageDecision roundtrip from DiagnosisAgent through write_episode_tool to DB

Covers:
1. Normal roundtrip — Full 7-key TriageDecision schema persisted and readable
2. Audit ordering — issue_action and write_episode are appended last in tools_called
3. step_count — Correctly persisted in DB audit
4. Backward compat — Old-style scalar params (decision_level, decision_explanation) still work
5. Full evidence chain — FusionResult estimates/checks/verdict stored in DB evidence
"""

import os
import json
import pytest
from datetime import datetime
from agent_layer.state_objects import StateObject, HealthEvent
from agent_layer.llm_provider import MockProvider
from agent_layer.tools import (
    ToolRegistry, tool, write_episode_tool,
    consult_fusion_tool, issue_action_tool,
)
from agent_layer.diagnosis_agent import DiagnosisAgent
from storage.db import init_db, DB_PATH
from storage.models import insert_sensing_window, insert_health_event, query_episodes_by_resident


# ── Fixtures ──

@pytest.fixture(autouse=True)
def clean_db():
    """Initialize DB and seed FK parent rows for each test."""
    init_db()
    base = datetime(2025, 6, 1, 12, 0, 0)
    insert_sensing_window(
        window_id="w1", timestamp=base,
        heart_rate=72.0, respiration_rate=16.0, body_temp=36.5,
        wifi_confidence=0.9, mmwave_confidence=0.85,
        nlos_flag=False, activity_state="sitting", fall_status="none",
    )
    insert_health_event(
        event_id="evt_001", window_id="w1",
        event_type="hr_abnormal", timestamp=base,
        trigger_reason="HR deviation 15 bpm",
    )
    yield
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


# ── Helpers ──

def make_hr_event(hr: float = 120.0) -> HealthEvent:
    s = StateObject(
        window_id="w1", timestamp=datetime.now(),
        heart_rate=hr, body_temp=36.5,
        wifi_confidence=0.9, mmwave_confidence=0.85,
        activity_state="sitting",
    )
    return HealthEvent(
        event_id="evt_001", event_type="hr_abnormal",
        timestamp=datetime.now(), state=s,
        trigger_reason="HR deviation 15 bpm",
    )


FULL_TRIAGE = json.dumps({
    "level": "L2",
    "label": "resident_alert",
    "event_interpretation": "HR 15bpm above baseline with degraded mmWave confidence",
    "evidence_used": ["query_history", "consult_fusion"],
    "uncertainty": {
        "sensing_quality": "degraded",
        "missing_evidence": ["trend_analysis"],
        "needs_recheck": True,
    },
    "action": {"channel": "screen", "recheck_after_sec": 600},
    "safety_boundary": "care_support_only",
})


# ── 1. Normal roundtrip ──

class TestNormalRoundtrip:
    """The full 7-key TriageDecision flows Agent → DB → readback"""

    @pytest.mark.asyncio
    async def test_all_seven_keys_persisted(self):
        """Every TriageDecision field survives the write → read cycle"""
        provider = MockProvider(response=FULL_TRIAGE)
        r = ToolRegistry()

        @tool(
            name="query_history", description="hist",
            parameters={"m": {"type": "string"}},
        )
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
            "id": "call_qh", "type": "function",
            "function": {"name": "query_history", "arguments": '{"m": "hr"}'},
        }]

        log = await agent.handle_event(make_hr_event())

        # ── Read from DB ──
        logs = query_episodes_by_resident("resident_01")
        assert len(logs) == 1
        decision = json.loads(logs[0]["decision"])

        # 7-key schema verification
        assert decision["level"] == "L2"
        assert decision["label"] == "resident_alert"
        assert "HR 15bpm" in decision["event_interpretation"]
        assert set(decision["evidence_used"]) == {"query_history", "consult_fusion"}
        # uncertainty (3 sub-keys)
        assert decision["uncertainty"]["sensing_quality"] == "degraded"
        assert decision["uncertainty"]["missing_evidence"] == ["trend_analysis"]
        assert decision["uncertainty"]["needs_recheck"] is True
        # action (2 sub-keys)
        assert decision["action"]["channel"] == "screen"
        assert decision["action"]["recheck_after_sec"] == 600
        # safety_boundary
        assert decision["safety_boundary"] == "care_support_only"

        # EpisodeLog also carries the decision
        assert log.decision["level"] == "L2"
        assert log.decision["label"] == "resident_alert"


# ── 2. Audit ordering ──

class TestAuditOrdering:
    """issue_action and write_episode appear last in tools_called"""

    @pytest.mark.asyncio
    async def test_tools_called_ends_with_orchestrator_tools(self):
        """tools_called ends with ['issue_action', 'write_episode']"""
        provider = MockProvider(response=FULL_TRIAGE)
        r = ToolRegistry()

        @tool(
            name="read_sensing_state", description="sensing",
            parameters={"r": {"type": "string"}},
        )
        def read_state(r: str) -> str:
            return {"status": "ok", "hr": 120.0}

        r.register(read_state)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=r,
            max_steps=3,
        )
        provider._tool_calls = [{
            "id": "call_rss", "type": "function",
            "function": {
                "name": "read_sensing_state",
                "arguments": '{"r": "resident_01"}',
            },
        }]

        await agent.handle_event(make_hr_event())
        logs = query_episodes_by_resident("resident_01")
        audit = json.loads(logs[0]["audit"])
        tools = audit["tools_called"]

        assert "read_sensing_state" in tools
        assert tools[-2] == "issue_action"
        assert tools[-1] == "write_episode"

    @pytest.mark.asyncio
    async def test_issue_action_before_write_episode(self):
        """issue_action is called BEFORE write_episode in agent flow"""
        provider = MockProvider(response=FULL_TRIAGE)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            max_steps=3,
        )
        await agent.handle_event(make_hr_event())
        logs = query_episodes_by_resident("resident_01")
        audit = json.loads(logs[0]["audit"])
        tools = audit["tools_called"]

        ia_idx = tools.index("issue_action")
        we_idx = tools.index("write_episode")
        assert ia_idx < we_idx, "issue_action must precede write_episode"


# ── 3. step_count ──

class TestStepCount:
    """step_count correctly persisted in DB audit"""

    @pytest.mark.asyncio
    async def test_step_count_from_first_decision(self):
        """Decision on first LLM response → step_count == 1"""
        provider = MockProvider(response=FULL_TRIAGE)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            max_steps=5,
        )
        await agent.handle_event(make_hr_event())

        logs = query_episodes_by_resident("resident_01")
        audit = json.loads(logs[0]["audit"])
        assert audit["step_count"] == 1

    @pytest.mark.asyncio
    async def test_step_count_matches_audit_field(self):
        """EpisodeLog.audit['step_count'] matches DB audit['step_count']"""
        provider = MockProvider(response=FULL_TRIAGE)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            max_steps=5,
        )
        log = await agent.handle_event(make_hr_event())

        logs = query_episodes_by_resident("resident_01")
        db_audit = json.loads(logs[0]["audit"])
        assert db_audit["step_count"] == log.audit["step_count"]

    @pytest.mark.asyncio
    async def test_step_count_with_tool_calls(self):
        """Agent using tools still records correct step_count"""
        provider = MockProvider(response=FULL_TRIAGE)
        r = ToolRegistry()

        @tool(
            name="query_history", description="hist",
            parameters={"m": {"type": "string"}},
        )
        def query_history(m: str) -> str:
            return {"status": "ok", "mean": 72.0}

        r.register(query_history)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=r,
            max_steps=5,
        )
        provider._tool_calls = [{
            "id": "call_qh", "type": "function",
            "function": {"name": "query_history", "arguments": '{"m": "hr"}'},
        }]
        log = await agent.handle_event(make_hr_event())

        logs = query_episodes_by_resident("resident_01")
        db_audit = json.loads(logs[0]["audit"])
        # Decision + tool call both happen in step 0 → step_count = 1
        assert db_audit["step_count"] == 1
        assert db_audit["step_count"] == log.audit["step_count"]


# ── 4. Backward compat ──

class TestBackwardCompat:
    """write_episode_tool still accepts old-style scalar params"""

    def test_scalar_params_create_db_record(self):
        """decision_level/decision_explanation produce a valid DB episode"""
        result = write_episode_tool(
            resident_id="resident_01",
            event_id="evt_001",
            decision_level="L2",
            decision_explanation="HR elevated beyond threshold",
            action_message="提醒休息",
        )
        assert result["status"] == "ok"
        assert result["level"] == "L2"

        logs = query_episodes_by_resident("resident_01")
        assert len(logs) == 1
        decision = json.loads(logs[0]["decision"])
        assert decision["level"] == "L2"
        assert "HR elevated" in decision["event_interpretation"]

    def test_scalar_params_defaults(self):
        """Old-style fills defaults for missing TriageDecision keys"""
        write_episode_tool(
            resident_id="resident_01",
            event_id="evt_001",
            decision_level="L1",
            decision_explanation="mild deviation",
        )
        logs = query_episodes_by_resident("resident_01")
        decision = json.loads(logs[0]["decision"])

        assert decision["level"] == "L1"
        assert decision["label"] == ""               # no label from old path
        assert decision["evidence_used"] == []        # no evidence from old path
        assert decision["uncertainty"]["sensing_quality"] == "unknown"
        assert decision["uncertainty"]["needs_recheck"] is True
        assert decision["safety_boundary"] == "care_support_only"
        assert decision["action"] == {}               # empty dict from old path

    def test_scalar_vs_dict_roundtrip_consistency(self):
        """Both calling conventions produce queryable episode_logs rows"""
        # Dict-style
        r1 = write_episode_tool(
            resident_id="resident_01", event_id="evt_001",
            decision={"level": "L2", "label": "alert",
                      "event_interpretation": "dict path",
                      "evidence_used": ["fusion"], "uncertainty": {},
                      "action": {}, "safety_boundary": "care_support_only"},
        )
        # Scalar-style with different event
        insert_health_event(
            event_id="evt_002", window_id="w1",
            event_type="hr_abnormal", timestamp=datetime.now(),
            trigger_reason="scalar test",
        )
        r2 = write_episode_tool(
            resident_id="resident_01", event_id="evt_002",
            decision_level="L1",
            decision_explanation="scalar path",
        )

        logs = query_episodes_by_resident("resident_01", limit=10)
        ids = [l["episode_id"] for l in logs]
        assert r1["episode_id"] in ids
        assert r2["episode_id"] in ids


# ── 5. Full evidence chain ──

class TestEvidenceChain:
    """FusionResult and tool evidence stored in DB"""

    @pytest.mark.asyncio
    async def test_fusion_result_estimates_checks_verdict(self):
        """consult_fusion result (estimates/checks/verdict) in DB evidence"""
        provider = MockProvider(response=FULL_TRIAGE)
        r = ToolRegistry()
        r.register(consult_fusion_tool)
        agent = DiagnosisAgent(
            resident_id="resident_01",
            llm_provider=provider,
            tool_registry=r,
            max_steps=3,
        )
        provider._tool_calls = [{
            "id": "call_cf", "type": "function",
            "function": {
                "name": "consult_fusion",
                "arguments": '{"resident_id": "resident_01", "metric": "hr"}',
            },
        }]

        await agent.handle_event(make_hr_event())
        logs = query_episodes_by_resident("resident_01")
        evidence = json.loads(logs[0]["evidence"])

        # Event metadata preserved
        assert evidence["event"]["event_id"] == "evt_001"
        assert evidence["event"]["event_type"] == "hr_abnormal"

        # Sensing summary preserved
        assert evidence["sensing_summary"]["heart_rate"] == 120.0
        assert evidence["sensing_summary"]["wifi_confidence"] == 0.9

        # Tool results contain fusion output
        assert len(evidence["tool_results"]) == 1
        tr = evidence["tool_results"][0]
        assert tr["tool"] == "consult_fusion"
        result = tr["result"]

        # Three FusionResult sections
        assert "estimates" in result
        assert "checks" in result
        assert "verdict" in result

        # estimates has per-modality keys
        assert "wifi" in result["estimates"]
        assert "mmwave" in result["estimates"]
        assert result["estimates"]["wifi"]["confidence"] == 0.9
        assert result["estimates"]["mmwave"]["confidence"] == 0.85

        # checks
        assert result["checks"]["consistent"] is True

        # verdict
        assert result["verdict"]["dominant_modality"] == "fusion"
        assert "fused_value" in result["verdict"]

    def test_direct_evidence_storage(self):
        """Evidence dict with arbitrary keys round-trips through write_episode_tool"""
        fusion_evidence = {
            "fusion": {
                "estimates": {
                    "wifi": {"value": 72.0, "confidence": 0.9},
                    "mmwave": {"value": 72.0, "confidence": 0.85},
                },
                "checks": {"consistent": True, "confidence_gap": 0.05},
                "verdict": {"fused_value": 72.0, "dominant_modality": "fusion"},
            },
        }
        result = write_episode_tool(
            resident_id="resident_01",
            event_id="evt_001",
            decision={"level": "L2", "label": "evidence_test",
                      "event_interpretation": "test", "evidence_used": [],
                      "uncertainty": {}, "action": {},
                      "safety_boundary": "care_support_only"},
            evidence=fusion_evidence,
        )
        logs = query_episodes_by_resident("resident_01")
        db_evidence = json.loads(logs[0]["evidence"])
        assert "fusion" in db_evidence
        assert db_evidence["fusion"]["verdict"]["dominant_modality"] == "fusion"
        assert db_evidence["fusion"]["checks"]["consistent"] is True
