"""Tests for Agent Tool system and design-spec tools"""

import os
import pytest
from datetime import datetime, timedelta
from agent_layer.tools import (
    tool, ToolRegistry, create_default_tools,
    read_sensing_state_tool, consult_fusion_tool,
    write_episode_tool, issue_action_tool,
)
from storage.db import init_db, DB_PATH
from storage.models import insert_sensing_window, insert_health_event


# ── Fixtures ──

@pytest.fixture(autouse=True)
def clean_db():
    """Initialize DB and seed FK parent rows for each test.

    Stagger timestamps so w_test (heart_rate=72) is the latest,
    ensuring deterministic ordering for tests that rely on
    query_latest_sensing_window().
    """
    init_db()
    base = datetime(2025, 6, 1, 12, 0, 0)
    insert_sensing_window(
        window_id="w_low_conf", timestamp=base,
        heart_rate=95.0, respiration_rate=22.0, body_temp=36.7,
        wifi_confidence=0.30, mmwave_confidence=0.25,
        nlos_flag=False, activity_state="running", fall_status="none",
    )
    insert_sensing_window(
        window_id="w_nlos", timestamp=base + timedelta(seconds=1),
        heart_rate=78.0, respiration_rate=18.0, body_temp=36.6,
        wifi_confidence=0.88, mmwave_confidence=0.92,
        nlos_flag=True, activity_state="walking", fall_status="none",
    )
    insert_sensing_window(
        window_id="w_test", timestamp=base + timedelta(seconds=2),
        heart_rate=72.0, respiration_rate=16.0, body_temp=36.5,
        wifi_confidence=0.9, mmwave_confidence=0.85,
        nlos_flag=False, activity_state="sitting", fall_status="none",
    )
    insert_health_event(
        event_id="he_test", window_id="w_test",
        event_type="hr_abnormal", timestamp=base + timedelta(seconds=2),
        trigger_reason="test seed",
    )
    yield
    try:
        os.remove(DB_PATH)
    except FileNotFoundError:
        pass


# ── Existing tests ──

def test_tool_decorator():
    registry = ToolRegistry()

    @tool(
        name="ping", description="Ping test",
        parameters={"input": {"type": "string"}},
    )
    def ping(input: str) -> str:
        return f"pong: {input}"

    registry.register(ping)
    schema = registry.schema()
    assert len(schema) == 1
    assert schema[0]["function"]["name"] == "ping"

    result = registry.execute("ping", {"input": "hello"})
    assert result == "pong: hello"


def test_tool_schema_format():
    registry = ToolRegistry()

    @tool(
        name="query_history",
        description="Query historical baseline",
        parameters={
            "resident_id": {"type": "string"},
            "metric": {"type": "string", "enum": ["hr", "rr", "temp"]},
        },
    )
    def query_history(resident_id: str, metric: str) -> dict:
        return {"baseline": "ok"}

    registry.register(query_history)
    schema = registry.schema()
    assert schema[0]["function"]["name"] == "query_history"
    assert "parameters" in schema[0]["function"]


def test_tool_call_unknown():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute("nonexistent", {})


@pytest.mark.asyncio
async def test_tool_parallel_execution():
    registry = ToolRegistry()

    @tool(name="a", description="tool a", parameters={})
    def tool_a() -> str:
        return "a"

    @tool(name="b", description="tool b", parameters={})
    def tool_b() -> str:
        return "b"

    registry.register(tool_a)
    registry.register(tool_b)

    assert registry.execute("a", {}) == "a"
    assert registry.execute("b", {}) == "b"


# ── New Tool Tests: read_sensing_state ──

def test_read_sensing_state_returns_full_snapshot():
    """read_sensing_state returns all fields from latest window"""
    result = read_sensing_state_tool("resident_01")
    assert result["status"] == "ok"
    assert result["resident_id"] == "resident_01"
    assert result["heart_rate"] == 72.0
    assert result["respiration_rate"] == 16.0
    assert result["body_temp"] == 36.5
    assert result["wifi_confidence"] == 0.9
    assert result["mmwave_confidence"] == 0.85
    assert result["nlos_flag"] is False
    assert result["fall_status"] == "none"
    assert result["activity_state"] == "sitting"
    assert "window_id" in result
    assert "timestamp" in result


def test_read_sensing_state_no_data():
    """read_sensing_state returns no_data for unknown resident"""
    result = read_sensing_state_tool("nonexistent")
    assert result["status"] == "no_data"


# ── New Tool Tests: consult_fusion ──

def test_consult_fusion_both_reliable_consistent():
    """Both WiFi and mmWave reliable → fusion dominant"""
    result = consult_fusion_tool("resident_01", "hr")
    assert result["status"] == "ok"
    assert result["verdict"]["dominant_modality"] == "fusion"
    assert result["verdict"]["fused_value"] == 72.0
    assert result["estimates"]["wifi"]["confidence"] == 0.9
    assert result["estimates"]["mmwave"]["confidence"] == 0.85
    assert result["estimates"]["mmwave"]["nlos_affected"] is False
    assert result["checks"]["consistent"] is True


def test_consult_fusion_nlos_occlusion():
    """NLOS occlusion → WiFi dominant"""
    insert_sensing_window(
        window_id="w_nlos_test", timestamp=datetime.now(),
        heart_rate=78.0, respiration_rate=18.0,
        wifi_confidence=0.88, mmwave_confidence=0.92,
        nlos_flag=True, activity_state="walking",
    )
    result = consult_fusion_tool("resident_01", "rr")
    assert result["status"] == "ok"
    assert result["verdict"]["dominant_modality"] == "wifi"
    assert "NLOS" in result["verdict"]["rationale"]


def test_consult_fusion_wifi_reliable_only():
    """Only WiFi reliable → wifi dominant"""
    insert_sensing_window(
        window_id="w_wifi_only", timestamp=datetime.now(),
        heart_rate=80.0,
        wifi_confidence=0.85, mmwave_confidence=0.30,
        nlos_flag=False,
    )
    result = consult_fusion_tool("resident_01", "hr")
    assert result["verdict"]["dominant_modality"] == "wifi"
    assert "WiFi reliable" in result["verdict"]["rationale"]


def test_consult_fusion_mmwave_reliable_only():
    """Only mmWave reliable → mmwave dominant"""
    insert_sensing_window(
        window_id="w_mmwave_only", timestamp=datetime.now(),
        heart_rate=80.0,
        wifi_confidence=0.20, mmwave_confidence=0.90,
        nlos_flag=False,
    )
    result = consult_fusion_tool("resident_01", "hr")
    assert result["verdict"]["dominant_modality"] == "mmwave"
    assert "mmWave reliable" in result["verdict"]["rationale"]


def test_consult_fusion_both_low_fallback():
    """Both modalities low confidence → wifi_fallback"""
    insert_sensing_window(
        window_id="w_both_low", timestamp=datetime.now(),
        heart_rate=95.0,
        wifi_confidence=0.30, mmwave_confidence=0.25,
        nlos_flag=False,
    )
    result = consult_fusion_tool("resident_01", "hr")
    assert result["verdict"]["dominant_modality"] == "best_effort"


def test_consult_fusion_no_data():
    """No data for unknown resident"""
    result = consult_fusion_tool("nonexistent", "hr")
    assert result["status"] == "no_data"


def test_fusion_result_has_three_sections():
    """FusionResult contains estimates, checks, verdict sections"""
    result = consult_fusion_tool("resident_01", "hr")
    assert "estimates" in result
    assert "checks" in result
    assert "verdict" in result

    # estimates contains per-modality keys
    assert "wifi" in result["estimates"]
    assert "mmwave" in result["estimates"]

    # checks contains consistency info
    assert "consistent" in result["checks"]
    assert "confidence_gap" in result["checks"]

    # verdict contains fused output
    assert "fused_value" in result["verdict"]
    assert "dominant_modality" in result["verdict"]
    assert "rationale" in result["verdict"]


# ── New Tool Tests: write_episode ──

def test_write_episode_creates_db_record():
    """write_episode persists an episode_log and returns episode_id"""
    result = write_episode_tool(
        resident_id="resident_01",
        event_id="he_test",
        decision_level="L2",
        decision_explanation="HR elevated beyond threshold",
        action_message="提醒休息",
    )
    assert result["status"] == "ok"
    assert "episode_id" in result
    assert result["channel"] == "screen"
    assert "提醒" in result["message"]


def test_write_episode_round_trip():
    """Written episode is queryable via list_recent_events"""
    result = write_episode_tool(
        resident_id="resident_01",
        event_id="he_test",
        decision_level="L3",
        decision_explanation="持续异常, 通知家属",
        action_message="持续异常, 通知家属",
    )
    ep_id = result["episode_id"]

    from storage.models import query_episodes_by_resident
    logs = query_episodes_by_resident("resident_01")
    ids = [l["episode_id"] for l in logs]
    assert ep_id in ids


def test_write_episode_all_levels():
    """All L0-L4 levels produce correct channels from resolve_action"""
    expected_channels = {
        "L0": "none",
        "L1": "none",
        "L2": "screen",
        "L3": "family_push",
        "L4": "emergency",
    }
    for level, channel in expected_channels.items():
        result = write_episode_tool(
            resident_id="resident_01",
            event_id="he_test",
            decision_level=level,
            decision_explanation=f"Test {level}",
        )
        assert result["status"] == "ok"
        assert result["channel"] == channel, f"Expected {channel} for {level}"


# ── New Tool Tests: issue_action ──

def test_issue_action_each_level():
    """issue_action returns correct channel/recheck for each level (from resolve_action)"""
    cases = {
        "L0": {"channel": "none", "recheck_after_sec": None},
        "L1": {"channel": "none", "recheck_after_sec": 300},
        "L2": {"channel": "screen", "recheck_after_sec": 600},
        "L3": {"channel": "family_push", "recheck_after_sec": 1800},
        "L4": {"channel": "emergency", "recheck_after_sec": 60},
    }
    for level, expected in cases.items():
        result = issue_action_tool(level)
        assert result["status"] == "ok"
        assert result["level"] == level
        assert result["channel"] == expected["channel"]
        assert result["recheck_after_sec"] == expected["recheck_after_sec"]


def test_issue_action_custom_message():
    """Custom message overrides default"""
    result = issue_action_tool("L2", "自定义消息")
    assert result["message"] == "自定义消息"


def test_issue_action_empty_message_returns_default():
    """Empty message keeps the default action message"""
    result = issue_action_tool("L0")
    assert len(result["message"]) > 0  # not empty; uses resolve_action default


# ── Registry Integration ──

def test_create_default_tools_contains_all():
    """create_default_tools registers all 7 tools (write_episode/issue_action are orchestrator-only)"""
    registry = create_default_tools()
    expected = {
        "query_history", "read_sensing_state", "consult_fusion",
        "get_latest_vitals", "list_recent_events",
        "check_resident_context", "trend_analysis",
    }
    assert set(registry.names) == expected


def test_new_tools_executable_directly():
    """write_episode and issue_action work when called directly (orchestrator-only, not in registry)"""
    from agent_layer.tools import write_episode_tool, issue_action_tool

    # issue_action direct call
    r4 = issue_action_tool("L3", "test")
    assert r4["status"] == "ok"
    assert r4["level"] == "L3"

    # write_episode_tool direct call (uses he_test seeded by clean_db fixture)
    r3 = write_episode_tool(
        resident_id="resident_01", event_id="he_test",
        decision={"level": "L1", "label": "direct_test", "event_interpretation": "direct",
                  "evidence_used": [], "uncertainty": {}, "action": {},
                  "safety_boundary": "care_support_only"},
    )
    assert r3["status"] == "ok"
    assert r3["level"] == "L1"
