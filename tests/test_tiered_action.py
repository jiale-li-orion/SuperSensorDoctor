import pytest
from agent_layer.tiered_action import ActionLevel, resolve_action


def test_action_levels():
    assert ActionLevel.L0.value == "L0"
    assert ActionLevel.L4.value == "L4"


def test_resolve_l0():
    a = resolve_action("L0")
    assert a["level"] == "L0"
    assert a["channel"] == "none"


def test_resolve_l4():
    a = resolve_action("L4")
    assert a["channel"] == "emergency"
    assert "立即" in a["message"]
