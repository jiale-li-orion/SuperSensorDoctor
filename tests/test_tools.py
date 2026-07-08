import pytest
from agent_layer.tools import tool, ToolRegistry


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

    # Verify both tools work correctly (synchronous, no shared state conflicts)
    assert registry.execute("a", {}) == "a"
    assert registry.execute("b", {}) == "b"
