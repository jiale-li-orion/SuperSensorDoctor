import json
from unittest.mock import patch

import httpx
import pytest
from agent_layer.llm_provider import DeepSeekProvider, ChatMessage


@pytest.mark.asyncio
async def test_chat_api_failure_returns_degraded_response():
    """chat() should return degraded response on API error instead of crashing."""
    provider = DeepSeekProvider(api_key="test", base_url="http://localhost:9999")
    messages = [ChatMessage(role="user", content="hello")]

    async def _raise(*args, **kwargs):
        raise httpx.HTTPStatusError("401 Unauthorized", request=httpx.Request("POST", "http://test"), response=httpx.Response(401))

    with patch.object(httpx.AsyncClient, "post", _raise):
        resp = await provider.chat(messages)

    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    body = json.loads(resp.content)
    assert body["label"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_chat_with_tools_failure_returns_degraded_response():
    """chat_with_tools() should return degraded response on API error instead of crashing."""
    provider = DeepSeekProvider(api_key="test", base_url="http://localhost:9999")
    messages = [ChatMessage(role="user", content="hello")]

    async def _raise(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out")

    with patch.object(httpx.AsyncClient, "post", _raise):
        resp = await provider.chat_with_tools(messages, [])

    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    body = json.loads(resp.content)
    assert body["label"] == "llm_unavailable"
