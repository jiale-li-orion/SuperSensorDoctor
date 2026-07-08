import pytest
from agent_layer.llm_provider import DeepSeekProvider, ChatMessage


@pytest.mark.asyncio
async def test_chat_no_tools():
    provider = DeepSeekProvider(api_key="test", base_url="http://localhost:9999")
    messages = [ChatMessage(role="user", content="hello")]
    with pytest.raises(Exception):  # connection refused is expected
        await provider.chat(messages)
