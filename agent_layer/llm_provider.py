"""LLM Provider 抽象层 — OpenAI 兼容实现"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class ChatMessage:
    role: str       # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatResponse:
    content: str
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """Think 阶段: 无 tools, 纯文本推理"""
        ...

    @abstractmethod
    async def chat_with_tools(
        self, messages: list[ChatMessage], tools: list[dict]
    ) -> ChatResponse:
        """Act 阶段: 带 tools 的推理"""
        ...


class DeepSeekProvider(LLMProvider):
    """DeepSeek API (OpenAI 兼容) 实现"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _to_api_messages(self, messages: list[ChatMessage]) -> list[dict]:
        result = []
        for m in messages:
            d = {"role": m.role, "content": m.content}
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            result.append(d)
        return result

    def _parse_response(self, raw: dict) -> ChatResponse:
        choice = raw["choices"][0]
        msg = choice["message"]
        return ChatResponse(
            content=msg.get("content", "") or "",
            tool_calls=msg.get("tool_calls", []),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self._to_api_messages(messages),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def chat_with_tools(
        self, messages: list[ChatMessage], tools: list[dict]
    ) -> ChatResponse:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": self._to_api_messages(messages),
                    "tools": tools,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())


class MockProvider(LLMProvider):
    """测试用 Mock LLM Provider"""

    def __init__(self, response: str = "mock response", tool_calls: list = None):
        self._response = response
        self._tool_calls = tool_calls or []

    async def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        return ChatResponse(content=self._response)

    async def chat_with_tools(
        self, messages: list[ChatMessage], tools: list[dict]
    ) -> ChatResponse:
        return ChatResponse(
            content=self._response,
            tool_calls=self._tool_calls,
            finish_reason="tool_calls" if self._tool_calls else "stop",
        )
