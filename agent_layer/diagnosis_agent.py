"""Diagnosis Agent — Think/Act ReAct 循环"""

import json
import re
import uuid
from datetime import datetime
from typing import Optional

from agent_layer.state_objects import StateObject, HealthEvent, EpisodeLog
from agent_layer.llm_provider import LLMProvider, ChatMessage, ChatResponse
from agent_layer.tools import ToolRegistry
from agent_layer.tiered_action import ActionLevel, resolve_action


SYSTEM_PROMPT = """你是一个家庭健康诊断医生。你的工作流程：

1. 【Think】分析当前异常事件与居民历史基线的差异，判断严重程度
2. 【Act】必要时调用工具查询更多信息
3. 【Decide】输出 JSON 决策

决策级别：
- L0: 正常范围，仅记录
- L1: 轻度偏离，持续观察
- L2: 异常，需要提醒居民
- L3: 持续异常，通知家属
- L4: 紧急情况，立即通知紧急联系人

最终输出必须是 JSON 格式，包含: level, explanation, action_message"""


class DiagnosisAgent:
    """ReAct 循环实现: Think → Act → Decide"""

    def __init__(
        self,
        resident_id: str,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        max_steps: int = 8,
    ):
        self.resident_id = resident_id
        self.llm = llm_provider
        self.tools = tool_registry
        self.max_steps = max_steps

    VALID_LEVELS = {"L0", "L1", "L2", "L3", "L4"}

    async def handle_event(self, event: HealthEvent) -> EpisodeLog:
        """主入口: 处理一个异常事件"""
        start_time = datetime.now()
        messages = self._build_context(event)
        tools_called: list[str] = []

        for step in range(self.max_steps):
            # ── Decide which LLM method to use ──
            has_tools = bool(self.tools.names)
            if has_tools:
                resp = await self.llm.chat_with_tools(messages, self.tools.schema())
            else:
                resp = await self.llm.chat(messages)

            # ── Act phase: process tool calls first ──
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    result = self.tools.execute(name, args)
                    tools_called.append(name)
                    messages.append(ChatMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tc.get("id", ""),
                    ))

            # ── Think phase: try to parse decision from response content ──
            decision = self._try_parse_decision(resp.content)
            if decision:
                action_cfg = resolve_action(decision.get("level", "L0"))
                return EpisodeLog(
                    episode_id=f"ep_{uuid.uuid4().hex[:8]}",
                    resident_id=self.resident_id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    evidence={
                        "sensing_summary": {
                            "heart_rate": event.state.heart_rate,
                            "body_temp": event.state.body_temp,
                            "fall_status": event.state.fall_status,
                        }
                    },
                    decision=decision,
                    action={
                        "channel": action_cfg["channel"],
                        "message": decision.get(
                            "action_message", action_cfg["message"]
                        ),
                        "recheck_after": decision.get(
                            "recheck_after_sec", action_cfg["recheck_after_sec"]
                        ),
                    },
                    audit={
                        "tools_called": tools_called,
                        "step_count": step + 1,
                        "event_id": event.event_id,
                    },
                )

            # ── No tool calls and no decision → give a nudge ──
            if not resp.tool_calls:
                messages.append(ChatMessage(
                    role="user",
                    content="请输出 JSON 格式的决策，包含 level、explanation 和 action_message。",
                ))

        # ── Max steps exceeded → fallback to L0 ──
        action_cfg = resolve_action("L0")
        return EpisodeLog(
            episode_id=f"ep_{uuid.uuid4().hex[:8]}",
            resident_id=self.resident_id,
            start_time=start_time,
            end_time=datetime.now(),
            evidence={},
            decision={
                "level": "L0",
                "explanation": "max steps reached, default to L0",
            },
            action={
                "channel": action_cfg["channel"],
                "message": action_cfg["message"],
                "recheck_after": action_cfg["recheck_after_sec"],
            },
            audit={
                "tools_called": tools_called,
                "step_count": self.max_steps,
                "event_id": event.event_id,
            },
        )

    def _build_context(self, event: HealthEvent) -> list[ChatMessage]:
        state = event.state
        context = (
            f"异常事件: {event.event_type}\n"
            f"触发原因: {event.trigger_reason}\n"
            f"居民: {self.resident_id}\n\n"
            f"当前体征:\n"
            f"- 心率: {state.heart_rate}\n"
            f"- 呼吸率: {state.respiration_rate}\n"
            f"- 体温: {state.body_temp}\n"
            f"- 跌倒状态: {state.fall_status}\n"
            f"- 活动状态: {state.activity_state}\n"
            f"- 置信度(WiFi/mmWave/红外): "
            f"{state.wifi_confidence}/{state.mmwave_confidence}/"
            f"{state.thermal_confidence}\n"
            f"- NLOS: {state.nlos_flag}"
        )
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=context),
        ]

    def _try_parse_decision(self, content: str) -> Optional[dict]:
        m = re.search(
            r'\{[^{}]*"level"[^{}]*\}', content, re.DOTALL
        )
        if not m:
            return None
        try:
            d = json.loads(m.group())
            if d.get("level") in self.VALID_LEVELS:
                return d
        except json.JSONDecodeError:
            return None
        return None
