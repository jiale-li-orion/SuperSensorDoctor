"""Diagnosis Agent — Think/Act ReAct 循环"""

import json

import uuid
from datetime import datetime
from typing import Optional

from agent_layer.state_objects import StateObject, HealthEvent, EpisodeLog
from agent_layer.llm_provider import LLMProvider, ChatMessage, ChatResponse
from agent_layer.tools import ToolRegistry
from agent_layer.tiered_action import ActionLevel, resolve_action
from agent_layer.tools import issue_action_tool, write_episode_tool


SYSTEM_PROMPT = """你是一个家庭看护分级助手。你只基于结构化传感证据、历史基线和工具结果进行看护分级。你的输出用于 care support，不构成医学诊断、治疗建议或用药建议。

1. 【Think】分析当前异常事件与居民历史基线的差异，判断严重程度
2. 【Act】必要时调用工具查询更多信息：read_sensing_state(当前快照), query_history(历史基线), consult_fusion(跨模态仲裁)
3. 【Decide】输出 JSON 格式的 TriageDecision

TriageDecision 字段说明：
{
  "level": "L0-L4 之一 (必填)",
  "label": "决策标签 (continuous_observation / resident_alert / family_notification / emergency)",
  "event_interpretation": "当前事件的看护风险解释 (必填)",
  "evidence_used": ["引用的工具名称列表"],
  "uncertainty": {
    "sensing_quality": "reliable / degraded / unreliable",
    "missing_evidence": ["缺失的证据/工具"],
    "needs_recheck": true/false
  },
  "action": {
    "channel": "none / screen / family_push / emergency",
    "recheck_after_sec": 300
  },
  "safety_boundary": "care_support_only"
}

决策级别：
- L0: 正常范围，仅记录
- L1: 轻度偏离，持续观察
- L2: 异常，需要提醒居民
- L3: 持续异常，通知家属
- L4: 紧急情况，立即通知紧急联系人 (如持续跌倒/生命体征严重异常)"""


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

        # ── Reflex arc: fast-path bypass LLM for known patterns ──
        reflex_decision = self._check_reflex(event)
        if reflex_decision:
            evidence = self._build_evidence(event, [])
            tools_called: list[str] = [f"nurse_rule:{event.event_type}", "issue_action", "write_episode"]
            level = reflex_decision["level"]
            action_result = issue_action_tool(level, reflex_decision.get("action", {}).get("message", ""))
            episode_result = write_episode_tool(
                resident_id=self.resident_id,
                event_id=event.event_id,
                decision=reflex_decision,
                evidence=evidence,
                tools_called=tools_called,
                step_count=0,
            )
            return EpisodeLog(
                episode_id=episode_result.get("episode_id", f"ep_{uuid.uuid4().hex[:8]}"),
                resident_id=self.resident_id,
                start_time=start_time,
                end_time=datetime.now(),
                evidence=evidence,
                decision=reflex_decision,
                action={
                    "channel": action_result["channel"],
                    "message": action_result["message"],
                    "recheck_after": action_result.get(
                        "recheck_after_sec",
                        resolve_action(level)["recheck_after_sec"],
                    ),
                },
                audit={
                    "tools_called": tools_called,
                    "step_count": 0,
                    "event_id": event.event_id,
                    "reflex": True,
                },
            )

        messages = self._build_context(event)
        tools_called: list[str] = []
        tool_results: list[dict] = []

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
                    tool_results.append({
                        "tool": name,
                        "arguments": args,
                        "result": result,
                    })
                    messages.append(ChatMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tc.get("id", ""),
                    ))

            # ── Think phase: try to parse decision from response content ──
            decision = self._try_parse_decision(resp.content)
            if decision:
                level = decision.get("level", "L0")
                evidence = self._build_evidence(event, tool_results)

                tools_called.append("issue_action")
                tools_called.append("write_episode")
                # Use new tools for action resolution + persistence
                action_result = issue_action_tool(level, decision.get("action", {}).get("message", ""))
                episode_result = write_episode_tool(
                    resident_id=self.resident_id,
                    event_id=event.event_id,
                    decision=decision,
                    evidence=evidence,
                    tools_called=tools_called,
                    step_count=step + 1,
                )

                return EpisodeLog(
                    episode_id=episode_result.get(
                        "episode_id", f"ep_{uuid.uuid4().hex[:8]}"
                    ),
                    resident_id=self.resident_id,
                    start_time=start_time,
                    end_time=datetime.now(),
                    evidence=evidence,
                    decision=decision,
                    action={
                        "channel": action_result["channel"],
                        "message": action_result["message"],
                        "recheck_after": action_result.get(
                            "recheck_after_sec",
                            resolve_action(level)["recheck_after_sec"],
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
                    content="请输出 JSON 格式的 TriageDecision，包含 level、event_interpretation、evidence_used 和 uncertainty。",
                ))

        # ── Max steps exceeded → fallback to L0 ──
        evidence = self._build_evidence(event, tool_results)
        fallback_decision = {
            "level": "L0",
            "label": "fallback_max_steps",
            "event_interpretation": "max steps reached, defaulting to silent record",
            "evidence_used": [t.split("(")[0] for t in tools_called] if tools_called else [],
            "uncertainty": {
                "sensing_quality": "unknown",
                "missing_evidence": ["llm_decision"],
                "needs_recheck": True,
            },
            "action": {"channel": "none", "recheck_after_sec": 300},
            "safety_boundary": "care_support_only",
        }
        tools_called.append("issue_action")
        tools_called.append("write_episode")
        action_result = issue_action_tool("L0", "max steps reached, defaulting")
        episode_result = write_episode_tool(
            resident_id=self.resident_id,
            event_id=event.event_id,
            decision=fallback_decision,
            evidence=evidence,
            tools_called=tools_called,
            step_count=self.max_steps,
        )
        return EpisodeLog(
            episode_id=episode_result.get(
                "episode_id", f"ep_{uuid.uuid4().hex[:8]}"
            ),
            resident_id=self.resident_id,
            start_time=start_time,
            end_time=datetime.now(),
            evidence=evidence,
            decision={
                "level": "L0",
                "explanation": "max steps reached, default to L0",
            },
            action={
                "channel": action_result["channel"],
                "message": action_result["message"],
                "recheck_after": action_result.get(
                    "recheck_after_sec",
                    resolve_action("L0")["recheck_after_sec"],
                ),
            },
            audit={
                "tools_called": tools_called,
                "step_count": self.max_steps,
                "event_id": event.event_id,
            },
        )

    @staticmethod
    def _check_reflex(event: HealthEvent) -> Optional[dict]:
        """Reflex arc: bypass LLM for known event patterns.

        Returns a TriageDecision dict if reflex applies, None otherwise.
        """
        state = event.state

        # Crisis patterns (L4)
        if event.event_type == "fall_detected":
            return {
                "level": "L4",
                "label": "emergency",
                "event_interpretation": "跌倒伴随生理指标异常，需要立即干预",
                "evidence_used": ["nurse_rule:fall_detected"],
                "uncertainty": {
                    "sensing_quality": "reliable",
                    "missing_evidence": [],
                    "needs_recheck": True,
                },
                "action": {"channel": "emergency", "message": "跌倒伴随生理指标异常，需要立即干预", "recheck_after_sec": 60},
                "safety_boundary": "care_support_only",
            }

        # Extreme HR → L4 (checked before processing event_type-specific rules)
        if state.heart_rate is not None and state.heart_rate > 150:
            return {
                "level": "L4",
                "label": "emergency",
                "event_interpretation": f"心率极端异常 (HR={state.heart_rate:.0f})，需要立即干预",
                "evidence_used": ["nurse_rule:hr_extreme"],
                "uncertainty": {
                    "sensing_quality": "reliable",
                    "missing_evidence": [],
                    "needs_recheck": True,
                },
                "action": {"channel": "emergency", "message": f"心率极端异常 (HR={state.heart_rate:.0f})，需要立即干预", "recheck_after_sec": 60},
                "safety_boundary": "care_support_only",
            }

        # Critical vital patterns (L3)
        if event.event_type == "rr_bradypnea":

            # Extreme bradypnea → L4
            if state.respiration_rate is not None and state.respiration_rate < 6:
                return {
                    "level": "L4",
                    "label": "emergency",
                    "event_interpretation": f"呼吸极端过缓 (RR={state.respiration_rate})，需要立即干预",
                    "evidence_used": ["nurse_rule:rr_extreme"],
                    "uncertainty": {
                        "sensing_quality": "reliable",
                        "missing_evidence": [],
                        "needs_recheck": True,
                    },
                    "action": {"channel": "emergency", "message": f"呼吸极端过缓 (RR={state.respiration_rate})，需要立即干预", "recheck_after_sec": 60},
                    "safety_boundary": "care_support_only",
                }
            return {
                "level": "L3",
                "label": "family_notification",
                "event_interpretation": f"呼吸过缓 (RR={state.respiration_rate})，需要家属关注",
                "evidence_used": ["nurse_rule:rr_bradypnea"],
                "uncertainty": {
                    "sensing_quality": "reliable",
                    "missing_evidence": [],
                    "needs_recheck": True,
                },
                "action": {"channel": "family_push", "message": f"呼吸过缓 (RR={state.respiration_rate})，需要家属关注", "recheck_after_sec": 300},
                "safety_boundary": "care_support_only",
            }

        if event.event_type == "rr_tachypnea":

            # Extreme tachypnea → L4
            if state.respiration_rate is not None and state.respiration_rate > 40:
                return {
                    "level": "L4",
                    "label": "emergency",
                    "event_interpretation": f"呼吸极端急促 (RR={state.respiration_rate})，需要立即干预",
                    "evidence_used": ["nurse_rule:rr_extreme"],
                    "uncertainty": {
                        "sensing_quality": "reliable",
                        "missing_evidence": [],
                        "needs_recheck": True,
                    },
                    "action": {"channel": "emergency", "message": f"呼吸极端急促 (RR={state.respiration_rate})，需要立即干预", "recheck_after_sec": 60},
                    "safety_boundary": "care_support_only",
                }
            return {
                "level": "L3",
                "label": "family_notification",
                "event_interpretation": f"呼吸急促 (RR={state.respiration_rate})，需要家属关注",
                "evidence_used": ["nurse_rule:rr_tachypnea"],
                "uncertainty": {
                    "sensing_quality": "reliable",
                    "missing_evidence": [],
                    "needs_recheck": True,
                },
                "action": {"channel": "family_push", "message": f"呼吸急促 (RR={state.respiration_rate})，需要家属关注", "recheck_after_sec": 300},
                "safety_boundary": "care_support_only",
            }

        return None

    def _build_context(self, event: HealthEvent) -> list[ChatMessage]:
        state = event.state

        # Extract duration from rule_markers if available
        duration_sec = event.rule_markers.get("duration_sec", 0)
        duration_str = f", 已持续 {duration_sec}s" if duration_sec > 0 else ""

        # Extract fusion context from rule_markers
        fusion_lines = []
        for metric in ["hr", "rr"]:
            delta = event.rule_markers.get(f"{metric}_modality_delta")
            dominant = event.rule_markers.get(f"{metric}_dominant")
            if delta is not None:
                fusion_lines.append(f"  - {metric}: 模态差异 delta={delta}, 信任 {dominant}")

        fusion_str = "\n" + "\n".join(fusion_lines) if fusion_lines else ""

        context = (
            f"异常事件: {event.event_type}\n"
            f"触发原因: {event.trigger_reason}{duration_str}\n"
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
            f"- NLOS: {state.nlos_flag}\n"
            f"- 姿势: {state.posture}\n"
            f"- 传感器接触: {state.sensor_contact}\n"
            f"- 缺失模态: {state.missing_modalities}"
            f"{fusion_str}"
        )
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=context),
        ]

    @staticmethod
    def validate_triage_decision(d: dict) -> dict:
        """校验 TriageDecision schema 约束，返回校验后的 dict。
        
        校验项：
        - level ∈ L0-L4 (由 caller 确保)
        - safety_boundary == care_support_only
        - evidence_used 是 list
        - uncertainty 是 dict
        - action 是 dict
        """
        # safety_boundary 强校验
        sb = d.get("safety_boundary", "care_support_only")
        if sb != "care_support_only":
            d["safety_boundary"] = "care_support_only"
        # 类型校验
        if not isinstance(d.get("evidence_used", []), list):
            d["evidence_used"] = []
        if not isinstance(d.get("uncertainty", {}), dict):
            d["uncertainty"] = {"sensing_quality": "unknown", "missing_evidence": [], "needs_recheck": True}
        if not isinstance(d.get("action", {}), dict):
            d["action"] = {}
        return d

    def _try_parse_decision(self, content: str) -> Optional[dict]:
        """通过花括号匹配提取首个有效 JSON 决策。
        
        支持嵌套 JSON (如 action: {"channel": "screen"})，
        不像简单的 regex 会被花括号打断。
        """
        depth = 0
        start = -1
        for i, ch in enumerate(content):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        d = json.loads(content[start:i + 1])
                        if d.get("level") in self.VALID_LEVELS:
                            # Apply defaults for optional fields
                            d.setdefault("label", "")
                            d.setdefault("event_interpretation", "")
                            d.setdefault("evidence_used", [])
                            d.setdefault("uncertainty", {
                                "sensing_quality": "unknown",
                                "missing_evidence": [],
                                "needs_recheck": True,
                            })
                            d.setdefault("action", {})
                            d.setdefault("safety_boundary", "care_support_only")
                            d = self.validate_triage_decision(d)
                            return d
                    except json.JSONDecodeError:
                        start = -1  # false positive, keep scanning
        return None

    def _build_evidence(self, event: HealthEvent,
                        tool_results: list[dict]) -> dict:
        """构建完整的审计证据链"""
        return {
            "event": {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "trigger_reason": event.trigger_reason,
            },
            "rule_markers": event.rule_markers,
            "sensing_summary": {
                "heart_rate": event.state.heart_rate,
                "respiration_rate": event.state.respiration_rate,
                "body_temp": event.state.body_temp,
                "fall_status": event.state.fall_status,
                "wifi_confidence": event.state.wifi_confidence,
                "mmwave_confidence": event.state.mmwave_confidence,
                "nlos_flag": event.state.nlos_flag,
                "activity_state": event.state.activity_state,
                "thermal_confidence": event.state.thermal_confidence,
                "posture": event.state.posture,
                "sensor_contact": event.state.sensor_contact,
                "missing_modalities": event.state.missing_modalities,
            },
            "tool_results": tool_results,
            "context": {
                "duration_sec": event.context.duration_sec if event.context else 0,
                "personal_baseline": event.context.personal_baseline if event.context else None,
                "recent_windows_count": len(event.context.recent_windows) if event.context else 0,
                "has_recent_windows": bool(event.context.recent_windows) if event.context else False,
            } if event.context else {
                "duration_sec": 0, "personal_baseline": None,
                "recent_windows_count": 0, "has_recent_windows": False,
            },
        }
