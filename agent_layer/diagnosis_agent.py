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
from agent_layer.clinical_policy import (
    build_clinical_summary,
    default_clinical_basis,
    ALLOWED_CLINICAL_SOURCES,
)


SYSTEM_PROMPT = """
你是 SuperSenseDoctor 的 evidence-grounded home-care triage agent。

你的任务是：
基于结构化传感证据、个人历史、时间持续性、活动上下文、
跨模态可靠性和系统提供的临床参考框架，对家庭看护事件进行分级。

你不进行疾病诊断，不开药，不提供治疗处方。
你的输出属于 care support 和 risk triage。

【最重要原则】

1. 区分"测量是否可信"和"生理状态是否异常"
   - 传感器低置信度不能直接解释成生理恶化。
   - 跨模态一致只能增强 measurement confidence，
     不能证明某一种具体疾病。
   - 如果异常数值来自 unreliable sensing，
     优先标记 uncertainty、missing evidence 和 recheck。

2. 区分 absolute abnormality 和 personal deviation
   - absolute_reference 来自系统提供的临床参考区间。
   - personal_baseline 来自该居民自己的历史。
   - 个人 z-score 是统计偏离证据，不等于疾病诊断。
   - 一个值即使没有越过绝对危险区间，
     仍可能相对于个人长期基线明显异常。

3. 区分 transient fluctuation 和 persistent deviation
   - 单个窗口不能自动证明持续恶化。
   - duration_sec 越长，持续性证据越强。
   - 必须明确当前是否有持续性证据。

4. 必须考虑 activity 和 posture
   - walking、running、刚起身等活动可能解释 HR/RR 暂时升高。
   - rest、lying、sleep context 下持续 HR/RR 升高更值得关注。
   - 不允许忽略 activity_state。

5. 跌倒事件必须考虑证据缺口
   - 系统可能知道 fall_status，但通常不知道：
     injury、loss of consciousness、是否能自行起身、
     是否有反复跌倒历史。
   - 缺失这些信息时，写入 uncertainty.missing_evidence。
   - 不允许虚构这些信息。

6. 临床参考来源边界
   - RCP NEWS2 reference 只用于 HR/RR/temperature 的绝对偏离参考。
   - 当前系统没有完整 NEWS2 所需参数，
     禁止声称"已计算 NEWS2 总分"。
   - NICE NG249 用于跌倒相关证据需求。
   - personal z-score、5-10 分钟复查和 L0-L4 映射属于项目策略，
     不得伪装成临床指南原文。

【证据优先顺序】

1. 已配置的 deterministic safety/reflex rule
2. sensing quality 和 failure flags
3. absolute physiological reference
4. personal baseline deviation
5. persistence / trend
6. activity / posture context
7. cross-modal agreement or conflict
8. historical episodes and previous decisions

【工具使用】

read_sensing_state:
读取当前结构化传感状态。

query_history:
读取个人历史、baseline 和 trend。

consult_fusion:
当 WiFi 和 mmWave 冲突、NLOS、低置信度时，
读取跨模态仲裁结果。

不要为了"显得像医生"而调用工具。
只在当前证据不足时调用。

【决策原则】

L0:
无值得行动的异常证据，仅记录。

L1:
轻度或短暂偏离；
证据尚不足；
需要观察或短期复查。

L2:
有可信异常，需要提醒居民确认状态。

L3:
存在持续异常、多个风险证据叠加，
或需要家属关注。

L4:
只用于明确的高风险 safety/reflex condition
或非常强的复合高风险证据。
不要因为单个轻度异常值直接输出 L4。

选择与现有证据相符的最低干预等级。
不确定时必须明确 uncertainty，不要伪造确定性。

【输出】

只输出 JSON：

{
  "level": "L0-L4",
  "label": "continuous_observation | resident_alert | family_notification | emergency",
  "event_interpretation": "基于证据的看护风险解释",

  "evidence_used": [
    "实际使用过的工具名称"
  ],

  "clinical_basis": [
    {
      "type": "absolute_reference | personal_baseline | persistence | activity_context | sensing_quality | fall_context",
      "finding": "具体证据及其解释",
      "source": "RCP_NEWS2_2017_REFERENCE | NICE_NG249_2025 | RESIDENT_HISTORY | SENSOR_FUSION | ACTIVITY_CONTEXT | PROJECT_POLICY"
    }
  ],

  "uncertainty": {
    "sensing_quality": "reliable | degraded | unreliable | unknown",
    "missing_evidence": [],
    "needs_recheck": true
  },

  "action": {
    "channel": "none | screen | family_push | emergency",
    "recheck_after_sec": 300
  },

  "safety_boundary": "care_support_only"
}
"""


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
            clinical_summary = build_clinical_summary(event)
            reflex_decision.setdefault(
                "clinical_basis",
                default_clinical_basis(clinical_summary),
            )
            evidence = self._build_evidence(event, [], clinical_summary)
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
                clinical_summary = build_clinical_summary(event)
                evidence = self._build_evidence(event, tool_results, clinical_summary)

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
        clinical_summary = build_clinical_summary(event)
        evidence = self._build_evidence(event, tool_results, clinical_summary)
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
        clinical_summary = build_clinical_summary(event)
        state = event.state

        context = {
            "event": {
                "event_type": event.event_type,
                "trigger_reason": event.trigger_reason,
                "timestamp": event.timestamp.isoformat(),
                "rule_markers": event.rule_markers,
            },

            "resident_id": self.resident_id,

            "current_state": {
                "heart_rate": state.heart_rate,
                "respiration_rate": state.respiration_rate,
                "body_temp": state.body_temp,

                "activity_state": state.activity_state,
                "posture": state.posture,
                "fall_status": state.fall_status,

                "wifi_confidence": state.wifi_confidence,
                "mmwave_confidence": state.mmwave_confidence,
                "thermal_confidence": state.thermal_confidence,

                "nlos_flag": state.nlos_flag,
                "missing_modalities": state.missing_modalities,

                "hr_wifi": state.hr_wifi,
                "hr_mm": state.hr_mm,
                "rr_wifi": state.rr_wifi,
                "rr_mm": state.rr_mm,

                "hr_conf": state.hr_conf,
                "rr_conf": state.rr_conf,
                "quality_event": state.quality_event,

                "hr_source": state.hr_source,
                "rr_source": state.rr_source,
            },

            "clinical_summary": clinical_summary,
        }

        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=json.dumps(
                    context,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            ),
        ]

    @staticmethod
    def validate_triage_decision(d: dict) -> dict:
        """校验 TriageDecision schema 约束，返回校验后的 dict。
        
        校验项：
        - level ∈ L0-L4 (由 caller 确保)
        - safety_boundary == care_support_only
        - evidence_used 是 list
        - clinical_basis 是 list, 每个元素有 allowed source
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

        # clinical_basis 校验
        basis = d.get("clinical_basis", [])
        if not isinstance(basis, list):
            basis = []
        validated_basis = []
        for item in basis:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            if source not in ALLOWED_CLINICAL_SOURCES:
                continue
            validated_basis.append({
                "type": str(item.get("type", "")),
                "finding": str(item.get("finding", "")),
                "source": source,
            })
        d["clinical_basis"] = validated_basis

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
                            d.setdefault("clinical_basis", [])
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
                        tool_results: list[dict],
                        clinical_summary: Optional[dict] = None) -> dict:
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
                "hr_wifi": event.state.hr_wifi,
                "hr_mm": event.state.hr_mm,
                "rr_wifi": event.state.rr_wifi,
                "rr_mm": event.state.rr_mm,
                "hr_conf": event.state.hr_conf,
                "rr_conf": event.state.rr_conf,
                "quality_event": event.state.quality_event,
                "hr_source": event.state.hr_source,
                "rr_source": event.state.rr_source,
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
            "clinical_summary": clinical_summary or build_clinical_summary(event),
        }
