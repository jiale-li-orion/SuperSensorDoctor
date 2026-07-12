"""Report Agent — 周报生成 + 自然语言 Q&A"""

import json
from datetime import datetime, timedelta
from typing import Optional

from agent_layer.state_objects import EpisodeLog
from agent_layer.llm_provider import LLMProvider, ChatMessage


REPORT_PROMPT = """
You are a home-care shift report writer for SuperSenseDoctor.

Your task: produce a concise Chinese-language weekly summary based on
structured clinical decision records from the past 7 days.

Rules:
- Only reference evidence explicitly listed below. Do not invent thresholds.
- Each decision has a "clinical_basis" field. Cite it by source tag.
- Distinguish between "absolute reference" (NEWS2), "personal baseline"
  (RESIDENT_HISTORY), "sensing quality" (SENSOR_FUSION), and "activity
  context" (ACTIVITY_CONTEXT).
- If no decisions occurred this week, output "本周无诊断记录。"
- Keep the report to 2-3 paragraphs.
- End with the most notable clinical finding, if any.
"""


class ReportAgent:
    """周报生成（LLM + 规则降级）"""

    async def generate_weekly_report(
        self,
        episodes: list[EpisodeLog],
        events: Optional[list] = None,
        llm_provider: Optional[LLMProvider] = None,
        reference_ts: Optional[datetime] = None,
    ) -> str:
        """生成周报摘要。

        Args:
            episodes: 诊断记录列表（dict with decision, evidence, action）
            events: 可选的事件列表 (含 rule_markers)
            llm_provider: 可选 LLM，传入时用 LLM 生成自然语言报告
            reference_ts: 参考时间戳，用于过滤最近 7 天
        """
        ref = reference_ts or datetime.now()
        # Strip tz for naive comparison
        if ref.tzinfo is not None:
            ref = ref.replace(tzinfo=None)
        cutoff = ref - timedelta(days=7)

        def _should_include(ep):
            if isinstance(ep, dict):
                ts_str = ep.get("start_time")
            else:
                ts_str = getattr(ep, "start_time", None)
            if ts_str is None:
                return False
            if isinstance(ts_str, datetime):
                return ts_str >= cutoff
            try:
                dt = datetime.fromisoformat(str(ts_str))
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt >= cutoff
            except (ValueError, TypeError):
                return False

        recent = [e for e in episodes if _should_include(e)]
        if not recent:
            return "本周无诊断记录。"

        # ── Build structured summary ──
        level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
        decision_entries = []

        for ep in recent:
            dec = ep.get("decision", {}) if isinstance(ep, dict) else getattr(ep, "decision", {})
            if isinstance(dec, str):
                try:
                    dec = json.loads(dec)
                except (json.JSONDecodeError, TypeError):
                    dec = {}
            level = dec.get("level", "L0")[:2]
            if level in level_counts:
                level_counts[level] += 1

            entry = {
                "level": level,
                "label": dec.get("label", ""),
                "event_interpretation": dec.get("event_interpretation", ""),
                "clinical_basis": dec.get("clinical_basis", []),
            }
            decision_entries.append(entry)

        # ── LLM path ──
        if llm_provider is not None:
            context = {
                "period": f"{cutoff.isoformat()} ~ {ref.isoformat()}",
                "total_decisions": len(recent),
                "level_counts": level_counts,
                "decisions": decision_entries,
            }

            if events:
                evt_types = {}
                for e in events:
                    et = e.get("event_type", "unknown") if isinstance(e, dict) else getattr(e, "event_type", "unknown")
                    evt_types[et] = evt_types.get(et, 0) + 1
                context["event_types"] = evt_types

            try:
                messages = [
                    ChatMessage(role="system", content=REPORT_PROMPT),
                    ChatMessage(role="user", content=json.dumps(context, ensure_ascii=False, indent=2, default=str)),
                ]
                resp = await llm_provider.chat(messages)
                if resp and resp.content:
                    return resp.content.strip()
            except Exception:
                pass  # fall through to rule-based

        # ── Rule-based fallback (enhanced) ──
        lines = [f"本周报告（{len(recent)} 次诊断）"]
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            if level_counts.get(level, 0) > 0:
                lines.append(f"  {level}: {level_counts[level]} 次")
        most_severe = max(level_counts, key=lambda k: (level_counts[k], k))
        lines.append(f"  最严重级别: {most_severe}")

        # ── Clinical basis per level ──
        sources_seen = set()
        for entry in decision_entries:
            for cb in entry.get("clinical_basis", []):
                src = cb.get("source", "")
                if src and src not in sources_seen:
                    sources_seen.add(src)
                    lines.append(f"  - 参考来源: {src}")

        # ── Confidence stats from events ──
        if events:
            def _ev(e, attr, default=None):
                return e.get(attr, default) if isinstance(e, dict) else getattr(e, attr, default)

            nlos_count = sum(1 for e in events if _ev(e, 'rule_markers', {}).get("nlos_flag") or
                           _ev(e, 'event_type') == "nlos_occlusion")
            low_conf_count = sum(1 for e in events if _ev(e, 'event_type') == "low_confidence")
            conflict_count = sum(1 for e in events if _ev(e, 'event_type') == "modality_conflict")

            wifi_confs = []
            mmwave_confs = []
            for e in events:
                wc = _ev(e, "wifi_confidence")
                mc = _ev(e, "mmwave_confidence")
                if wc is not None: wifi_confs.append(float(wc))
                if mc is not None: mmwave_confs.append(float(mc))

            lines.append(f"\n  传感状态:")
            if nlos_count > 0:
                lines.append(f"  - NLOS 遮挡: {nlos_count} 次")
            if low_conf_count > 0:
                lines.append(f"  - 低置信度: {low_conf_count} 次")
            if conflict_count > 0:
                lines.append(f"  - 模态冲突: {conflict_count} 次")
            if wifi_confs:
                avg_wifi = sum(wifi_confs) / len(wifi_confs)
                lines.append(f"  - 平均 WiFi 置信度: {avg_wifi:.2f}")
            if mmwave_confs:
                avg_mmwave = sum(mmwave_confs) / len(mmwave_confs)
                lines.append(f"  - 平均 mmWave 置信度: {avg_mmwave:.2f}")

        return "\n".join(lines)

    # ── 自然语言 Q&A (暂用关键词匹配, 后续替换为 LLM) ──

    QA_PAIRS = {
        "发烧": "建议测量体温，若持续超过 38.5°C 请咨询医生。",
        "心率": "静息心率正常范围 60-100 bpm，持续偏高建议休息后复测。",
        "跌倒": "跌倒事件已记录。确认老人当前状态，必要时联系家属。",
        "血压": "血压数据暂未接入。建议使用专用血压计测量。",
        "模态": "系统使用 Wi-Fi 传感和毫米波雷达双模态监测，经过 FusionEngine 置信度融合仲裁后输出体征估计。",
        "置信度": "传感置信度受 NLOS 遮挡、传感器接触质量和环境干扰影响。LLM 决策时会参考置信度进行分级。",
    }

    def answer_question(self, question: str) -> str:
        """回答居民/家属的提问 (关键词匹配)"""
        for keyword, answer in self.QA_PAIRS.items():
            if keyword in question:
                return answer
        return "抱歉，我暂时无法回答这个问题。建议联系医生或查看健康手册。"
