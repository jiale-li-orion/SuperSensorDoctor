"""Report Agent — 周报生成 + 自然语言 Q&A"""

import json
from datetime import datetime, timedelta
from typing import Optional

from agent_layer.state_objects import EpisodeLog
from agent_layer.llm_provider import LLMProvider, ChatMessage


REPORT_PROMPT = """
你是 SuperSenseDoctor 的周报生成助手。请基于过去 7 天的临床决策记录生成一份结构完整的中文周报。

【约束】
- 只能引用下方数据中明确列出的证据，不得虚构阈值或来源
- 每一条 decision 的 clinical_basis 字段包含决策依据，标注了来源标签：
  RCP_NEWS2_2017_REFERENCE = 绝对生理参考区间
  RESIDENT_HISTORY = 个人纵向基线
  SENSOR_FUSION = 传感融合可靠性
  ACTIVITY_CONTEXT = 活动/姿势上下文
  PROJECT_POLICY = 项目特有策略
  NICE_NG249_2025 = 跌倒评估指南
- 如果没有决策记录，只输出"本周无诊断记录。"

【输出结构】

## 📊 本周摘要
<总体统计：诊断总数、最高风险级别、最高频级别>

## 📋 决策详情
<按级别列出主要事件，每条附 clinical_basis 来源标签和关键发现>

## 🔬 传感质量
<NLOS 遮挡、低置信度、模态冲突的次数（有则写，无则跳过）>

## 📌 重点关注
<本周最值得关注的临床发现和后续建议>
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
                nlos_count = 0
                low_conf_count = 0
                conflict_count = 0
                wifi_confs = []
                mmwave_confs = []
                for e in events:
                    et = e.get("event_type", "unknown") if isinstance(e, dict) else getattr(e, "event_type", "unknown")
                    evt_types[et] = evt_types.get(et, 0) + 1
                    if et == "nlos_occlusion": nlos_count += 1
                    if et == "low_confidence": low_conf_count += 1
                    if et == "modality_conflict": conflict_count += 1
                    wc = e.get("wifi_confidence") if isinstance(e, dict) else getattr(e, "wifi_confidence", None)
                    mc = e.get("mmwave_confidence") if isinstance(e, dict) else getattr(e, "mmwave_confidence", None)
                    if wc is not None: wifi_confs.append(float(wc))
                    if mc is not None: mmwave_confs.append(float(mc))
                context["event_types"] = evt_types
                context["sensing_quality"] = {
                    "nlos_count": nlos_count,
                    "low_conf_count": low_conf_count,
                    "conflict_count": conflict_count,
                    "avg_wifi_confidence": round(sum(wifi_confs)/len(wifi_confs), 2) if wifi_confs else None,
                    "avg_mmwave_confidence": round(sum(mmwave_confs)/len(mmwave_confs), 2) if mmwave_confs else None,
                }

            # Action breakdown
            action_channels = {}
            for ep in recent:
                action = ep.get("action", {}) if isinstance(ep, dict) else getattr(ep, "action", {})
                if isinstance(action, str):
                    try:
                        action = json.loads(action)
                    except (json.JSONDecodeError, TypeError):
                        action = {}
                ch = action.get("channel", "none") if isinstance(action, dict) else "none"
                action_channels[ch] = action_channels.get(ch, 0) + 1
            if action_channels:
                context["action_channels"] = action_channels

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

        # ── Rule-based fallback (enhanced, markdown format) ──
        level_label = {"L0": "记录", "L1": "观察", "L2": "提醒", "L3": "通知", "L4": "紧急"}
        source_label = {
            "RCP_NEWS2_2017_REFERENCE": "临床参考",
            "RESIDENT_HISTORY": "个人基线",
            "SENSOR_FUSION": "传感融合",
            "ACTIVITY_CONTEXT": "活动上下文",
            "PROJECT_POLICY": "项目策略",
            "NICE_NG249_2025": "跌倒评估",
        }
        most_severe = max(level_counts, key=lambda k: (level_counts[k], k)) if any(level_counts.values()) else "L0"
        most_frequent = max(level_counts, key=level_counts.get) if any(level_counts.values()) else "L0"
        lines = [f"## 📊 本周摘要\n\n共 **{len(recent)}** 次诊断。"]
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            if level_counts.get(level, 0) > 0:
                lines.append(f"- **{level_label[level]}**（{level}）: {level_counts[level]} 次")
        lines.append(f"\n最高风险: **{level_label[most_severe]}**（{most_severe}）")
        lines.append(f"最高频: **{level_label[most_frequent]}**（{most_frequent}）")

        # ── Clinical basis sources referenced ──
        sources_seen = set()
        for entry in decision_entries:
            for cb in entry.get("clinical_basis", []):
                src = cb.get("source", "")
                if src and src not in sources_seen:
                    sources_seen.add(src)
        if sources_seen:
            lines.append("\n**参考来源**: " + ", ".join(
                f"{source_label.get(s, s)}（{s}）" for s in sorted(sources_seen)))

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

            lines.append(f"\n### 🔬 传感质量")
            if nlos_count > 0:
                lines.append(f"- NLOS 遮挡: **{nlos_count}** 次")
            if low_conf_count > 0:
                lines.append(f"- 低置信度: **{low_conf_count}** 次")
            if conflict_count > 0:
                lines.append(f"- 模态冲突: **{conflict_count}** 次")
            if wifi_confs:
                avg_wifi = sum(wifi_confs) / len(wifi_confs)
                lines.append(f"- 平均 WiFi 置信度: **{avg_wifi:.0%}**")
            if mmwave_confs:
                avg_mmwave = sum(mmwave_confs) / len(mmwave_confs)
                lines.append(f"- 平均 mmWave 置信度: **{avg_mmwave:.0%}**")

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
