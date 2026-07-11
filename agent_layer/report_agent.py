"""Report Agent — 周报生成 + 自然语言 Q&A"""

from datetime import datetime, timedelta
from typing import Optional

from agent_layer.state_objects import EpisodeLog


class ReportAgent:
    """非 LLM 报告生成（关键词匹配 Q&A + 规则统计）"""

    def generate_weekly_report(self, episodes: list[EpisodeLog],
                               events: Optional[list] = None) -> str:
        """生成周报摘要。
        
        Args:
            episodes: 诊断记录列表
            events: 可选的事件列表 (含 rule_markers, 用于计算融合/置信度统计)
        """
        recent = [e for e in episodes if e.start_time >= datetime.now() - timedelta(days=7)]

        if not recent:
            return "本周无诊断记录。"

        total = len(recent)
        level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
        for ep in recent:
            level = ep.decision.get("level", "L0")
            level_counts[level] = level_counts.get(level, 0) + 1

        lines = [f"本周报告（{total} 次诊断）"]
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            if level_counts.get(level, 0) > 0:
                lines.append(f"  {level}: {level_counts[level]} 次")
        most_severe = max(level_counts, key=lambda k: (level_counts[k], k))
        lines.append(f"  最严重级别: {most_severe}")

        # ── 置信度与融合统计 (Phase H) ──
        if events:
            # Helper to read from dict or object (backward compat)
            def _ev(e, attr, default=None):
                return e.get(attr, default) if isinstance(e, dict) else getattr(e, attr, default)

            total_events = len(events)
            nlos_count = sum(1 for e in events if _ev(e, 'rule_markers', {}).get("nlos_flag") or
                           _ev(e, 'event_type') == "nlos_occlusion")
            low_conf_count = sum(1 for e in events if _ev(e, 'event_type') == "low_confidence")
            conflict_count = sum(1 for e in events if _ev(e, 'event_type') == "modality_conflict")

            # Average confidence from events that have it (dict: top-level keys; object: .state attr)
            def _wifi_conf(e):
                if isinstance(e, dict):
                    return e.get("wifi_confidence")
                if hasattr(e, 'state') and e.state:
                    return getattr(e.state, 'wifi_confidence', None)
                return None

            def _mmwave_conf(e):
                if isinstance(e, dict):
                    return e.get("mmwave_confidence")
                if hasattr(e, 'state') and e.state:
                    return getattr(e.state, 'mmwave_confidence', None)
                return None

            wifi_confs = [_wifi_conf(e) for e in events]
            wifi_confs = [c for c in wifi_confs if c is not None]
            mmwave_confs = [_mmwave_conf(e) for e in events]
            mmwave_confs = [c for c in mmwave_confs if c is not None]

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
