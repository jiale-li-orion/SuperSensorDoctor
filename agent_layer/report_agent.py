"""Report Agent — 周报生成 + 自然语言 Q&A"""

from datetime import datetime, timedelta
from typing import Optional

from agent_layer.state_objects import EpisodeLog


class ReportAgent:
    """非 LLM 报告生成（关键词匹配 Q&A + 规则统计）"""

    def generate_weekly_report(self, episodes: list[EpisodeLog]) -> str:
        """生成周报摘要 (关键词匹配, 暂不调用 LLM)"""
        recent = [e for e in episodes if e.start_time >= datetime.now() - timedelta(days=7)]

        if not recent:
            return "本周无诊断记录。"

        total = len(recent)
        level_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "L4": 0}
        for ep in recent:
            level = ep.decision.get("level", "L0")
            level_counts[level] = level_counts.get(level, 0) + 1

        lines = [f"📊 本周报告（{total} 次诊断）"]
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            if level_counts.get(level, 0) > 0:
                lines.append(f"  {level}: {level_counts[level]} 次")
        lines.append(f"  最严重级别: {max(level_counts, key=lambda k: (level_counts[k], k))}")
        return "\n".join(lines)

    # ── 自然语言 Q&A (暂用关键词匹配, 后续替换为 LLM) ──

    QA_PAIRS = {
        "发烧": "建议测量体温，若持续超过 38.5°C 请咨询医生。",
        "心率": "静息心率正常范围 60-100 bpm，持续偏高建议休息后复测。",
        "跌倒": "跌倒事件已记录。确认老人当前状态，必要时联系家属。",
        "血压": "血压数据暂未接入。建议使用专用血压计测量。",
    }

    def answer_question(self, question: str) -> str:
        """回答居民/家属的提问 (关键词匹配)"""
        for keyword, answer in self.QA_PAIRS.items():
            if keyword in question:
                return answer
        return "抱歉，我暂时无法回答这个问题。建议联系医生或查看健康手册。"
