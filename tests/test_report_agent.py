import pytest
from datetime import datetime, timedelta
from agent_layer.report_agent import ReportAgent
from agent_layer.state_objects import EpisodeLog


def make_ep(level: str, days_ago: int = 0) -> EpisodeLog:
    t = datetime.now() - timedelta(days=days_ago)
    return EpisodeLog(
        episode_id=f"ep_{level}_{days_ago}",
        resident_id="resident_01",
        start_time=t,
        end_time=t + timedelta(minutes=1),
        decision={"level": level, "explanation": "test"},
        action={"channel": "none"},
        audit={},
    )


class TestReportAgent:
    def test_weekly_report(self):
        episodes = [
            make_ep("L4", 0),
            make_ep("L2", 1),
            make_ep("L1", 3),
            make_ep("L0", 5),
            make_ep("L0", 10),  # 超出 7 天
        ]
        agent = ReportAgent()
        report = agent.generate_weekly_report(episodes)
        assert "本周" in report
        assert "L4" in report

    def test_answer_known_question(self):
        agent = ReportAgent()
        ans = agent.answer_question("发烧了怎么办？")
        assert "体温" in ans
        assert "医生" in ans

    def test_answer_unknown_question(self):
        agent = ReportAgent()
        ans = agent.answer_question("今天天气怎么样？")
        assert "无法回答" in ans
