"""Tests for the Report Agent — canonical blocks, LLM path, fallback path."""

import pytest
from datetime import datetime, timedelta

from agent_layer.report_agent import (
    ReportAgent,
    REPORT_BLOCKS,
    build_report_context,
    render_fallback_report,
)
from agent_layer.llm_provider import MockProvider
from agent_layer.state_objects import EpisodeLog


def make_ep(level: str, days_ago: int = 0, **overrides) -> EpisodeLog:
    t = datetime.now() - timedelta(days=days_ago)
    base = dict(
        episode_id=f"ep_{level}_{days_ago}",
        resident_id="resident_01",
        start_time=t,
        end_time=t + timedelta(minutes=1),
        decision={"level": level, "explanation": "test"},
        action={"channel": "none"},
        audit={},
    )
    base.update(overrides)
    return EpisodeLog(**base)


class TestReportStructure:
    @pytest.mark.asyncio
    async def test_weekly_report_uses_canonical_blocks(self):
        episodes = [
            make_ep("L4", 0),
            make_ep("L2", 1),
            make_ep("L1", 3),
            make_ep("L0", 5),
            make_ep("L0", 10),  # outside the 7-day window
        ]
        report = await ReportAgent().generate_weekly_report(episodes)
        assert "Weekly Summary" in report
        assert "L4" in report
        for heading in (
            "Evidence Trail",
            "Sensing Quality and Fusion Arbitration",
            "Action Routing",
            "Uncertainty and Reflex Path",
            "Vital Trends",
            "Personal Baseline",
            "Published Validation Results",
            "Indicator and Screening Coverage",
            "Privacy and Human Oversight",
        ):
            assert heading in report, heading

    def test_window_excludes_old_episodes(self):
        episodes = [make_ep("L3", 0), make_ep("L3", 30)]
        context = build_report_context(episodes)
        assert context["summary"]["total_records"] == 1

    @pytest.mark.asyncio
    async def test_empty_period_reports_no_records(self):
        report = await ReportAgent().generate_weekly_report([])
        assert "No triage records" in report
        assert "Published Validation Results" in report

    def test_context_blocks_match_declared_order(self):
        context = build_report_context([])
        for block in REPORT_BLOCKS:
            assert block in context, block


class TestEvidenceTrail:
    def test_trail_links_trigger_to_episode(self):
        ep = make_ep(
            "L3",
            0,
            evidence={
                "event": {
                    "event_id": "evt_1",
                    "event_type": "fall_detected",
                    "trigger_reason": "Fall detected with elevated HR",
                }
            },
            decision={
                "level": "L3",
                "event_interpretation": "Fall with sustained HR elevation",
                "clinical_basis": [
                    {"type": "fall_context", "finding": "Fall recorded",
                     "source": "NICE_NG249_2025"}
                ],
                "safety_boundary": "care_support_only",
            },
            action={"channel": "family_push"},
            audit={"tools_called": ["read_sensing_state", "write_episode"],
                   "step_count": 3, "reflex": False},
        )
        context = build_report_context([ep])
        trail = context["evidence_trail"]
        assert trail["trigger"]["event_id"] == "evt_1"
        assert trail["tier"] == "L3"
        assert trail["channel"] == "family_push"
        assert trail["anchors"][0]["source_label"].startswith("Fall assessment")
        assert trail["episode_id"] == "ep_L3_0"

    def test_reflex_flag_surfaces(self):
        ep = make_ep("L4", 0, audit={"reflex": True, "step_count": 0})
        context = build_report_context([ep])
        assert context["evidence_trail"]["reflex"] is True
        assert "reflex path" in render_fallback_report(context)


class TestSensingQuality:
    def test_quality_events_counted_and_explained(self):
        events = [
            {"event_type": "nlos_occlusion"},
            {"event_type": "low_confidence"},
            {"event_type": "modality_conflict"},
            {"event_type": "modality_conflict"},
        ]
        context = build_report_context([make_ep("L1", 0)], events=events)
        quality = context["sensing_quality"]
        assert quality["nlos_count"] == 1
        assert quality["low_confidence_count"] == 1
        assert quality["modality_conflict_count"] == 2
        assert "not health status" in quality["note"]

    def test_dominant_branch_extracted_from_markers(self):
        events = [{"event_type": "hr_abnormal",
                   "rule_markers": {"hr_dominant": "wifi"}}]
        context = build_report_context([make_ep("L2", 0)], events=events)
        assert context["sensing_quality"]["dominant_branches"][0]["branch"] == "wifi"


class TestUncertainty:
    def test_missing_evidence_and_recheck_aggregated(self):
        ep = make_ep("L2", 0, decision={
            "level": "L2",
            "uncertainty": {
                "sensing_quality": "degraded",
                "missing_evidence": ["trend_analysis", "trend_analysis"],
                "needs_recheck": True,
            },
        })
        context = build_report_context([ep])
        unc = context["uncertainty"]
        assert unc["needs_recheck_count"] == 1
        assert unc["missing_evidence"][0]["count"] == 2
        assert unc["sensing_quality_levels"][0]["level"] == "degraded"


class TestLlmPath:
    @pytest.mark.asyncio
    async def test_llm_output_preferred(self):
        provider = MockProvider(response="## Weekly Summary\n\nLLM prose.")
        report = await ReportAgent().generate_weekly_report(
            [make_ep("L2", 0)], llm_provider=provider
        )
        assert report == "## Weekly Summary\n\nLLM prose."

    @pytest.mark.asyncio
    async def test_provider_failure_falls_back(self):
        class Boom:
            async def chat(self, messages):
                raise RuntimeError("provider down")

        report = await ReportAgent().generate_weekly_report(
            [make_ep("L2", 0)], llm_provider=Boom()
        )
        assert "Weekly Summary" in report
        assert "Published Validation Results" in report

    @pytest.mark.asyncio
    async def test_empty_llm_content_falls_back(self):
        provider = MockProvider(response="   ")
        report = await ReportAgent().generate_weekly_report(
            [make_ep("L2", 0)], llm_provider=provider
        )
        assert "Weekly Summary" in report


class TestQa:
    def test_answer_known_question(self):
        ans = ReportAgent().answer_question("How is heart rate measured?")
        assert "WiFi BFI" in ans

    def test_answer_unknown_question(self):
        ans = ReportAgent().answer_question("What is the weather today?")
        assert "care-support scope" in ans
