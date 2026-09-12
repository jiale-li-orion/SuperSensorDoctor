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


class TestBilingual:
    """The report must render as pure Chinese or pure English, never mixed."""

    def _episodes(self):
        return [make_ep("L3", 0, decision={
            "level": "L3",
            "event_interpretation": "Fall with sustained HR elevation",
            "clinical_basis": [
                {"type": "fall_context", "finding": "Fall recorded",
                 "source": "NICE_NG249_2025"}
            ],
            "uncertainty": {
                "sensing_quality": "degraded",
                "missing_evidence": ["trend_analysis"],
                "needs_recheck": True,
            },
        }, action={"channel": "family_push"},
           audit={"reflex": False, "step_count": 3,
                  "tools_called": ["read_sensing_state"]})]

    def test_english_report(self):
        ctx = build_report_context(self._episodes(), lang="en")
        report = render_fallback_report(ctx, "en")
        assert "Weekly Summary" in report
        assert "Evidence Trail" in report
        assert "Published Validation Results" in report
        assert "本周" not in report

    def test_chinese_report(self):
        ctx = build_report_context(self._episodes(), lang="zh")
        report = render_fallback_report(ctx, "zh")
        assert "本周摘要" in report
        assert "证据链" in report
        assert "论文已发表验证结果" in report
        assert "Weekly Summary" not in report

    def test_chinese_labels_are_localized(self):
        ctx = build_report_context(self._episodes(), lang="zh")
        assert ctx["evidence_trail"]["tier_label"] == "家属告警"
        assert ctx["evidence_trail"]["channel_label"].startswith("家属推送")
        assert ctx["evidence_trail"]["anchors"][0]["source_label"].startswith("跌倒评估")
        assert ctx["sensing_quality"]["note"].startswith("质量事件")

    def test_english_labels_are_localized(self):
        ctx = build_report_context(self._episodes(), lang="en")
        assert ctx["evidence_trail"]["tier_label"] == "Family notification"
        assert ctx["evidence_trail"]["anchors"][0]["source_label"].startswith("Fall assessment")

    def test_lang_defaults_to_english(self):
        ctx = build_report_context(self._episodes())
        assert ctx["lang"] == "en"
        assert "Weekly Summary" in render_fallback_report(ctx)

    def test_unknown_lang_falls_back_to_english(self):
        ctx = build_report_context(self._episodes(), lang="de")
        assert ctx["lang"] == "en"

    @pytest.mark.asyncio
    async def test_chinese_report_via_agent(self):
        report = await ReportAgent().generate_weekly_report(
            self._episodes(), lang="zh"
        )
        assert "本周摘要" in report
        assert "Weekly Summary" not in report

    def test_qa_answers_in_both_languages(self):
        agent = ReportAgent()
        assert "WiFi BFI" in agent.answer_question("How is heart rate measured?", "en")
        assert "WiFi BFI" in agent.answer_question("心率是怎么测的？", "zh")
        assert "看护支持范围" in agent.answer_question("今天天气？", "zh")

    def test_qa_matches_chinese_keywords_regardless_of_answer_language(self):
        agent = ReportAgent()
        # A Chinese question asked while the UI is in English still matches.
        assert "thermometer" in agent.answer_question("体温怎么测的？", "en").lower()


class TestQualityRateDeduplication:
    """The quality rate must be a bounded record share, not a signal ratio."""

    @staticmethod
    def _ep(episode_id, event_id, nlos=False, quality=False):
        # Episodes arrive from the DB as dicts and DO carry event_id, which is
        # what lets an event and its episode evidence dedupe to one signal.
        return {
            "episode_id": episode_id,
            "event_id": event_id,
            "resident_id": "resident_01",
            "start_time": datetime.now().isoformat(),
            "decision": {"level": "L1"},
            "action": {"channel": "none"},
            "audit": {},
            "evidence": {"sensing_summary": {"nlos_flag": nlos,
                                             "quality_event": int(quality)}},
        }

    def test_same_nlos_is_not_double_counted(self):
        # One NLOS occurrence: seen as a HealthEvent AND in episode evidence.
        records = [self._ep("ep1", "evt_1", nlos=True)]
        events = [{"event_id": "evt_1", "event_type": "nlos_occlusion"},
                  {"event_id": "evt_1", "event_type": "nlos_occlusion"}]
        q = build_report_context(records, events)["sensing_quality"]
        assert q["nlos_count"] == 1

    def test_rate_never_exceeds_100_percent(self):
        # Many distinct signals crammed into one record: the case that
        # previously produced a >100% rate.
        records = [self._ep("ep1", "evt_1", nlos=True, quality=True)]
        events = [
            {"event_id": "e1", "event_type": "low_confidence"},
            {"event_id": "e2", "event_type": "modality_conflict"},
            {"event_id": "e3", "event_type": "low_confidence"},
            {"event_id": "e4", "event_type": "modality_conflict"},
        ]
        q = build_report_context(records, events)["sensing_quality"]
        assert q["quality_event_rate_pct"] <= 100.0
        assert q["quality_flag_count"] <= q["total_records"]
        # The signal total may legitimately exceed the record count.
        assert q["quality_signal_count"] > q["quality_flag_count"]

    def test_signal_count_is_separate_from_flag_count(self):
        records = [self._ep(f"ep{i}", f"evt{i}") for i in range(4)]
        events = [{"event_id": "e1", "event_type": "modality_conflict"},
                  {"event_id": "e2", "event_type": "modality_conflict"}]
        q = build_report_context(records, events)["sensing_quality"]
        assert q["quality_signal_count"] == 2
        assert q["quality_flag_count"] == 0
        assert q["quality_event_rate_pct"] == 0.0


class TestDecisionPathInference:
    """A zero-step record must never be labelled a deliberative ReAct loop."""

    def test_explicit_reflex_flag_wins(self):
        ep = make_ep("L4", 0, audit={"reflex": True, "step_count": 0,
                                     "tools_called": ["nurse_rule"]})
        trail = build_report_context([ep])["evidence_trail"]
        assert trail["reflex"] is True
        assert trail["reflex_inferred"] is False

    def test_legacy_seed_record_is_inferred_as_reflex(self):
        # No `reflex` key, zero steps, deterministic tools only.
        ep = make_ep("L2", 0, audit={"step_count": 0, "tools_called": ["nurse:seed"]})
        trail = build_report_context([ep])["evidence_trail"]
        assert trail["reflex"] is True
        assert trail["reflex_inferred"] is True
        assert "inferred" in render_fallback_report(build_report_context([ep]), "en")

    def test_real_deliberation_is_not_misclassified(self):
        ep = make_ep("L2", 0, audit={"step_count": 3,
                                     "tools_called": ["read_sensing_state",
                                                      "query_history"]})
        trail = build_report_context([ep])["evidence_trail"]
        assert trail["reflex"] is False
        assert trail["reflex_inferred"] is False


class TestQualityFlagAssociation:
    """A record whose triggering event IS a quality event counts as flagged.

    Without this the rate reads 0% while the signal counts above it are
    non-zero, because those signals live in health_events, not in the
    episode's evidence blob.
    """

    def _ep(self, episode_id, event_id, nlos=False, quality=False):
        return {
            "episode_id": episode_id, "event_id": event_id,
            "resident_id": "resident_01",
            "start_time": datetime.now().isoformat(),
            "decision": {"level": "L1"}, "action": {"channel": "none"},
            "audit": {},
            "evidence": {"sensing_summary": {"nlos_flag": nlos,
                                             "quality_event": int(quality)}},
        }

    def test_event_linked_record_is_counted(self):
        records = [self._ep("ep1", "evt_q"), self._ep("ep2", "evt_normal")]
        events = [{"event_id": "evt_q", "event_type": "nlos_occlusion"}]
        q = build_report_context(records, events)["sensing_quality"]
        assert q["nlos_count"] == 1
        assert q["quality_flag_count"] == 1
        assert q["quality_event_rate_pct"] == 50.0

    def test_evidence_flagged_record_is_counted(self):
        records = [self._ep("ep1", "evt_a", nlos=True)]
        q = build_report_context(records, [])["sensing_quality"]
        assert q["quality_flag_count"] == 1
        assert q["quality_event_rate_pct"] == 100.0

    def test_rate_matches_signal_counts_when_signals_are_linked(self):
        records = [self._ep(f"ep{i}", f"evt{i}") for i in range(4)]
        events = [{"event_id": "evt0", "event_type": "nlos_occlusion"},
                  {"event_id": "evt1", "event_type": "low_confidence"},
                  {"event_id": "evt2", "event_type": "modality_conflict"}]
        q = build_report_context(records, events)["sensing_quality"]
        assert q["quality_signal_count"] == 3
        assert q["quality_flag_count"] == 3
        assert q["quality_event_rate_pct"] == 75.0
