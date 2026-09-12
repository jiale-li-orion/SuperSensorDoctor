"""Report Agent — weekly care report (LLM prose) + deterministic fallback.

Two rendering paths share ONE evidence context so they can never disagree:

  primary   ``ReportAgent.generate_weekly_report``  -> LLM prose (English)
  fallback  ``render_fallback_report``              -> deterministic markdown

Both emit the same document blocks, listed in ``REPORT_BLOCKS``. The LLM path
is used when a provider is supplied and answers in time; the deterministic
renderer covers timeouts, provider failures, and offline operation.

The report is **care support**, not diagnosis. Every statement must be
traceable to the episode evidence chain, and published validation numbers are
reported results, never resident-specific evidence.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Optional

from agent_layer.state_objects import EpisodeLog
from agent_layer.llm_provider import LLMProvider, ChatMessage
from agent_layer.validation_results import validation_block
from agent_layer.report_data import (
    DEFAULT_WINDOW_DAYS,
    load_report_window,
)


# ── Display vocabulary ─────────────────────────────────────────────────────
# Every label is a ``(zh, en)`` pair. ``build_report_context`` resolves the
# pairs for one language, so both renderers stay language-consistent.

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("zh", "en")


def _pick(pair: tuple, lang: str) -> str:
    """Resolve a ``(zh, en)`` label pair for ``lang``."""
    return pair[0] if lang == "zh" else pair[1]


# L0-L4 policy card, matching the paper's tier->channel mapping.
TIER_LABELS = {
    "L0": ("常规观察", "Routine observation"),
    "L1": ("持续观察", "Continued observation"),
    "L2": ("居民提醒", "Resident prompt"),
    "L3": ("家属告警", "Family notification"),
    "L4": ("紧急路由", "Emergency routing"),
}

TIER_CHANNELS = {
    "L0": "none",
    "L1": "none",
    "L2": "screen",
    "L3": "family_push",
    "L4": "emergency",
}

CHANNEL_LABELS = {
    "none": ("仅入库", "Record only"),
    "screen": ("居民屏幕", "Resident screen"),
    "family_push": ("家属推送", "Family notification"),
    "emergency": ("紧急路由（需人工确认）", "Emergency routing (human confirmation)"),
}

SOURCE_LABELS = {
    "RCP_NEWS2_2017_REFERENCE": (
        "绝对生理参考区间（RCP NEWS2 2017）",
        "Absolute physiological reference band (RCP NEWS2 2017)",
    ),
    "NICE_NG249_2025": (
        "跌倒评估证据需求（NICE NG249 2025）",
        "Fall assessment evidence needs (NICE NG249 2025)",
    ),
    "RESIDENT_HISTORY": ("个人纵向基线", "Personal longitudinal baseline"),
    "SENSOR_FUSION": ("跨模态测量可靠性", "Cross-modal measurement reliability"),
    "ACTIVITY_CONTEXT": ("活动与姿势上下文", "Activity and posture context"),
    "PROJECT_POLICY": ("项目分级策略", "Project triage policy"),
}

SENSING_QUALITY_LABELS = {
    "reliable": ("可靠", "Reliable"),
    "degraded": ("降级", "Degraded"),
    "unreliable": ("不可靠", "Unreliable"),
    "unknown": ("未知", "Unknown"),
}

EVENT_LABELS = {
    "hr_abnormal": ("心率异常", "Heart-rate deviation"),
    "rr_bradypnea": ("呼吸过缓", "Bradypnea"),
    "rr_tachypnea": ("呼吸急促", "Tachypnea"),
    "rr_baseline_deviation": ("呼吸基线偏离", "Respiratory baseline deviation"),
    "temp_abnormal": ("体温异常", "Surface-temperature deviation"),
    "fall_detected": ("跌倒", "Fall detected"),
    "fall_no_physiological_change": ("跌倒（无生理变化）", "Fall without physiological change"),
    "low_confidence": ("双射频分支低置信度", "Low confidence in both radio branches"),
    "nlos_occlusion": ("NLOS 遮挡", "NLOS degradation"),
    "modality_conflict": ("模态冲突", "Modality conflict"),
}

# Canonical document blocks, in render order. Keys match the context dict
# returned by ``build_report_context`` so both renderers stay in lockstep.
REPORT_BLOCKS = [
    "summary",
    "evidence_trail",
    "sensing_quality",
    "action_routing",
    "uncertainty",
    "vital_trends",
    "personal_baseline",
    "validation",
    "indicator_coverage",
    "privacy_boundary",
]

PRIVACY_BOUNDARY = {
    "title": ("隐私与人工监督", "Privacy and Human Oversight"),
    "statements": [
        ("原始与派生感知记录保留在原型主机本地。",
         "Raw and derived sensing records remain local to the prototype host."),
        ("选用外部 OpenAI 兼容模型时，只有最小化的结构化事件上下文离开设备。",
         "When an external OpenAI-compatible model is selected, only a minimized "
         "structured event context leaves the device."),
        ("本报告属于看护支持，不构成医学诊断、治疗方案或用药建议。",
         "This report is care support. It is not a medical diagnosis, treatment "
         "plan, or medication advice."),
        ("高风险路由需要经过授权的人工确认。",
         "High-risk routing requires authorized human confirmation."),
    ],
}


REPORT_PROMPT = """
You are the weekly care-report writer for SuperSenseDoctor, a contactless
home health-tracking agent. You turn a week of triage records into a
longitudinal care-support report for a family member or a reviewing clinician.

[HARD CONSTRAINTS]
- Use ONLY the evidence in the JSON payload below. Never invent thresholds,
  sources, measurements, or events.
- Never diagnose, never prescribe, never suggest medication. This is care
  support and risk triage only.
- Distinguish "is the measurement trustworthy" from "is the physiology
  abnormal". Low sensor confidence is not physiological deterioration.
- Reported validation numbers in `validation` are published results from the
  paper. Never present them as this resident's data.
- If a block has no data, write exactly "No data available for this period."
  under that heading. Do not omit the heading.

[EVIDENCE AND SOURCE LABELS]
Every `clinical_basis` entry carries a `source` tag. Use these meanings:
  RCP_NEWS2_2017_REFERENCE - absolute physiological reference band. The system
    does NOT compute a complete NEWS2 score.
  NICE_NG249_2025 - fall assessment evidence needs.
  RESIDENT_HISTORY - this resident's own longitudinal baseline.
  SENSOR_FUSION - cross-modal measurement reliability.
  ACTIVITY_CONTEXT - activity and posture context.
  PROJECT_POLICY - project-specific z-score threshold and recheck policy.

[TIER POLICY CARD]
  L0 Routine observation   - record only
  L1 Continued observation - record only, scheduled recheck
  L2 Resident prompt       - resident screen
  L3 Family notification   - family push
  L4 Emergency routing     - emergency channel, requires human confirmation

[OUTPUT]
Write English markdown with exactly these `##` headings, in this order:

## Weekly Summary
Total triage records, tier distribution L0-L4, highest tier reached, most
frequent tier, and whether any record required escalation.

## Evidence Trail
For the representative record, state the trace explicitly:
Nurse event -> evidence anchors -> triage tier -> delivery channel ->
persisted EpisodeLog. Include the reflex-path flag and the tools used.

## Sensing Quality and Fusion Arbitration
NLOS degradation, low-confidence, and modality-conflict counts; quality-event
rate; which branch dominated; and mean per-branch confidence. State clearly
that quality events affect measurement reliability, not health status.

## Action Routing
Delivery-channel distribution and event-type distribution.

## Uncertainty and Reflex Path
How many records declared missing evidence or requested a recheck, the
missing-evidence categories, and the reflex vs. deliberative split.

## Vital Trends
Latest, mean, and range for heart rate, respiratory rate, and surface
temperature, from the evidence chains.

## Personal Baseline
Per-metric z-score relative to this resident's own baseline, or N/A.

## Published Validation Results
Restate the reported sensing metrics and the Agent checklist result, clearly
labelled as published results from the paper.

## Indicator and Screening Coverage
Which indicator families the system covers, and for which condition families
they act as screening directions.

## Privacy and Human Oversight
Local-only raw data, minimized external context, care-support boundary, and
human confirmation for high-risk routing.
"""


REPORT_PROMPT_ZH = """
你是 SuperSenseDoctor 的周报撰写助手。SuperSenseDoctor 是一套非接触式居家健康追踪 Agent。
你的任务是把一周的分诊记录写成一份面向家属或复核医生的纵向看护支持报告。

【硬性约束】
- 只能使用下方 JSON 载荷中明确给出的证据。绝不虚构阈值、来源、测量值或事件。
- 绝不做诊断，绝不开具处方，绝不建议用药。本报告仅用于看护支持与风险分级。
- 区分"测量是否可信"与"生理是否异常"：传感器低置信度不等于生理恶化。
- `validation` 中的验证数字是论文已发表结果，绝不可当作该居民的数据呈现。
- 某个区块没有数据时，在该标题下只写"本周期无数据。"，不要省略标题。

【证据来源标签】
每条 `clinical_basis` 都带 `source` 标签，含义如下：
  RCP_NEWS2_2017_REFERENCE — 绝对生理参考区间。系统不计算完整 NEWS2 评分。
  NICE_NG249_2025 — 跌倒评估证据需求。
  RESIDENT_HISTORY — 该居民自身的纵向基线。
  SENSOR_FUSION — 跨模态测量可靠性。
  ACTIVITY_CONTEXT — 活动与姿势上下文。
  PROJECT_POLICY — 项目特有的 z-score 阈值与复查策略。

【分级策略卡】
  L0 常规观察   — 仅入库
  L1 持续观察   — 仅入库，安排复查
  L2 居民提醒   — 居民屏幕
  L3 家属告警   — 推送家属
  L4 紧急路由   — 紧急渠道，需授权人工确认

【输出】
用中文 markdown 输出，且必须严格包含以下 `##` 标题，顺序不变：

## 本周摘要
分诊记录总数、L0-L4 分级分布、达到的最高分级、最高频分级，以及是否有记录需要升级处置。

## 证据链
针对代表性记录，明确写出这条链路：
护理事件 → 证据依据 → 分诊分级 → 投递渠道 → 落库 EpisodeLog。
并包含反射弧标志与调用过的工具。

## 传感质量与融合仲裁
NLOS 遮挡、低置信度、模态冲突次数；质量标记占比；主导分支；各分支平均置信度。
必须明确说明：质量事件影响的是测量可靠性，不是健康状态。

## 行动路由
投递渠道分布与事件类型分布。

## 不确定性与反射弧
有多少记录声明了证据缺口或请求复查、缺口类别，以及反射弧与推演式决策的比例。

## 体征趋势
心率、呼吸率与体表温度的最新值、均值与范围，均取自证据链。

## 个人基线
各指标相对该居民自身基线的 z-score，无数据则写 N/A。

## 论文已发表验证结果
复述已报告的传感指标与 Agent 检查项结果，并明确标注为论文已发表结果。

## 指标与筛查覆盖
系统覆盖的指标族，以及它们对应哪些疾病族作为筛查方向。

## 隐私与人工监督
原始数据本地留存、外部上下文最小化、看护支持边界，以及高风险路由需人工确认。
"""


def report_prompt(lang: str = DEFAULT_LANG) -> str:
    """Return the report system prompt for ``lang``."""
    return REPORT_PROMPT_ZH if lang == "zh" else REPORT_PROMPT


# ── Accessors (episodes may be dataclasses or plain dicts) ─────────────────

def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    try:
        dt = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _tier(decision: dict) -> str:
    """Normalize a decision level to a bare L0-L4 token."""
    raw = str(decision.get("level", "L0") or "L0")
    tier = raw[:2].upper()
    return tier if tier in TIER_LABELS else "L0"


def _event_label(event_type: str, lang: str = DEFAULT_LANG) -> str:
    pair = EVENT_LABELS.get(event_type)
    if pair is None:
        return event_type or ""
    return _pick(pair, lang)


def _privacy_boundary(lang: str) -> dict:
    """Resolve the privacy/oversight block for one language."""
    return {
        "title": _pick(PRIVACY_BOUNDARY["title"], lang),
        "statements": [_pick(s, lang) for s in PRIVACY_BOUNDARY["statements"]],
    }


# ── Context builder ───────────────────────────────────────────────────────

def build_report_context(
    episodes: list,
    events: Optional[list] = None,
    reference_ts: Optional[datetime] = None,
    window_days: int = 7,
    lang: str = DEFAULT_LANG,
) -> dict:
    """Build the canonical, deterministic evidence context for a weekly report.

    This is the single source of truth consumed by both the LLM renderer and
    the deterministic fallback renderer, so the two paths cannot disagree on
    counts, tiers, or traces.

    Args:
        episodes: EpisodeLog objects or episode dicts (decision/evidence/
            action/audit keys, JSON strings accepted).
        events: optional HealthEvent objects or event dicts for the period.
        reference_ts: report anchor time; defaults to now.
        window_days: lookback window in days.
        lang: ``"en"`` or ``"zh"``. Only display labels are localized; every
            code, count, and identifier stays language-neutral.

    Returns:
        A dict keyed by ``REPORT_BLOCKS``.
    """
    lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
    ref = reference_ts or datetime.now()
    if ref.tzinfo is not None:
        ref = ref.replace(tzinfo=None)
    cutoff = ref - timedelta(days=window_days)
    events = events or []

    # ── Period filter ──
    recent = []
    for ep in episodes:
        ts = _parse_ts(_field(ep, "start_time"))
        if ts is not None and ts >= cutoff:
            recent.append((ts, ep))
    recent.sort(key=lambda pair: pair[0])

    period = {
        "start": cutoff.isoformat(),
        "end": ref.isoformat(),
        "start_date": cutoff.strftime("%Y-%m-%d"),
        "end_date": ref.strftime("%Y-%m-%d"),
        "reference": ref.strftime("%Y-%m-%d %H:%M"),
        "window_days": window_days,
    }

    if not recent:
        return {
            "period": period,
            "empty": True,
            "lang": lang,
            "summary": _empty_summary(),
            "evidence_trail": None,
            "sensing_quality": _empty_sensing_quality(events, lang),
            "action_routing": _empty_action_routing(),
            "uncertainty": _empty_uncertainty(),
            "vital_trends": _empty_vital_trends(lang),
            "personal_baseline": [],
            "validation": validation_block(),
            "indicator_coverage": validation_block(),
            "privacy_boundary": _privacy_boundary(lang),
        }

    # ── Per-episode normalization ──
    records = []
    for ts, ep in recent:
        decision = _as_dict(_field(ep, "decision", {}))
        action = _as_dict(_field(ep, "action", {}))
        audit = _as_dict(_field(ep, "audit", {}))
        evidence = _as_dict(_field(ep, "evidence", {}))
        records.append({
            "episode_id": _field(ep, "episode_id", ""),
            "event_id": _field(ep, "event_id", ""),
            "timestamp": ts,
            "tier": _tier(decision),
            "decision": decision,
            "action": action,
            "audit": audit,
            "evidence": evidence,
        })

    # ── Summary ──
    level_counts = {tier: 0 for tier in TIER_LABELS}
    for rec in records:
        level_counts[rec["tier"]] += 1
    severity_order = ["L0", "L1", "L2", "L3", "L4"]
    highest_tier = max(
        (t for t in severity_order if level_counts[t] > 0), default="L0"
    )
    most_frequent = max(level_counts, key=lambda t: (level_counts[t], t))

    reflex_count = sum(1 for rec in records if rec["audit"].get("reflex"))
    escalated = sum(1 for rec in records if rec["tier"] in ("L3", "L4"))

    summary = {
        "total_records": len(records),
        "tier_counts": level_counts,
        "highest_tier": highest_tier,
        "highest_tier_label": _pick(TIER_LABELS[highest_tier], lang),
        "most_frequent_tier": most_frequent,
        "most_frequent_tier_label": _pick(TIER_LABELS[most_frequent], lang),
        "escalated_count": escalated,
        "reflex_count": reflex_count,
        "deliberative_count": len(records) - reflex_count,
        "tier_labels": {t: _pick(p, lang) for t, p in TIER_LABELS.items()},
    }

    # ── Evidence trail (representative record: highest tier, then latest) ──
    representative = max(
        records,
        key=lambda r: (severity_order.index(r["tier"]), r["timestamp"]),
    )
    evidence_trail = _build_evidence_trail(representative, lang)

    # ── Sensing quality and fusion arbitration ──
    sensing_quality = _build_sensing_quality(records, events, lang)

    # ── Action routing ──
    action_routing = _build_action_routing(records, events, lang)

    # ── Uncertainty and reflex path ──
    uncertainty = _build_uncertainty(records, lang)

    # ── Vital trends and personal baseline from evidence chains ──
    vital_trends = _build_vital_trends(records, lang)
    personal_baseline = _build_personal_baseline(records, lang)

    validation = validation_block()

    return {
        "period": period,
        "empty": False,
        "lang": lang,
        "summary": summary,
        "evidence_trail": evidence_trail,
        "sensing_quality": sensing_quality,
        "action_routing": action_routing,
        "uncertainty": uncertainty,
        "vital_trends": vital_trends,
        "personal_baseline": personal_baseline,
        "validation": validation,
        "indicator_coverage": validation,
        "privacy_boundary": _privacy_boundary(lang),
    }


def _empty_summary() -> dict:
    return {
        "total_records": 0,
        "tier_counts": {tier: 0 for tier in TIER_LABELS},
        "highest_tier": None,
        "highest_tier_label": "",
        "most_frequent_tier": None,
        "most_frequent_tier_label": "",
        "escalated_count": 0,
        "reflex_count": 0,
        "deliberative_count": 0,
    }


def _build_evidence_trail(rec: dict, lang: str = DEFAULT_LANG) -> dict:
    """Trigger -> evidence anchors -> tier -> channel -> EpisodeLog."""
    decision = rec["decision"]
    evidence = rec["evidence"]
    audit = rec["audit"]
    event = _as_dict(evidence.get("event", {}))

    anchors = []
    for item in _as_list(decision.get("clinical_basis", [])):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        anchors.append({
            "type": item.get("type", ""),
            "finding": item.get("finding", ""),
            "source": source,
            "source_label": _pick(SOURCE_LABELS[source], lang) if source in SOURCE_LABELS
            else (source or _pick(("未标注", "unspecified"), lang)),
        })

    tools_called = [str(t) for t in _as_list(audit.get("tools_called", []))]
    channel = str(rec["action"].get("channel", "none") or "none")

    # Decision path. `reflex` is the authoritative flag. Older records predate
    # it, so when it is absent we fall back to a conservative inference: zero
    # reasoning steps plus deterministic-only tools cannot have come from the
    # deliberative ReAct loop.
    #
    # `reflex_inferred` marks only the cases where that inference OVERRODE the
    # default reading (absent flag => deliberative). When inference agrees with
    # the default there is nothing surprising to disclose, and flagging every
    # legacy row would just add noise.
    reflex_raw = audit.get("reflex")
    step_count = audit.get("step_count")
    if reflex_raw is None:
        deterministic_only = bool(tools_called) and all(
            t.startswith("nurse") or t in ("issue_action", "write_episode")
            for t in tools_called
        )
        reflex = bool(step_count == 0 and deterministic_only)
        reflex_inferred = reflex
    else:
        reflex = bool(reflex_raw)
        reflex_inferred = False

    return {
        "episode_id": rec["episode_id"],
        "timestamp": rec["timestamp"].isoformat(),
        "trigger": {
            "event_id": event.get("event_id", rec["event_id"]),
            "event_type": event.get("event_type", ""),
            "event_label": _event_label(event.get("event_type", ""), lang),
            "reason": event.get("trigger_reason", ""),
        },
        "interpretation": decision.get("event_interpretation", ""),
        "anchors": anchors,
        "tier": rec["tier"],
        "tier_label": _pick(TIER_LABELS[rec["tier"]], lang),
        "channel": channel,
        "channel_label": _pick(CHANNEL_LABELS[channel], lang) if channel in CHANNEL_LABELS else channel,
        "reflex": reflex,
        "reflex_inferred": reflex_inferred,
        "step_count": step_count,
        "tools_called": tools_called,
        "evidence_used": _as_list(decision.get("evidence_used", [])),
        "safety_boundary": decision.get("safety_boundary", "care_support_only"),
    }


def _build_sensing_quality(records: list, events: list, lang: str = DEFAULT_LANG) -> dict:
    """Quality events and cross-modal arbitration, from events + evidence.

    Two different quantities are reported, and they must not be conflated:

    * ``quality_flag_count`` / ``quality_event_rate_pct`` — how many *records*
      carry at least one quality signal, and that share of all records. This is
      the rate, so it is bounded to 0-100%.
    * ``quality_signal_count`` — the total number of distinct quality signals,
      which may legitimately exceed the record count when a record carries
      several.

    Signal identity is keyed on ``event_id`` so that one NLOS occurrence seen
    both as a HealthEvent and inside an episode's evidence chain is counted
    once, not twice.
    """
    nlos_keys: set = set()
    low_conf_keys: set = set()
    conflict_keys: set = set()
    flagged_records: set = set()
    wifi_confs, mmwave_confs = [], []
    dominant_counts: dict = {}

    for idx, ev in enumerate(events):
        et = _field(ev, "event_type", "") or ""
        markers = _as_dict(_field(ev, "rule_markers", {}))
        key = _field(ev, "event_id") or f"evt#{idx}"

        if et == "nlos_occlusion" or markers.get("nlos_flag"):
            nlos_keys.add(key)
        if et == "low_confidence":
            low_conf_keys.add(key)
        if et == "modality_conflict":
            conflict_keys.add(key)

        for metric in ("hr", "rr"):
            dominant = markers.get(f"{metric}_dominant")
            if dominant:
                bucket = f"{metric}:{dominant}"
                dominant_counts[bucket] = dominant_counts.get(bucket, 0) + 1

        wc = _field(ev, "wifi_confidence")
        mc = _field(ev, "mmwave_confidence")
        if wc is not None:
            wifi_confs.append(float(wc))
        if mc is not None:
            mmwave_confs.append(float(mc))

    for rec in records:
        summary = _as_dict(rec["evidence"].get("sensing_summary", {}))
        # Prefer the triggering event id so an event and its episode evidence
        # resolve to the same key; fall back to the episode id.
        key = rec.get("event_id") or rec["episode_id"]

        if summary.get("quality_event") or summary.get("nlos_flag"):
            flagged_records.add(rec["episode_id"])
        if summary.get("nlos_flag"):
            nlos_keys.add(key)

        wc, mc = summary.get("wifi_confidence"), summary.get("mmwave_confidence")
        if wc is not None:
            wifi_confs.append(float(wc))
        if mc is not None:
            mmwave_confs.append(float(mc))
        for metric, field in (("hr", "hr_source"), ("rr", "rr_source")):
            src = summary.get(field)
            if src:
                bucket = f"{metric}:{src}"
                dominant_counts[bucket] = dominant_counts.get(bucket, 0) + 1

    # A record is "quality flagged" when it carries a quality signal either in
    # its own evidence OR because its triggering event IS a quality event.
    # Without the second condition the rate reads as 0% while the counts above
    # it show non-zero NLOS / low-confidence / conflict events, because those
    # signals live in health_events rather than in the episode evidence.
    quality_event_keys = nlos_keys | low_conf_keys | conflict_keys
    for rec in records:
        key = rec.get("event_id") or rec["episode_id"]
        if key in quality_event_keys:
            flagged_records.add(rec["episode_id"])

    nlos = len(nlos_keys)
    low_conf = len(low_conf_keys)
    conflict = len(conflict_keys)
    quality_signal_count = nlos + low_conf + conflict
    quality_flag_count = len(flagged_records)

    total_records = len(records)
    if total_records:
        # Bounded by construction: the numerator is a subset of the records.
        quality_rate = round(quality_flag_count / total_records * 100, 1)
        quality_rate = min(quality_rate, 100.0)
    else:
        quality_rate = 0.0

    dominant = [
        {"metric": k.split(":", 1)[0], "branch": k.split(":", 1)[1], "count": v}
        for k, v in sorted(dominant_counts.items(), key=lambda kv: -kv[1])
    ]

    return {
        "nlos_count": nlos,
        "low_confidence_count": low_conf,
        "modality_conflict_count": conflict,
        "quality_flag_count": quality_flag_count,
        "quality_signal_count": quality_signal_count,
        "total_records": total_records,
        "quality_event_rate_pct": quality_rate,
        "mean_wifi_confidence_pct": round(sum(wifi_confs) / len(wifi_confs) * 100) if wifi_confs else None,
        "mean_mmwave_confidence_pct": round(sum(mmwave_confs) / len(mmwave_confs) * 100) if mmwave_confs else None,
        "dominant_branches": dominant,
        "note": _pick(
            ("质量事件影响的是测量可靠性，不是健康状态。测量降级不等于生理恶化。",
             "Quality events affect measurement reliability, not health status. "
             "A degraded measurement is not physiological deterioration."),
            lang,
        ),
    }


def _empty_sensing_quality(events: list, lang: str = DEFAULT_LANG) -> dict:
    return _build_sensing_quality([], events, lang)


def _build_action_routing(records: list, events: list, lang: str = DEFAULT_LANG) -> dict:
    channel_counts: dict = {}
    for rec in records:
        ch = str(rec["action"].get("channel", "none") or "none")
        channel_counts[ch] = channel_counts.get(ch, 0) + 1

    type_counts: dict = {}
    for ev in events:
        et = _field(ev, "event_type", "") or "unknown"
        type_counts[et] = type_counts.get(et, 0) + 1
    if not type_counts:
        for rec in records:
            event = _as_dict(rec["evidence"].get("event", {}))
            et = event.get("event_type") or "unknown"
            type_counts[et] = type_counts.get(et, 0) + 1

    return {
        "channels": [
            {"channel": ch,
             "label": _pick(CHANNEL_LABELS[ch], lang) if ch in CHANNEL_LABELS else ch,
             "count": n}
            for ch, n in sorted(channel_counts.items(), key=lambda kv: -kv[1])
        ],
        "event_types": [
            {"type": et, "label": _event_label(et, lang), "count": n}
            for et, n in sorted(type_counts.items(), key=lambda kv: -kv[1])
        ],
    }


def _empty_action_routing() -> dict:
    return {"channels": [], "event_types": []}


def _build_uncertainty(records: list, lang: str = DEFAULT_LANG) -> dict:
    needs_recheck = 0
    missing_evidence: dict = {}
    quality_levels: dict = {}

    for rec in records:
        unc = _as_dict(rec["decision"].get("uncertainty", {}))
        if unc.get("needs_recheck"):
            needs_recheck += 1
        sq = str(unc.get("sensing_quality", "unknown") or "unknown")
        quality_levels[sq] = quality_levels.get(sq, 0) + 1
        for item in _as_list(unc.get("missing_evidence", [])):
            key = str(item)
            missing_evidence[key] = missing_evidence.get(key, 0) + 1

    total = len(records) or 1
    return {
        "total_records": len(records),
        "needs_recheck_count": needs_recheck,
        "needs_recheck_pct": round(needs_recheck / total * 100, 1) if records else 0.0,
        "missing_evidence": [
            {"item": k, "count": v}
            for k, v in sorted(missing_evidence.items(), key=lambda kv: -kv[1])
        ],
        "sensing_quality_levels": [
            {"level": k,
             "label": _pick(SENSING_QUALITY_LABELS[k], lang) if k in SENSING_QUALITY_LABELS else k,
             "count": v}
            for k, v in sorted(quality_levels.items(), key=lambda kv: -kv[1])
        ],
    }


def _empty_uncertainty() -> dict:
    return {
        "total_records": 0,
        "needs_recheck_count": 0,
        "needs_recheck_pct": 0.0,
        "missing_evidence": [],
        "sensing_quality_levels": [],
    }


def _build_vital_trends(records: list, lang: str = DEFAULT_LANG) -> dict:
    """Latest/mean/range per metric, read from each evidence chain."""
    buckets = {"heart_rate": [], "respiration_rate": [], "body_temp": []}
    labels = {
        "heart_rate": (("心率", "Heart rate"), "bpm"),
        "respiration_rate": (("呼吸率", "Respiratory rate"), "/min"),
        "body_temp": (("体表温度", "Surface temperature"), "°C"),
    }

    for rec in records:
        summary = _as_dict(rec["evidence"].get("sensing_summary", {}))
        for key in buckets:
            value = summary.get(key)
            if value is None:
                continue
            try:
                buckets[key].append((rec["timestamp"], float(value)))
            except (TypeError, ValueError):
                continue

    metrics = []
    for key, samples in buckets.items():
        samples.sort(key=lambda pair: pair[0])
        name_pair, unit = labels[key]
        name = _pick(name_pair, lang)
        if not samples:
            metrics.append({"key": key, "name": name, "unit": unit, "latest": None,
                            "mean": None, "min": None, "max": None, "count": 0})
            continue
        values = [v for _, v in samples]
        metrics.append({
            "key": key,
            "name": name,
            "unit": unit,
            "latest": round(samples[-1][1], 1),
            "mean": round(sum(values) / len(values), 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
            "count": len(values),
        })
    return {"metrics": metrics}


def _empty_vital_trends(lang: str = DEFAULT_LANG) -> dict:
    return _build_vital_trends([], lang)


def _build_personal_baseline(records: list, lang: str = DEFAULT_LANG) -> list:
    """Latest per-metric z-score relative to the resident's own baseline."""
    latest: dict = {}
    for rec in records:
        context = _as_dict(rec["evidence"].get("context", {}))
        baseline = _as_dict(context.get("personal_baseline", {}))
        if not baseline:
            continue
        for metric in ("heart_rate", "respiration_rate", "body_temp"):
            entry = baseline.get(metric) if isinstance(baseline.get(metric), dict) else None
            if entry:
                latest[metric] = entry

    labels = {
        "heart_rate": (("心率", "Heart rate"), "bpm"),
        "respiration_rate": (("呼吸率", "Respiratory rate"), "/min"),
        "body_temp": (("体表温度", "Surface temperature"), "°C"),
    }
    out = []
    for key, (name_pair, unit) in labels.items():
        name = _pick(name_pair, lang)
        entry = latest.get(key)
        if entry:
            out.append({
                "key": key,
                "name": name,
                "unit": unit,
                "value": entry.get("value"),
                "mean": entry.get("mean"),
                "std": entry.get("std"),
                "z_score": entry.get("z_score"),
            })
        else:
            out.append({"key": key, "name": name, "unit": unit,
                        "value": None, "mean": None, "std": None, "z_score": None})
    return out


# ── Deterministic fallback renderer ───────────────────────────────────────

def render_fallback_report(context: dict, lang: Optional[str] = None) -> str:
    """Render the canonical context as markdown without an LLM.

    Emits the same ``##`` blocks as the LLM contract so downstream rendering
    and display never depend on which path produced the report.

    ``lang`` defaults to the language the context was built with. ``S(zh, en)``
    is the local phrase picker for this renderer's own prose.
    """
    lang = lang or context.get("lang") or DEFAULT_LANG
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG

    def S(zh: str, en: str) -> str:
        return zh if lang == "zh" else en

    lines: list[str] = []
    period = context["period"]
    summary = context["summary"]
    tier_labels = summary.get("tier_labels") or {
        t: _pick(p, lang) for t, p in TIER_LABELS.items()
    }

    # ── Weekly Summary ──
    lines.append("## " + S("本周摘要", "Weekly Summary"))
    if context.get("empty") or summary["total_records"] == 0:
        lines.append("")
        lines.append(S(
            f"报告周期（{period['start_date']} 至 {period['end_date']}）内没有分诊记录。",
            f"No triage records in the reporting period "
            f"({period['start_date']} to {period['end_date']}).",
        ))
        _append_static_blocks(lines, context, lang)
        return "\n".join(lines)

    lines.append("")
    lines.append(S(
        f"报告周期 **{period['start_date']} 至 {period['end_date']}**"
        f"（{period['window_days']} 天窗口）。",
        f"Period **{period['start_date']} to {period['end_date']}** "
        f"({period['window_days']}-day window).",
    ))
    lines.append("")
    lines.append("- " + S("分诊记录总数", "Total triage records")
                 + f": **{summary['total_records']}**")
    for tier in ["L0", "L1", "L2", "L3", "L4"]:
        count = summary["tier_counts"].get(tier, 0)
        if count:
            lines.append(f"  - {tier} {tier_labels[tier]}: {count}")
    lines.append("- " + S("达到的最高分级", "Highest tier reached")
                 + f": **{summary['highest_tier']} "
                   f"{summary['highest_tier_label']}**")
    lines.append("- " + S("最高频分级", "Most frequent tier")
                 + f": **{summary['most_frequent_tier']} "
                   f"{summary['most_frequent_tier_label']}**")
    lines.append("- " + S("需要升级处置的记录（L3/L4）", "Records requiring escalation (L3/L4)")
                 + f": **{summary['escalated_count']}**")

    # ── Evidence Trail ──
    lines.append("")
    lines.append("## " + S("证据链", "Evidence Trail"))
    trail = context["evidence_trail"]
    if not trail:
        lines.append("")
        lines.append(S("本周期无数据。", "No data available for this period."))
    else:
        lines.append("")
        lines.append(S(
            f"代表性记录 `{trail['episode_id']}`，时间 {trail['timestamp']}：",
            f"Representative record `{trail['episode_id']}` at {trail['timestamp']}:",
        ))
        lines.append("")
        lines.append(
            "- **" + S("护理事件", "Nurse event") + "**: "
            f"{trail['trigger']['event_label']} "
            f"(`{trail['trigger']['event_type']}`, id `{trail['trigger']['event_id']}`)"
        )
        if trail["trigger"]["reason"]:
            lines.append(f"  - {S('触发原因', 'Trigger')}: {trail['trigger']['reason']}")
        if trail["interpretation"]:
            lines.append(f"- **{S('解释', 'Interpretation')}**: {trail['interpretation']}")
        if trail["anchors"]:
            lines.append(f"- **{S('证据依据', 'Evidence anchors')}**:")
            for anchor in trail["anchors"]:
                finding = anchor["finding"] or anchor["type"] \
                    or S("未具体说明", "unspecified")
                lines.append(f"  - {finding} — _{anchor['source_label']}_")
        else:
            lines.append(f"- **{S('证据依据', 'Evidence anchors')}**: "
                         + S("无记录", "none recorded"))
        lines.append(f"- **{S('分诊分级', 'Triage tier')}**: "
                     f"{trail['tier']} {trail['tier_label']}")
        lines.append(f"- **{S('投递渠道', 'Delivery channel')}**: "
                     f"{trail['channel_label']} (`{trail['channel']}`)")
        lines.append(f"- **{S('落库 EpisodeLog', 'Persisted EpisodeLog')}**: "
                     f"`{trail['episode_id']}`")
        path = S("反射弧（确定性，不经过 LLM 推演）",
                 "reflex path (deterministic, no LLM deliberation)") \
            if trail["reflex"] else \
            S(f"推演式 ReAct 循环（{trail['step_count']} 步）",
              f"deliberative ReAct loop "
              f"({trail['step_count']} "
              f"{'step' if trail['step_count'] == 1 else 'steps'})")
        if trail.get("reflex_inferred"):
            path += S("（依据审计字段推断）", " (inferred from the audit fields)")
        lines.append(f"- **{S('决策路径', 'Decision path')}**: {path}")
        if trail["tools_called"]:
            lines.append("- **" + S("调用工具", "Tools used") + "**: "
                         + ", ".join(f"`{t}`" for t in trail["tools_called"]))
        lines.append(f"- **{S('安全边界', 'Safety boundary')}**: "
                     f"`{trail['safety_boundary']}`")

    # ── Sensing Quality and Fusion Arbitration ──
    lines.append("")
    lines.append("## " + S("传感质量与融合仲裁", "Sensing Quality and Fusion Arbitration"))
    quality = context["sensing_quality"]
    lines.append("")
    lines.append(f"- {S('NLOS 遮挡', 'NLOS degradation')}: **{quality['nlos_count']}**")
    lines.append(f"- {S('双射频分支低置信度', 'Low confidence in both radio branches')}: "
                 f"**{quality['low_confidence_count']}**")
    lines.append(f"- {S('模态冲突', 'Modality conflicts')}: "
                 f"**{quality['modality_conflict_count']}**")
    lines.append(f"- {S('质量信号总数', 'Total quality signals')}: "
                 f"**{quality['quality_signal_count']}**")
    lines.append(f"- {S('带质量标记的记录占比', 'Quality-flagged records')}: "
                 f"**{quality['quality_event_rate_pct']}%** "
                 f"({quality['quality_flag_count']} / "
                 f"{quality.get('total_records', summary['total_records'])} "
                 f"{S('条记录', 'records')})")
    if quality["mean_wifi_confidence_pct"] is not None:
        lines.append(f"- {S('WiFi BFI 平均置信度', 'Mean WiFi BFI confidence')}: "
                     f"**{quality['mean_wifi_confidence_pct']}%**")
    if quality["mean_mmwave_confidence_pct"] is not None:
        lines.append(f"- {S('mmWave 平均置信度', 'Mean mmWave confidence')}: "
                     f"**{quality['mean_mmwave_confidence_pct']}%**")
    if quality["dominant_branches"]:
        branches = ", ".join(
            f"{d['metric'].upper()} → {d['branch']} ({d['count']})"
            for d in quality["dominant_branches"]
        )
        lines.append(f"- {S('主导仲裁分支', 'Dominant arbitration branch')}: {branches}")
    lines.append("")
    lines.append(quality["note"])

    # ── Action Routing ──
    lines.append("")
    lines.append("## " + S("行动路由", "Action Routing"))
    routing = context["action_routing"]
    lines.append("")
    if routing["channels"]:
        for item in routing["channels"]:
            lines.append(f"- {item['label']} (`{item['channel']}`): **{item['count']}**")
    else:
        lines.append("- " + S("本周期无数据。", "No data available for this period."))
    if routing["event_types"]:
        lines.append("")
        lines.append(S("事件类型：", "Event types:"))
        lines.append("")
        for item in routing["event_types"]:
            lines.append(f"- {item['label']}: {item['count']}")

    # ── Uncertainty and Reflex Path ──
    lines.append("")
    lines.append("## " + S("不确定性与反射弧", "Uncertainty and Reflex Path"))
    unc = context["uncertainty"]
    lines.append("")
    lines.append(f"- {S('请求复查的记录', 'Records requesting a recheck')}: "
                 f"**{unc['needs_recheck_count']}** ({unc['needs_recheck_pct']}%)")
    if unc["missing_evidence"]:
        lines.append("- " + S("声明的证据缺口：", "Missing evidence declared:"))
        for item in unc["missing_evidence"]:
            lines.append(f"  - {item['item']}: {item['count']}")
    else:
        lines.append("- " + S("声明的证据缺口：无", "Missing evidence declared: none"))
    if unc["sensing_quality_levels"]:
        levels = ", ".join(
            f"{lv['label']} ({lv['count']})" for lv in unc["sensing_quality_levels"]
        )
        lines.append(f"- {S('自评传感质量', 'Self-assessed sensing quality')}: {levels}")
    lines.append(
        f"- {S('决策路径', 'Decision path')}: "
        f"{S('反射弧', 'reflex')} **{summary['reflex_count']}** / "
        f"{S('推演', 'deliberative')} **{summary['deliberative_count']}**"
    )

    # ── Vital Trends ──
    lines.append("")
    lines.append("## " + S("体征趋势", "Vital Trends"))
    lines.append("")
    any_vitals = False
    for metric in context["vital_trends"]["metrics"]:
        if metric["latest"] is None:
            lines.append(f"- {metric['name']}: " + S("无数据", "no data"))
            continue
        any_vitals = True
        lines.append(S(
            f"- {metric['name']}：最新 **{metric['latest']} {metric['unit']}**，"
            f"均值 {metric['mean']}，范围 {metric['min']}–{metric['max']}"
            f"（{metric['count']} 个采样）",
            f"- {metric['name']}: latest **{metric['latest']} {metric['unit']}**, "
            f"mean {metric['mean']}, range {metric['min']}-{metric['max']} "
            f"({metric['count']} samples)",
        ))
    if not any_vitals:
        lines.append("")
        lines.append(S("本周期无数据。", "No data available for this period."))

    # ── Personal Baseline ──
    lines.append("")
    lines.append("## " + S("个人基线", "Personal Baseline"))
    lines.append("")
    for item in context["personal_baseline"]:
        if item["z_score"] is None:
            lines.append(f"- {item['name']}: " + S("无数据", "N/A"))
        else:
            lines.append(S(
                f"- {item['name']}：z = **{item['z_score']:+.2f}**"
                f"（当前 {item['value']}，基线 {item['mean']} ± {item['std']}）",
                f"- {item['name']}: z = **{item['z_score']:+.2f}** "
                f"(value {item['value']}, baseline {item['mean']} ± {item['std']})",
            ))

    _append_static_blocks(lines, context, lang)
    return "\n".join(lines)


def _append_static_blocks(lines: list, context: dict, lang: str = DEFAULT_LANG) -> None:
    """Blocks that do not depend on the week's episodes."""
    def S(zh: str, en: str) -> str:
        return zh if lang == "zh" else en

    # ── Published Validation Results ──
    lines.append("")
    lines.append("## " + S("论文已发表验证结果", "Published Validation Results"))
    lines.append("")
    validation = context["validation"]
    paper = validation["paper"]
    lines.append(S(
        f"以下为 _《{paper['title']}》_（{paper['venue']}，"
        f"[{paper['doi']}]({paper['url']})）报告的实验结果。"
        "这些是论文基准，不是该居民的数据。",
        f"Reported results from _{paper['title']}_ ({paper['venue']}, "
        f"[{paper['doi']}]({paper['url']})). These are published benchmarks, "
        "not this resident's data.",
    ))
    lines.append("")
    indicator_columns_zh = validation.get("indicator_columns_zh", [])
    for row in validation["vital_signs"]:
        lines.append(S(
            f"- {row.get('task_zh', row['task'])}"
            f"（参考标准：{row.get('ground_truth_zh', row['ground_truth'])}）：",
            f"- {row['task']} (reference: {row['ground_truth']}):",
        ))
        for method in row["methods"]:
            mark = S(" **（最终融合值）**", " **(final fused estimate)**") \
                if method.get("is_final") else ""
            lines.append(
                f"  - {method['method']}: MAE {method['mae']} {row['unit']}, "
                f"RMSD {method['rmsd']} {row['unit']}{mark}"
            )
    fall = validation["fall_recognition"]
    lines.append(S(
        f"- {fall.get('task_zh', fall['task'])}"
        f"（{fall.get('ground_truth_zh', fall['ground_truth'])}）："
        f"**准确率 {fall['accuracy_pct']}%**",
        f"- {fall['task']} ({fall['ground_truth']}): **{fall['accuracy_pct']}% accuracy**",
    ))
    multi = validation["multi_interval"]
    lines.append(S(
        f"- 多区间验证：{multi['intervals']} 个区间，"
        f"{multi['state_rows']} 行一秒状态数据",
        f"- Multi-interval validation: {multi['intervals']} intervals, "
        f"{multi['state_rows']} one-second state rows",
    ))
    agent_eval = validation["agent_evaluation"]
    overall = agent_eval["overall"]
    lines.append(S(
        f"- Agent 检查项：**{overall['criteria_passed']}/{overall['criteria_total']} "
        f"（{overall['pass_rate_pct']}%）**",
        f"- Agent checklist: **{overall['criteria_passed']}/{overall['criteria_total']} "
        f"({overall['pass_rate_pct']}%)**",
    ))
    lines.append(S(
        f"  - 确定性筛查：事件匹配 {agent_eval['deterministic']['matched']}/"
        f"{agent_eval['deterministic']['cases']}；融合仲裁 "
        f"{agent_eval['deterministic']['fusion_matched']}/"
        f"{agent_eval['deterministic']['fusion_cases']}",
        f"  - Deterministic screening: {agent_eval['deterministic']['matched']}/"
        f"{agent_eval['deterministic']['cases']} event matches; "
        f"fusion arbitration {agent_eval['deterministic']['fusion_matched']}/"
        f"{agent_eval['deterministic']['fusion_cases']}",
    ))
    triage = agent_eval["triage"]
    lines.append(S(
        f"  - 隐藏答案分诊（{triage['model']}）：分级与渠道 "
        f"{triage['tier_and_channel_matched']}/{triage['scenarios']}；"
        f"证据链保留 {triage['trace_preserved']}/{triage['scenarios']}",
        f"  - Hidden-answer triage ({triage['model']}): tier and channel "
        f"{triage['tier_and_channel_matched']}/{triage['scenarios']}; "
        f"trace preserved {triage['trace_preserved']}/{triage['scenarios']}",
    ))
    lines.append(S(
        f"  - P95 延迟：确定性 {agent_eval['latency']['deterministic_p95_ms']} ms，"
        f"LLM 分诊 {agent_eval['latency']['llm_triage_p95_ms']} ms",
        f"  - P95 latency: deterministic "
        f"{agent_eval['latency']['deterministic_p95_ms']} ms, LLM triage "
        f"{agent_eval['latency']['llm_triage_p95_ms']} ms",
    ))

    # ── Indicator and Screening Coverage ──
    lines.append("")
    lines.append("## " + S("指标与筛查覆盖", "Indicator and Screening Coverage"))
    lines.append("")
    for row in validation["indicator_coverage"]:
        if lang == "zh":
            labels = indicator_columns_zh or validation["indicator_columns"]
            condition = row.get("condition_zh", row["condition"])
        else:
            labels = validation["indicator_columns"]
            condition = row["condition"]
        covered = [label for key, label in labels if row["coverage"].get(key)]
        lines.append("- **" + condition + "**: "
                     + (", ".join(covered) if covered else S("无", "none")))
    lines.append("")
    lines.append(S(
        validation.get("clinical_boundary_zh", validation["clinical_boundary"]),
        validation["clinical_boundary"],
    ))

    # ── Privacy and Human Oversight ──
    lines.append("")
    lines.append("## " + context["privacy_boundary"]["title"])
    lines.append("")
    for statement in context["privacy_boundary"]["statements"]:
        lines.append(f"- {statement}")



class ReportAgent:
    """Weekly care report: LLM prose with a deterministic markdown fallback."""

    def build_context(
        self,
        episodes: list,
        events: Optional[list] = None,
        reference_ts: Optional[datetime] = None,
        lang: str = DEFAULT_LANG,
    ) -> dict:
        """Public accessor for the canonical evidence context."""
        return build_report_context(episodes, events, reference_ts, lang=lang)

    def load_window(
        self,
        resident_id: str,
        reference_ts: Optional[datetime] = None,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ):
        """Load the report window straight from storage.

        Prefer this over handing the agent a pre-sliced episode list: the
        window is then selected by time range, so a busy week is never
        truncated by a row cap. See ``agent_layer.report_data``.
        """
        return load_report_window(resident_id, reference_ts, window_days)

    async def generate_weekly_report_for(
        self,
        resident_id: str,
        events: Optional[list] = None,
        llm_provider: Optional[LLMProvider] = None,
        reference_ts: Optional[datetime] = None,
        timeout: float = 10.0,
        lang: str = DEFAULT_LANG,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> str:
        """Generate the report for a resident, loading its own data window.

        Convenience wrapper over :meth:`generate_weekly_report` for callers
        that should not be responsible for selecting the time window.
        """
        window = self.load_window(resident_id, reference_ts, window_days)
        return await self.generate_weekly_report(
            window.episodes,
            events=events,
            llm_provider=llm_provider,
            reference_ts=window.reference_ts,
            timeout=timeout,
            lang=lang,
        )

    async def generate_weekly_report(
        self,
        episodes: list,
        events: Optional[list] = None,
        llm_provider: Optional[LLMProvider] = None,
        reference_ts: Optional[datetime] = None,
        timeout: float = 10.0,
        lang: str = DEFAULT_LANG,
    ) -> str:
        """Generate the weekly report as markdown in ``lang``.

        Args:
            episodes: episode records (EpisodeLog or dicts).
            events: optional events for the period.
            llm_provider: when supplied, LLM prose is attempted first.
            reference_ts: report anchor time.
            timeout: LLM call budget in seconds.
            lang: ``"en"`` or ``"zh"``. Selects both the prompt and the
                deterministic fallback language.

        Returns:
            Markdown using the canonical ``REPORT_BLOCKS`` headings.
        """
        if lang not in SUPPORTED_LANGS:
            lang = DEFAULT_LANG
        context = build_report_context(episodes, events, reference_ts, lang=lang)
        fallback = render_fallback_report(context, lang)

        if llm_provider is None:
            return fallback

        try:
            messages = [
                ChatMessage(role="system", content=report_prompt(lang)),
                ChatMessage(
                    role="user",
                    content=json.dumps(context, ensure_ascii=False, indent=2, default=str),
                ),
            ]
            resp = await asyncio.wait_for(llm_provider.chat(messages), timeout=timeout)
            if resp and resp.content and resp.content.strip():
                return resp.content.strip()
        except asyncio.TimeoutError:
            pass  # fall through to the deterministic renderer
        except Exception:
            pass  # provider failure must never surface as report content

        return fallback

    # ── Natural-language Q&A (keyword matching; care-support wording) ──

    # Each entry is (keywords_en, keywords_zh, answer_en, answer_zh). The
    # keyword lists are kept separate because question language and answer
    # language need not match.
    QA_ENTRIES = [
        (
            ("fever", "temperature", "feverish"),
            ("发烧", "发热", "体温"),
            "Surface temperature is tracked as a trend, not as a clinical "
            "measurement. A rising personal trend is worth a recheck with a "
            "dedicated thermometer.",
            "体表温度是按趋势追踪的，不是临床测量值。如果相对个人基线呈上升趋势，"
            "建议用专用体温计复测。",
        ),
        (
            ("heart rate", "heart"),
            ("心率", "心跳"),
            "Heart rate is estimated from WiFi BFI and mmWave radar, then "
            "arbitrated by confidence. The report shows deviation from this "
            "resident's own baseline rather than a fixed clinical range.",
            "心率由 WiFi BFI 与毫米波雷达估计，再按置信度仲裁。报告展示的是相对"
            "该居民个人基线的偏离，而不是固定临床区间。",
        ),
        (
            ("respiration", "breathing", "respiratory"),
            ("呼吸", "呼吸率"),
            "Respiratory rate is the most reliable contactless metric in this "
            "system and is tracked against the resident's personal baseline.",
            "呼吸率是本系统中最可靠的非接触指标，按居民个人基线追踪。",
        ),
        (
            ("fall", "fell", "fallen"),
            ("跌倒", "摔倒"),
            "Fall events are recorded with their evidence chain. Confirm the "
            "resident's current state, and involve family or a clinician when "
            "the event escalated to L3 or L4.",
            "跌倒事件连同证据链一并记录。请确认居民当前状态；若事件升级到 L3 或 "
            "L4，请联系家属或医生。",
        ),
        (
            ("blood pressure", "hypertension"),
            ("血压",),
            "Blood pressure is not instrumented in this system. Use a "
            "dedicated cuff.",
            "本系统未接入血压测量，请使用专用血压计。",
        ),
        (
            ("modality", "fusion", "mmwave", "wifi"),
            ("模态", "融合", "毫米波"),
            "The system uses WiFi BFI and mmWave radar as complementary radio "
            "branches. FusionEngine arbitrates them by confidence and quality "
            "constraints, and preserves disagreement as an explicit quality event.",
            "系统把 WiFi BFI 与毫米波雷达作为互补的射频分支，由 FusionEngine 按"
            "置信度与质量约束仲裁；分歧会被保留为显式的质量事件。",
        ),
        (
            ("confidence", "reliability", "nlos"),
            ("置信度", "遮挡", "可靠"),
            "Sensing confidence reflects NLOS blockage, contact quality, and "
            "environmental interference. Low confidence affects measurement "
            "reliability; it is not evidence of a health change.",
            "传感置信度反映 NLOS 遮挡、接触质量与环境干扰。低置信度影响的是测量"
            "可靠性，不是健康变化的证据。",
        ),
        (
            ("privacy", "data", "local"),
            ("隐私", "数据", "本地"),
            "Raw and derived sensing records stay on the local device. When an "
            "external model is used, only a minimized structured event context "
            "is sent.",
            "原始与派生感知记录保留在本地设备。使用外部模型时，只发送最小化的"
            "结构化事件上下文。",
        ),
    ]

    UNKNOWN_ANSWER = (
        "This question is outside the care-support scope of this system. "
        "Please consult a clinician or the resident's health handbook."
    )

    UNKNOWN_ANSWER_ZH = (
        "该问题超出了本系统的看护支持范围。请咨询医生或查阅居民健康手册。"
    )

    def answer_question(self, question: str, lang: str = DEFAULT_LANG) -> str:
        """Answer a resident/family question by keyword match.

        Matches against both keyword lists so a Chinese question still gets an
        answer when ``lang`` is English, and vice versa.
        """
        q = (question or "").lower()
        for keywords_en, keywords_zh, answer_en, answer_zh in self.QA_ENTRIES:
            if any(k in q for k in keywords_en) or any(k in q for k in keywords_zh):
                return answer_zh if lang == "zh" else answer_en
        return self.UNKNOWN_ANSWER_ZH if lang == "zh" else self.UNKNOWN_ANSWER
