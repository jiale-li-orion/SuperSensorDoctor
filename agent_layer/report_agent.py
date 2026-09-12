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


# ── Display vocabulary ─────────────────────────────────────────────────────

# L0-L4 policy card, matching the paper's tier->channel mapping.
TIER_LABELS = {
    "L0": "Routine observation",
    "L1": "Continued observation",
    "L2": "Resident prompt",
    "L3": "Family notification",
    "L4": "Emergency routing",
}

TIER_CHANNELS = {
    "L0": "none",
    "L1": "none",
    "L2": "screen",
    "L3": "family_push",
    "L4": "emergency",
}

CHANNEL_LABELS = {
    "none": "Record only",
    "screen": "Resident screen",
    "family_push": "Family notification",
    "emergency": "Emergency routing (human confirmation)",
}

SOURCE_LABELS = {
    "RCP_NEWS2_2017_REFERENCE": "Absolute physiological reference band (RCP NEWS2 2017)",
    "NICE_NG249_2025": "Fall assessment evidence needs (NICE NG249 2025)",
    "RESIDENT_HISTORY": "Personal longitudinal baseline",
    "SENSOR_FUSION": "Cross-modal measurement reliability",
    "ACTIVITY_CONTEXT": "Activity and posture context",
    "PROJECT_POLICY": "Project triage policy",
}

SENSING_QUALITY_LABELS = {
    "reliable": "Reliable",
    "degraded": "Degraded",
    "unreliable": "Unreliable",
    "unknown": "Unknown",
}

EVENT_LABELS = {
    "hr_abnormal": "Heart-rate deviation",
    "rr_bradypnea": "Bradypnea",
    "rr_tachypnea": "Tachypnea",
    "rr_baseline_deviation": "Respiratory baseline deviation",
    "temp_abnormal": "Surface-temperature deviation",
    "fall_detected": "Fall detected",
    "fall_no_physiological_change": "Fall without physiological change",
    "low_confidence": "Low confidence in both radio branches",
    "nlos_occlusion": "NLOS degradation",
    "modality_conflict": "Modality conflict",
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
    "title": "Privacy and Human Oversight",
    "statements": [
        "Raw and derived sensing records remain local to the prototype host.",
        "When an external OpenAI-compatible model is selected, only a minimized "
        "structured event context leaves the device.",
        "This report is care support. It is not a medical diagnosis, treatment "
        "plan, or medication advice.",
        "High-risk routing requires authorized human confirmation.",
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


def _event_label(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type or "unknown")


# ── Context builder ───────────────────────────────────────────────────────

def build_report_context(
    episodes: list,
    events: Optional[list] = None,
    reference_ts: Optional[datetime] = None,
    window_days: int = 7,
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

    Returns:
        A dict keyed by ``REPORT_BLOCKS``.
    """
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
            "summary": _empty_summary(),
            "evidence_trail": None,
            "sensing_quality": _empty_sensing_quality(events),
            "action_routing": _empty_action_routing(),
            "uncertainty": _empty_uncertainty(),
            "vital_trends": _empty_vital_trends(),
            "personal_baseline": [],
            "validation": validation_block(),
            "indicator_coverage": validation_block(),
            "privacy_boundary": PRIVACY_BOUNDARY,
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
        "highest_tier_label": TIER_LABELS[highest_tier],
        "most_frequent_tier": most_frequent,
        "most_frequent_tier_label": TIER_LABELS[most_frequent],
        "escalated_count": escalated,
        "reflex_count": reflex_count,
        "deliberative_count": len(records) - reflex_count,
    }

    # ── Evidence trail (representative record: highest tier, then latest) ──
    representative = max(
        records,
        key=lambda r: (severity_order.index(r["tier"]), r["timestamp"]),
    )
    evidence_trail = _build_evidence_trail(representative)

    # ── Sensing quality and fusion arbitration ──
    sensing_quality = _build_sensing_quality(records, events)

    # ── Action routing ──
    action_routing = _build_action_routing(records, events)

    # ── Uncertainty and reflex path ──
    uncertainty = _build_uncertainty(records)

    # ── Vital trends and personal baseline from evidence chains ──
    vital_trends = _build_vital_trends(records)
    personal_baseline = _build_personal_baseline(records)

    validation = validation_block()

    return {
        "period": period,
        "empty": False,
        "summary": summary,
        "evidence_trail": evidence_trail,
        "sensing_quality": sensing_quality,
        "action_routing": action_routing,
        "uncertainty": uncertainty,
        "vital_trends": vital_trends,
        "personal_baseline": personal_baseline,
        "validation": validation,
        "indicator_coverage": validation,
        "privacy_boundary": PRIVACY_BOUNDARY,
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


def _build_evidence_trail(rec: dict) -> dict:
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
            "source_label": SOURCE_LABELS.get(source, source or "unspecified"),
        })

    tools_called = [str(t) for t in _as_list(audit.get("tools_called", []))]
    channel = str(rec["action"].get("channel", "none") or "none")

    return {
        "episode_id": rec["episode_id"],
        "timestamp": rec["timestamp"].isoformat(),
        "trigger": {
            "event_id": event.get("event_id", rec["event_id"]),
            "event_type": event.get("event_type", ""),
            "event_label": _event_label(event.get("event_type", "")),
            "reason": event.get("trigger_reason", ""),
        },
        "interpretation": decision.get("event_interpretation", ""),
        "anchors": anchors,
        "tier": rec["tier"],
        "tier_label": TIER_LABELS[rec["tier"]],
        "channel": channel,
        "channel_label": CHANNEL_LABELS.get(channel, channel),
        "reflex": bool(audit.get("reflex")),
        "step_count": audit.get("step_count"),
        "tools_called": tools_called,
        "evidence_used": _as_list(decision.get("evidence_used", [])),
        "safety_boundary": decision.get("safety_boundary", "care_support_only"),
    }


def _build_sensing_quality(records: list, events: list) -> dict:
    """Quality events and cross-modal arbitration, from events + evidence."""
    nlos = low_conf = conflict = 0
    quality_flags = 0
    wifi_confs, mmwave_confs = [], []
    dominant_counts: dict = {}

    for ev in events:
        et = _field(ev, "event_type", "") or ""
        markers = _as_dict(_field(ev, "rule_markers", {}))

        if et == "nlos_occlusion" or markers.get("nlos_flag"):
            nlos += 1
        if et == "low_confidence":
            low_conf += 1
        if et == "modality_conflict":
            conflict += 1

        for metric in ("hr", "rr"):
            dominant = markers.get(f"{metric}_dominant")
            if dominant:
                key = f"{metric}:{dominant}"
                dominant_counts[key] = dominant_counts.get(key, 0) + 1

        wc = _field(ev, "wifi_confidence")
        mc = _field(ev, "mmwave_confidence")
        if wc is not None:
            wifi_confs.append(float(wc))
        if mc is not None:
            mmwave_confs.append(float(mc))

    for rec in records:
        summary = _as_dict(rec["evidence"].get("sensing_summary", {}))
        if summary.get("quality_event"):
            quality_flags += 1
        if summary.get("nlos_flag"):
            nlos += 1
        wc, mc = summary.get("wifi_confidence"), summary.get("mmwave_confidence")
        if wc is not None:
            wifi_confs.append(float(wc))
        if mc is not None:
            mmwave_confs.append(float(mc))
        for metric, field in (("hr", "hr_source"), ("rr", "rr_source")):
            src = summary.get(field)
            if src:
                key = f"{metric}:{src}"
                dominant_counts[key] = dominant_counts.get(key, 0) + 1

    total_windows = len(records) or 1
    quality_signals = nlos + low_conf + conflict + quality_flags
    quality_rate = round(quality_signals / total_windows * 100, 1) if records else 0.0

    dominant = [
        {"metric": k.split(":", 1)[0], "branch": k.split(":", 1)[1], "count": v}
        for k, v in sorted(dominant_counts.items(), key=lambda kv: -kv[1])
    ]

    return {
        "nlos_count": nlos,
        "low_confidence_count": low_conf,
        "modality_conflict_count": conflict,
        "quality_event_count": quality_flags,
        "quality_signal_count": quality_signals,
        "quality_event_rate_pct": quality_rate,
        "mean_wifi_confidence_pct": round(sum(wifi_confs) / len(wifi_confs) * 100) if wifi_confs else None,
        "mean_mmwave_confidence_pct": round(sum(mmwave_confs) / len(mmwave_confs) * 100) if mmwave_confs else None,
        "dominant_branches": dominant,
        "note": (
            "Quality events affect measurement reliability, not health status. "
            "A degraded measurement is not physiological deterioration."
        ),
    }


def _empty_sensing_quality(events: list) -> dict:
    base = _build_sensing_quality([], events)
    return base


def _build_action_routing(records: list, events: list) -> dict:
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
            {"channel": ch, "label": CHANNEL_LABELS.get(ch, ch), "count": n}
            for ch, n in sorted(channel_counts.items(), key=lambda kv: -kv[1])
        ],
        "event_types": [
            {"type": et, "label": _event_label(et), "count": n}
            for et, n in sorted(type_counts.items(), key=lambda kv: -kv[1])
        ],
    }


def _empty_action_routing() -> dict:
    return {"channels": [], "event_types": []}


def _build_uncertainty(records: list) -> dict:
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
            {"level": k, "label": SENSING_QUALITY_LABELS.get(k, k), "count": v}
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


def _build_vital_trends(records: list) -> dict:
    """Latest/mean/range per metric, read from each evidence chain."""
    buckets = {"heart_rate": [], "respiration_rate": [], "body_temp": []}
    labels = {
        "heart_rate": ("Heart rate", "bpm"),
        "respiration_rate": ("Respiratory rate", "/min"),
        "body_temp": ("Surface temperature", "°C"),
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
        name, unit = labels[key]
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


def _empty_vital_trends() -> dict:
    return _build_vital_trends([])


def _build_personal_baseline(records: list) -> list:
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
        "heart_rate": ("Heart rate", "bpm"),
        "respiration_rate": ("Respiratory rate", "/min"),
        "body_temp": ("Surface temperature", "°C"),
    }
    out = []
    for key, (name, unit) in labels.items():
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

def render_fallback_report(context: dict) -> str:
    """Render the canonical context as English markdown without an LLM.

    Emits the same ``##`` blocks as the LLM contract so downstream rendering
    and display never depend on which path produced the report.
    """
    lines: list[str] = []
    period = context["period"]

    # ── Weekly Summary ──
    lines.append("## Weekly Summary")
    summary = context["summary"]
    if context.get("empty") or summary["total_records"] == 0:
        lines.append("")
        lines.append(
            f"No triage records in the reporting period "
            f"({period['start_date']} to {period['end_date']})."
        )
        _append_static_blocks(lines, context)
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"Period **{period['start_date']} to {period['end_date']}** "
        f"({period['window_days']}-day window)."
    )
    lines.append("")
    lines.append(f"- Total triage records: **{summary['total_records']}**")
    for tier in ["L0", "L1", "L2", "L3", "L4"]:
        count = summary["tier_counts"].get(tier, 0)
        if count:
            lines.append(f"  - {tier} {TIER_LABELS[tier]}: {count}")
    lines.append(
        f"- Highest tier reached: **{summary['highest_tier']} "
        f"{summary['highest_tier_label']}**"
    )
    lines.append(
        f"- Most frequent tier: **{summary['most_frequent_tier']} "
        f"{summary['most_frequent_tier_label']}**"
    )
    lines.append(
        f"- Records requiring escalation (L3/L4): **{summary['escalated_count']}**"
    )

    # ── Evidence Trail ──
    lines.append("")
    lines.append("## Evidence Trail")
    trail = context["evidence_trail"]
    if not trail:
        lines.append("")
        lines.append("No data available for this period.")
    else:
        lines.append("")
        lines.append(
            "Representative record "
            f"`{trail['episode_id']}` at {trail['timestamp']}:"
        )
        lines.append("")
        lines.append(
            f"- **Nurse event**: {trail['trigger']['event_label']} "
            f"(`{trail['trigger']['event_type']}`, id `{trail['trigger']['event_id']}`)"
        )
        if trail["trigger"]["reason"]:
            lines.append(f"  - Trigger: {trail['trigger']['reason']}")
        if trail["interpretation"]:
            lines.append(f"- **Interpretation**: {trail['interpretation']}")
        if trail["anchors"]:
            lines.append("- **Evidence anchors**:")
            for anchor in trail["anchors"]:
                finding = anchor["finding"] or anchor["type"] or "unspecified"
                lines.append(f"  - {finding} — _{anchor['source_label']}_")
        else:
            lines.append("- **Evidence anchors**: none recorded")
        lines.append(
            f"- **Triage tier**: {trail['tier']} {trail['tier_label']}"
        )
        lines.append(
            f"- **Delivery channel**: {trail['channel_label']} (`{trail['channel']}`)"
        )
        lines.append(
            f"- **Persisted EpisodeLog**: `{trail['episode_id']}`"
        )
        path = "reflex path (deterministic, no LLM deliberation)" if trail["reflex"] \
            else f"deliberative ReAct loop ({trail['step_count']} steps)"
        lines.append(f"- **Decision path**: {path}")
        if trail["tools_called"]:
            lines.append(
                "- **Tools used**: " + ", ".join(f"`{t}`" for t in trail["tools_called"])
            )
        lines.append(f"- **Safety boundary**: `{trail['safety_boundary']}`")

    # ── Sensing Quality and Fusion Arbitration ──
    lines.append("")
    lines.append("## Sensing Quality and Fusion Arbitration")
    quality = context["sensing_quality"]
    lines.append("")
    lines.append(f"- NLOS degradation: **{quality['nlos_count']}**")
    lines.append(f"- Low confidence in both radio branches: **{quality['low_confidence_count']}**")
    lines.append(f"- Modality conflicts: **{quality['modality_conflict_count']}**")
    lines.append(f"- Quality-flagged share: **{quality['quality_event_rate_pct']}%** "
                 f"({quality['quality_signal_count']} signals over "
                 f"{context['summary']['total_records']} records)")
    if quality["mean_wifi_confidence_pct"] is not None:
        lines.append(f"- Mean WiFi BFI confidence: **{quality['mean_wifi_confidence_pct']}%**")
    if quality["mean_mmwave_confidence_pct"] is not None:
        lines.append(f"- Mean mmWave confidence: **{quality['mean_mmwave_confidence_pct']}%**")
    if quality["dominant_branches"]:
        branches = ", ".join(
            f"{d['metric'].upper()} -> {d['branch']} ({d['count']})"
            for d in quality["dominant_branches"]
        )
        lines.append(f"- Dominant arbitration branch: {branches}")
    lines.append("")
    lines.append(quality["note"])

    # ── Action Routing ──
    lines.append("")
    lines.append("## Action Routing")
    routing = context["action_routing"]
    lines.append("")
    if routing["channels"]:
        for item in routing["channels"]:
            lines.append(f"- {item['label']} (`{item['channel']}`): **{item['count']}**")
    else:
        lines.append("- No data available for this period.")
    if routing["event_types"]:
        lines.append("")
        lines.append("Event types:")
        lines.append("")
        for item in routing["event_types"]:
            lines.append(f"- {item['label']}: {item['count']}")

    # ── Uncertainty and Reflex Path ──
    lines.append("")
    lines.append("## Uncertainty and Reflex Path")
    unc = context["uncertainty"]
    lines.append("")
    lines.append(
        f"- Records requesting a recheck: **{unc['needs_recheck_count']}** "
        f"({unc['needs_recheck_pct']}%)"
    )
    if unc["missing_evidence"]:
        lines.append("- Missing evidence declared:")
        for item in unc["missing_evidence"]:
            lines.append(f"  - {item['item']}: {item['count']}")
    else:
        lines.append("- Missing evidence declared: none")
    if unc["sensing_quality_levels"]:
        levels = ", ".join(
            f"{lv['label']} ({lv['count']})" for lv in unc["sensing_quality_levels"]
        )
        lines.append(f"- Self-assessed sensing quality: {levels}")
    lines.append(
        f"- Decision path: reflex **{summary['reflex_count']}** / "
        f"deliberative **{summary['deliberative_count']}**"
    )

    # ── Vital Trends ──
    lines.append("")
    lines.append("## Vital Trends")
    lines.append("")
    any_vitals = False
    for metric in context["vital_trends"]["metrics"]:
        if metric["latest"] is None:
            lines.append(f"- {metric['name']}: no data")
            continue
        any_vitals = True
        lines.append(
            f"- {metric['name']}: latest **{metric['latest']} {metric['unit']}**, "
            f"mean {metric['mean']}, range {metric['min']}-{metric['max']} "
            f"({metric['count']} samples)"
        )
    if not any_vitals:
        lines.append("")
        lines.append("No data available for this period.")

    # ── Personal Baseline ──
    lines.append("")
    lines.append("## Personal Baseline")
    lines.append("")
    for item in context["personal_baseline"]:
        if item["z_score"] is None:
            lines.append(f"- {item['name']}: N/A")
        else:
            lines.append(
                f"- {item['name']}: z = **{item['z_score']:+.2f}** "
                f"(value {item['value']}, baseline {item['mean']} ± {item['std']})"
            )

    _append_static_blocks(lines, context)
    return "\n".join(lines)


def _append_static_blocks(lines: list, context: dict) -> None:
    """Blocks that do not depend on the week's episodes."""
    # ── Published Validation Results ──
    lines.append("")
    lines.append("## Published Validation Results")
    lines.append("")
    validation = context["validation"]
    paper = validation["paper"]
    lines.append(
        f"Reported results from _{paper['title']}_ ({paper['venue']}, "
        f"[{paper['doi']}]({paper['url']})). These are published benchmarks, "
        "not this resident's data."
    )
    lines.append("")
    for row in validation["vital_signs"]:
        lines.append(f"- {row['task']} (reference: {row['ground_truth']}):")
        for method in row["methods"]:
            mark = " **(final fused estimate)**" if method.get("is_final") else ""
            lines.append(
                f"  - {method['method']}: MAE {method['mae']} {row['unit']}, "
                f"RMSD {method['rmsd']} {row['unit']}{mark}"
            )
    fall = validation["fall_recognition"]
    lines.append(
        f"- {fall['task']} ({fall['ground_truth']}): **{fall['accuracy_pct']}% accuracy**"
    )
    multi = validation["multi_interval"]
    lines.append(
        f"- Multi-interval validation: {multi['intervals']} intervals, "
        f"{multi['state_rows']} one-second state rows"
    )
    agent_eval = validation["agent_evaluation"]
    overall = agent_eval["overall"]
    lines.append(
        f"- Agent checklist: **{overall['criteria_passed']}/{overall['criteria_total']} "
        f"({overall['pass_rate_pct']}%)**"
    )
    lines.append(
        f"  - Deterministic screening: {agent_eval['deterministic']['matched']}/"
        f"{agent_eval['deterministic']['cases']} event matches; "
        f"fusion arbitration {agent_eval['deterministic']['fusion_matched']}/"
        f"{agent_eval['deterministic']['fusion_cases']}"
    )
    triage = agent_eval["triage"]
    lines.append(
        f"  - Hidden-answer triage ({triage['model']}): tier and channel "
        f"{triage['tier_and_channel_matched']}/{triage['scenarios']}; "
        f"trace preserved {triage['trace_preserved']}/{triage['scenarios']}"
    )
    lines.append(
        f"  - P95 latency: deterministic "
        f"{agent_eval['latency']['deterministic_p95_ms']} ms, LLM triage "
        f"{agent_eval['latency']['llm_triage_p95_ms']} ms"
    )

    # ── Indicator and Screening Coverage ──
    lines.append("")
    lines.append("## Indicator and Screening Coverage")
    lines.append("")
    for row in validation["indicator_coverage"]:
        covered = [
            label for key, label in validation["indicator_columns"]
            if row["coverage"].get(key)
        ]
        lines.append(f"- **{row['condition']}**: {', '.join(covered) if covered else 'none'}")
    lines.append("")
    lines.append(validation["clinical_boundary"])

    # ── Privacy and Human Oversight ──
    lines.append("")
    lines.append(f"## {context['privacy_boundary']['title']}")
    lines.append("")
    for statement in context["privacy_boundary"]["statements"]:
        lines.append(f"- {statement}")


# ── Agent ─────────────────────────────────────────────────────────────────

class ReportAgent:
    """Weekly care report: LLM prose with a deterministic markdown fallback."""

    def build_context(
        self,
        episodes: list,
        events: Optional[list] = None,
        reference_ts: Optional[datetime] = None,
    ) -> dict:
        """Public accessor for the canonical evidence context."""
        return build_report_context(episodes, events, reference_ts)

    async def generate_weekly_report(
        self,
        episodes: list,
        events: Optional[list] = None,
        llm_provider: Optional[LLMProvider] = None,
        reference_ts: Optional[datetime] = None,
        timeout: float = 10.0,
    ) -> str:
        """Generate the weekly report as English markdown.

        Args:
            episodes: episode records (EpisodeLog or dicts).
            events: optional events for the period.
            llm_provider: when supplied, LLM prose is attempted first.
            reference_ts: report anchor time.
            timeout: LLM call budget in seconds.

        Returns:
            Markdown using the canonical ``REPORT_BLOCKS`` headings.
        """
        context = build_report_context(episodes, events, reference_ts)
        fallback = render_fallback_report(context)

        if llm_provider is None:
            return fallback

        try:
            messages = [
                ChatMessage(role="system", content=REPORT_PROMPT),
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

    QA_PAIRS = {
        "fever": (
            "Surface temperature is tracked as a trend, not as a clinical "
            "measurement. A rising personal trend is worth a recheck with a "
            "dedicated thermometer."
        ),
        "temperature": (
            "The thermal array provides a body-surface temperature trend. It "
            "supports screening direction only and does not replace a "
            "clinical thermometer."
        ),
        "heart rate": (
            "Heart rate is estimated from WiFi BFI and mmWave radar, then "
            "arbitrated by confidence. The report shows deviation from this "
            "resident's own baseline rather than a fixed clinical range."
        ),
        "respiration": (
            "Respiratory rate is the most reliable contactless metric in this "
            "system and is tracked against the resident's personal baseline."
        ),
        "fall": (
            "Fall events are recorded with their evidence chain. Confirm the "
            "resident's current state, and involve family or a clinician when "
            "the event escalated to L3 or L4."
        ),
        "blood pressure": (
            "Blood pressure is not instrumented in this system. Use a "
            "dedicated cuff."
        ),
        "modality": (
            "The system uses WiFi BFI and mmWave radar as complementary radio "
            "branches. FusionEngine arbitrates them by confidence and quality "
            "constraints, and preserves disagreement as an explicit quality event."
        ),
        "confidence": (
            "Sensing confidence reflects NLOS blockage, contact quality, and "
            "environmental interference. Low confidence affects measurement "
            "reliability; it is not evidence of a health change."
        ),
        "privacy": (
            "Raw and derived sensing records stay on the local device. When an "
            "external model is used, only a minimized structured event context "
            "is sent."
        ),
    }

    UNKNOWN_ANSWER = (
        "This question is outside the care-support scope of this system. "
        "Please consult a clinician or the resident's health handbook."
    )

    def answer_question(self, question: str) -> str:
        """Answer a resident/family question by keyword match."""
        q = (question or "").lower()
        for keyword, answer in self.QA_PAIRS.items():
            if keyword in q:
                return answer
        return self.UNKNOWN_ANSWER
