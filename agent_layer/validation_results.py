"""Published validation results for SuperSenseDoctor.

Single source of truth shared by both rendering paths of the weekly care
report:

  - ``ReportAgent.generate_weekly_report`` (LLM prose, primary)
  - ``render_fallback_report``            (deterministic markdown, fallback)

and by the web display layer.

All values are transcribed from the published paper:

    Xuwen Zhang et al. 2026. SuperSenseDoctor: A Multimodal and Contactless
    Agent for Health Tracking. In Companion of the 2026 ACM International
    Joint Conference on Pervasive and Ubiquitous Computing (UbiComp
    Companion '26). ACM, New York, NY, USA.
    https://doi.org/10.1145/3798063.3837323

    Table 1  -> VITAL_SIGN_VALIDATION, FALL_RECOGNITION
    §4.1     -> MULTI_INTERVAL_VALIDATION
    §4.2     -> AGENT_EVALUATION
    Table 2  -> INDICATOR_COVERAGE

These are REPORTED RESULTS, not live measurements. They are displayed as a
read-only reference block and must never be presented as resident-specific
evidence.
"""

from typing import Optional

# ── Paper identity ────────────────────────────────────────────────────────

PAPER = {
    "title": "SuperSenseDoctor: A Multimodal and Contactless Agent for Health Tracking",
    "venue": "UbiComp Companion '26",
    "location": "Shanghai, China",
    "doi": "10.1145/3798063.3837323",
    "url": "https://doi.org/10.1145/3798063.3837323",
}

# Provenance tag rendered next to the numbers so a viewer can never mistake
# reported results for live resident data.
PROVENANCE = "reported_results"

# ── Table 1: multimodal sensing validation ────────────────────────────────
# Vital-sign rows report MAE and RMSD in bpm.

VITAL_SIGN_VALIDATION = [
    {
        "task": "Heart rate",
        "task_zh": "心率",
        "ground_truth": "Huawei Watch GT 3",
        "ground_truth_zh": "Huawei Watch GT 3",
        "unit": "bpm",
        "methods": [
            {"method": "WiFi BFI", "mae": 4.237, "rmsd": 4.396},
            {"method": "mmWave", "mae": 4.121, "rmsd": 4.290},
            {"method": "Final (fused)", "mae": 1.994, "rmsd": 3.142, "is_final": True},
        ],
    },
    {
        "task": "Respiratory rate",
        "task_zh": "呼吸率",
        "ground_truth": "Respiration belt",
        "ground_truth_zh": "呼吸带",
        "unit": "bpm",
        "methods": [
            {"method": "WiFi BFI", "mae": 2.689, "rmsd": 3.293},
            {"method": "mmWave", "mae": 1.629, "rmsd": 1.876},
            {"method": "Final (fused)", "mae": 0.197, "rmsd": 0.263, "is_final": True},
        ],
    },
]

FALL_RECOGNITION = {
    "task": "Fall recognition",
    "task_zh": "跌倒识别",
    "ground_truth": "Labeled event replays",
    "ground_truth_zh": "带标注的事件回放",
    "method": "BFI and mmWave evidence",
    "accuracy": 0.965,
    "accuracy_pct": 96.5,
}

# Improvement of the fused estimate over each single branch (§4.1).
FUSION_IMPROVEMENT = [
    {"task": "Heart rate", "vs": "mmWave", "mae_reduction_pct": 51.6},
    {"task": "Heart rate", "vs": "WiFi BFI", "mae_reduction_pct": 52.9},
    {"task": "Respiratory rate", "vs": "mmWave", "mae_reduction_pct": 87.9},
    {"task": "Respiratory rate", "vs": "WiFi BFI", "mae_reduction_pct": 92.7},
]

# ── §4.1: multi-interval sensing validation ───────────────────────────────

MULTI_INTERVAL_VALIDATION = {
    "intervals": 9,
    "state_rows": 2686,
    "window_sec": 1,
    "interval_detail": "seven 300 s intervals + one 249 s interval + one 337 s interval",
}

# ── §4.2: Agent scenario evaluation ───────────────────────────────────────

AGENT_EVALUATION = {
    "deterministic": {
        "label": "Deterministic rule screening",
        "cases": 60,
        "cases_description": (
            "manually specified state objects spanning normal sensing, heart-rate and "
            "respiratory deviations, falls, modality conflicts, NLOS, low confidence in "
            "both radio branches, compound risk, and threshold boundaries"
        ),
        "matched": 58,
        "fusion_cases": 24,
        "fusion_matched": 23,
    },
    "triage": {
        "label": "LLM triage (hidden answer key)",
        "scenarios": 24,
        "scenarios_description": (
            "each scenario specifies the expected L0-L4 tier, action channel, evidence "
            "anchors, JSON shape, care-support boundary, and severe risk status"
        ),
        "tier_and_channel_matched": 22,
        "trace_preserved": 23,
        "model": "GPT-5.4",
        "p95_latency_ms": 6560.7,
    },
    "overall": {
        "criteria_total": 213,
        "criteria_passed": 206,
        "pass_rate": 0.967,
        "pass_rate_pct": 96.7,
    },
    "latency": {
        "deterministic_p95_ms": 0.56,
        "llm_triage_p95_ms": 6560.7,
    },
}

# ── Table 2: disease-relevant indicator coverage ──────────────────────────
# "x" = indicator contributes to the screening direction, "-" = not covered.

INDICATOR_COLUMNS = [
    ("respiratory_rate", "Respiratory rate and pattern"),
    ("heart_rate", "Heart rate"),
    ("gait_balance", "Gait and balance"),
    ("activity_mobility", "Activity and mobility"),
    ("sleep_rhythm", "Sleep rhythm and fragmentation"),
    ("surface_temp", "Surface-temp. trend"),
]

# Chinese display labels for the same columns, positionally aligned with
# INDICATOR_COLUMNS so a bilingual renderer can zip the two.
INDICATOR_COLUMNS_ZH = [
    ("respiratory_rate", "呼吸率与呼吸模式"),
    ("heart_rate", "心率"),
    ("gait_balance", "步态与平衡"),
    ("activity_mobility", "活动与行动能力"),
    ("sleep_rhythm", "睡眠节律与碎片化"),
    ("surface_temp", "体表温度趋势"),
]

INDICATOR_COVERAGE = [
    {
        "condition": "COPD exacerbation",
        "condition_zh": "慢阻肺急性加重",
        "refs": [5],
        "coverage": {
            "respiratory_rate": True, "heart_rate": True, "gait_balance": False,
            "activity_mobility": True, "sleep_rhythm": False, "surface_temp": False,
        },
    },
    {
        "condition": "Obstructive sleep apnea",
        "condition_zh": "阻塞性睡眠呼吸暂停",
        "refs": [9, 15],
        "coverage": {
            "respiratory_rate": True, "heart_rate": True, "gait_balance": False,
            "activity_mobility": False, "sleep_rhythm": True, "surface_temp": False,
        },
    },
    {
        "condition": "Heart-failure decompensation",
        "condition_zh": "心力衰竭失代偿",
        "refs": [3],
        "coverage": {
            "respiratory_rate": True, "heart_rate": True, "gait_balance": False,
            "activity_mobility": True, "sleep_rhythm": False, "surface_temp": False,
        },
    },
    {
        "condition": "Parkinson's disease",
        "condition_zh": "帕金森病",
        "refs": [13],
        "coverage": {
            "respiratory_rate": False, "heart_rate": False, "gait_balance": True,
            "activity_mobility": True, "sleep_rhythm": True, "surface_temp": False,
        },
    },
    {
        "condition": "Cognitive decline and dementia",
        "condition_zh": "认知衰退与痴呆",
        "refs": [4, 14],
        "coverage": {
            "respiratory_rate": False, "heart_rate": False, "gait_balance": True,
            "activity_mobility": True, "sleep_rhythm": True, "surface_temp": False,
        },
    },
    {
        "condition": "Acute respiratory infection",
        "condition_zh": "急性呼吸道感染",
        "refs": [12],
        "coverage": {
            "respiratory_rate": True, "heart_rate": True, "gait_balance": False,
            "activity_mobility": True, "sleep_rhythm": True, "surface_temp": True,
        },
    },
]

# ── Table 2 support: the paper's explicit non-goal ────────────────────────

CLINICAL_BOUNDARY = (
    "Indicator coverage describes screening directions, not diagnoses. "
    "Clinical interpretation is left to qualified professionals."
)

CLINICAL_BOUNDARY_ZH = (
    "指标覆盖描述的是筛查方向，不是诊断。临床解释应由专业医务人员完成。"
)


def validation_block() -> dict:
    """Assemble the read-only validation block consumed by both renderers."""
    return {
        "provenance": PROVENANCE,
        "paper": PAPER,
        "vital_signs": VITAL_SIGN_VALIDATION,
        "fall_recognition": FALL_RECOGNITION,
        "fusion_improvement": FUSION_IMPROVEMENT,
        "multi_interval": MULTI_INTERVAL_VALIDATION,
        "agent_evaluation": AGENT_EVALUATION,
        "indicator_columns": INDICATOR_COLUMNS,
        "indicator_columns_zh": INDICATOR_COLUMNS_ZH,
        "indicator_coverage": INDICATOR_COVERAGE,
        "clinical_boundary": CLINICAL_BOUNDARY,
        "clinical_boundary_zh": CLINICAL_BOUNDARY_ZH,
    }


def headline_metrics() -> list[dict]:
    """Flat list of the four headline numbers, for compact display."""
    hr = next(r for r in VITAL_SIGN_VALIDATION if r["task"] == "Heart rate")
    rr = next(r for r in VITAL_SIGN_VALIDATION if r["task"] == "Respiratory rate")
    hr_final = next(m for m in hr["methods"] if m.get("is_final"))
    rr_final = next(m for m in rr["methods"] if m.get("is_final"))
    return [
        {
            "key": "hr_mae",
            "label": "Heart rate MAE",
            "value": hr_final["mae"],
            "unit": "bpm",
            "detail": f"RMSD {hr_final['rmsd']} bpm",
        },
        {
            "key": "rr_mae",
            "label": "Respiratory rate MAE",
            "value": rr_final["mae"],
            "unit": "bpm",
            "detail": f"RMSD {rr_final['rmsd']} bpm",
        },
        {
            "key": "fall_accuracy",
            "label": "Fall recognition",
            "value": FALL_RECOGNITION["accuracy_pct"],
            "unit": "%",
            "detail": "accuracy",
        },
        {
            "key": "agent_checklist",
            "label": "Agent checklist",
            "value": AGENT_EVALUATION["overall"]["pass_rate_pct"],
            "unit": "%",
            "detail": (
                f"{AGENT_EVALUATION['overall']['criteria_passed']}/"
                f"{AGENT_EVALUATION['overall']['criteria_total']} criteria"
            ),
        },
    ]
