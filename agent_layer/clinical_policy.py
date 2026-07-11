"""Clinical reference layer for SuperSenseDoctor.

Important:
- NEWS2 bands are used as absolute physiological deviation references only.
- This project does NOT compute a complete NEWS2 score because SpO2,
  systolic blood pressure, consciousness/new confusion, and oxygen
  supplementation are not available.
- Personal z-score and L0-L4 mapping are project policies, not clinical
  guideline claims.
"""

from __future__ import annotations

from typing import Optional

from agent_layer.state_objects import HealthEvent


CLINICAL_POLICY_VERSION = "clinical-v1-2026-07"

ALLOWED_CLINICAL_SOURCES = {
    "RCP_NEWS2_2017_REFERENCE",
    "NICE_NG249_2025",
    "RESIDENT_HISTORY",
    "SENSOR_FUSION",
    "ACTIVITY_CONTEXT",
    "PROJECT_POLICY",
}


def news2_reference(metric: str, value: Optional[float]) -> dict:
    """Map one available vital sign to the corresponding NEWS2 reference band.

    This is NOT a complete NEWS2 score.
    """
    if value is None:
        return {
            "status": "missing",
            "value": None,
            "source": "RCP_NEWS2_2017_REFERENCE",
        }

    value = float(value)

    if metric == "rr":
        if value <= 8:
            band, reference_score = "marked_low", 3
        elif value <= 11:
            band, reference_score = "low", 1
        elif value <= 20:
            band, reference_score = "reference", 0
        elif value <= 24:
            band, reference_score = "elevated", 2
        else:
            band, reference_score = "marked_high", 3

    elif metric == "hr":
        if value <= 40:
            band, reference_score = "marked_low", 3
        elif value <= 50:
            band, reference_score = "low", 1
        elif value <= 90:
            band, reference_score = "reference", 0
        elif value <= 110:
            band, reference_score = "elevated", 1
        elif value <= 130:
            band, reference_score = "high", 2
        else:
            band, reference_score = "marked_high", 3

    elif metric == "temp":
        if value <= 35.0:
            band, reference_score = "marked_low", 3
        elif value <= 36.0:
            band, reference_score = "low", 1
        elif value <= 38.0:
            band, reference_score = "reference", 0
        elif value <= 39.0:
            band, reference_score = "elevated", 1
        else:
            band, reference_score = "high", 2

    else:
        return {
            "status": "unsupported",
            "value": value,
        }

    return {
        "status": "ok",
        "value": value,
        "band": band,
        "reference_score": reference_score,
        "source": "RCP_NEWS2_2017_REFERENCE",
        "warning": "Reference band only; not a complete NEWS2 score.",
    }


def build_clinical_summary(event: HealthEvent) -> dict:
    """Build system-generated medical reasoning context.

    The LLM should interpret this object, not invent its own thresholds.
    """
    state = event.state
    ctx = event.context

    personal_baseline = ctx.personal_baseline if ctx else None
    duration_sec = (
        ctx.duration_sec
        if ctx is not None
        else int(event.rule_markers.get("duration_sec", 0))
    )

    fall_detected = state.fall_status == "fall"

    fall_missing_evidence = []
    if fall_detected:
        fall_missing_evidence = [
            "injury_status",
            "loss_of_consciousness",
            "able_to_get_up_independently",
            "recurrent_falls_history",
        ]

    return {
        "policy_version": CLINICAL_POLICY_VERSION,

        "scope": {
            "full_news2_score_available": False,
            "reason": (
                "System lacks SpO2, systolic blood pressure, "
                "consciousness/new confusion, and oxygen supplementation."
            ),
        },

        "absolute_reference": {
            "heart_rate": news2_reference("hr", state.heart_rate),
            "respiration_rate": news2_reference(
                "rr", state.respiration_rate
            ),
            "body_temp": news2_reference("temp", state.body_temp),
        },

        "personal_baseline": personal_baseline,

        "persistence": {
            "duration_sec": duration_sec,
            "is_persistent_5min": duration_sec >= 300,
            "is_persistent_10min": duration_sec >= 600,
            "source": "PROJECT_POLICY",
        },

        "activity_context": {
            "activity_state": state.activity_state,
            "posture": state.posture,
            "source": "ACTIVITY_CONTEXT",
        },

        "sensing_quality": {
            "wifi_confidence": state.wifi_confidence,
            "mmwave_confidence": state.mmwave_confidence,
            "thermal_confidence": state.thermal_confidence,
            "hr_conf": state.hr_conf,
            "rr_conf": state.rr_conf,
            "nlos_flag": state.nlos_flag,
            "quality_event": bool(state.quality_event),
            "missing_modalities": state.missing_modalities,
            "hr_wifi": state.hr_wifi,
            "hr_mm": state.hr_mm,
            "rr_wifi": state.rr_wifi,
            "rr_mm": state.rr_mm,
            "source": "SENSOR_FUSION",
        },

        "fall_context": {
            "fall_detected": fall_detected,
            "missing_assessment_evidence": fall_missing_evidence,
            "source": "NICE_NG249_2025",
        },

        "source_boundaries": {
            "RCP_NEWS2_2017_REFERENCE": (
                "Absolute physiological reference bands only."
            ),
            "NICE_NG249_2025": (
                "Fall-related assessment evidence requirements."
            ),
            "RESIDENT_HISTORY": (
                "Resident-specific longitudinal data."
            ),
            "SENSOR_FUSION": (
                "Measurement reliability evidence, not disease evidence."
            ),
            "PROJECT_POLICY": (
                "Project-specific z-score, recheck and L0-L4 policy."
            ),
        },
    }


def default_clinical_basis(summary: dict) -> list[dict]:
    """Build auditable clinical basis for deterministic/reflex decisions."""
    basis = []

    for metric, result in summary.get("absolute_reference", {}).items():
        if result.get("status") == "ok":
            basis.append({
                "type": "absolute_reference",
                "finding": (
                    f"{metric}={result['value']} "
                    f"falls in {result['band']} reference band"
                ),
                "source": "RCP_NEWS2_2017_REFERENCE",
            })

    if summary.get("personal_baseline"):
        basis.append({
            "type": "personal_baseline",
            "finding": "Resident-specific baseline evidence available",
            "source": "RESIDENT_HISTORY",
        })

    if summary.get("fall_context", {}).get("fall_detected"):
        basis.append({
            "type": "fall_context",
            "finding": "Fall detected; post-fall assessment evidence considered",
            "source": "NICE_NG249_2025",
        })

    return basis
