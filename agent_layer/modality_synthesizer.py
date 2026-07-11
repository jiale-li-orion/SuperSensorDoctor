"""Per-modality vital signs synthesizer.
 
When real per-modality sensor data is available (from portable_v2 pipeline
via StateObject.hr_wifi/rr_wifi etc.), use it directly per-modality.
When only fused values are available, synthesize per-modality estimates
by applying modality-specific confidence models.
"""

import json
from typing import Optional
from agent_layer.state_objects import StateObject


def synthesize_modalities(state: StateObject) -> str:
    """Generate per-modality estimates JSON string from a StateObject.
 
    Returns a JSON string like:
    {"wifi": {"hr": 72.0, "rr": 16.0, "confidence": 0.85, "nlos_affected": false},
     "mmwave": {"hr": 71.0, "rr": 15.5, "confidence": 0.72, "nlos_affected": true},
     "thermal": {"temp": 36.5, "confidence": 0.9, "nlos_affected": false}}
    """
    # If real per-modality data is available (from teammate pipeline), use it
    if state.missing_modalities:
        if isinstance(state.missing_modalities, str):
            try:
                parsed = json.loads(state.missing_modalities)
                if isinstance(parsed, dict) and ("wifi" in parsed or "mmwave" in parsed):
                    return state.missing_modalities
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(state.missing_modalities, dict):
            if "wifi" in state.missing_modalities or "mmwave" in state.missing_modalities:
                return json.dumps(state.missing_modalities)
 
    # ── Per-modality estimates ──
    hr = state.heart_rate
    rr = state.respiration_rate
    temp = state.body_temp
    wifi_conf = state.wifi_confidence or 0.5
    mmwave_conf = state.mmwave_confidence or 0.5
    thermal_conf = state.thermal_confidence or 0.5

    # When real per-modality data exists (from portable_v2 pipeline), use it.
    # Each modality gets its own independent estimate with per-metric confidence.
    # This enables meaningful cross-modality consistency checks in FusionEngine.
    has_per_modality = state.hr_wifi is not None or state.rr_wifi is not None

    if has_per_modality:
        # Use real per-modality values + combine hardware and signal confidence
        wifi_hr_est = state.hr_wifi if state.hr_wifi is not None else hr
        wifi_rr_est = state.rr_wifi if state.rr_wifi is not None else rr
        mm_hr_est = state.hr_mm if state.hr_mm is not None else hr
        mm_rr_est = state.rr_mm if state.rr_mm is not None else rr

        # Combine hardware reliability (wifi_conf) with per-metric signal confidence
        wifi_conf_used = min(wifi_conf, (state.hr_conf or state.rr_conf or 1.0), 0.95)
        mm_conf_used = min(mmwave_conf * (0.5 if state.nlos_flag else 1.0),
                           (state.hr_conf or state.rr_conf or 1.0), 0.95)

        modalities = {
            "wifi": {
                "hr": wifi_hr_est,
                "rr": wifi_rr_est,
                "confidence": round(wifi_conf_used, 2),
                "nlos_affected": False,
            },
            "mmwave": {
                "hr": mm_hr_est,
                "rr": mm_rr_est,
                "confidence": round(mm_conf_used, 2),
                "nlos_affected": bool(state.nlos_flag),
            },
            "thermal": {
                "temp": temp,
                "confidence": min(thermal_conf, 0.95),
                "nlos_affected": False,
            },
        }
    else:
        # Fallback: fused value duplicated across modalities (original behavior)
        modalities = {
            "wifi": {
                "hr": hr,
                "rr": rr,
                "confidence": min(wifi_conf, 0.95),
                "nlos_affected": False,
            },
            "mmwave": {
                "hr": hr,
                "rr": rr,
                "confidence": min(mmwave_conf * (0.5 if state.nlos_flag else 1.0), 0.95),
                "nlos_affected": bool(state.nlos_flag),
            },
            "thermal": {
                "temp": temp,
                "confidence": min(thermal_conf, 0.95),
                "nlos_affected": False,
            },
        }
    return json.dumps(modalities)


def parse_modalities(modalities_json: Optional[str]) -> dict:
    """Safely parse modalities JSON string. Returns empty dict on failure."""
    if not modalities_json:
        return {}
    try:
        return json.loads(modalities_json) if isinstance(modalities_json, str) else modalities_json
    except (json.JSONDecodeError, TypeError):
        return {}


def get_modality_estimate(modalities: dict, modality: str, metric: str) -> Optional[float]:
    """Extract a specific metric from a specific modality.
 
    Args:
        modalities: Parsed modalities dict from parse_modalities()
        modality: "wifi", "mmwave", or "thermal"
        metric: "hr", "rr", or "temp"
    Returns:
        The value or None if not available.
    """
    mod = modalities.get(modality, {})
    return mod.get(metric)
