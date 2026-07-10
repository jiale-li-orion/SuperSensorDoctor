"""Per-modality vital signs synthesizer.
 
When real per-modality sensor data is available (from teammate pipeline),
use it directly. When only fused values are available, synthesize per-modality
estimates by applying modality-specific confidence models.
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
 
    # Synthesize from fused values + confidence scores
    hr = state.heart_rate
    rr = state.respiration_rate
    temp = state.body_temp
    wifi_conf = state.wifi_confidence or 0.5
    mmwave_conf = state.mmwave_confidence or 0.5
    thermal_conf = state.thermal_confidence or 0.5
 
    # Per-modality: each modality gets the same fused value (since we lack
    # raw sensor data to compute independent estimates), but each has its own
    # modality-specific confidence.
    # mmWave confidence is halved when NLOS is active.
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
