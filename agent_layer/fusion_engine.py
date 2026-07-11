"""FusionEngine — 多模态融合仲裁引擎。

Uses 3-step chain arbitration:
  Step 1: Evaluate per-modality confidence
  Step 2: Check cross-modality consistency
  Step 3: Output fused values with rationale

DEPENDS ON: modality_synthesizer, confidence, state_objects (FusionResult)
"""

from datetime import datetime
from typing import Optional

from agent_layer.state_objects import StateObject, FusionResult
from agent_layer.modality_synthesizer import (
    parse_modalities, get_modality_estimate, synthesize_modalities,
)
from agent_layer.confidence import (
    estimate_wifi_confidence, estimate_mmwave_confidence, estimate_thermal_confidence,
)


class FusionEngine:
    """Multi-modal sensor fusion engine.

    Produces a FusionResult per metric (hr, rr, temp) with per-modality
    estimates, consistency checks, and a dominant-modality verdict.
    """

    # Thresholds for consistency checks
    CONSISTENCY_DELTA = {"hr": 5.0, "rr": 3.0, "temp": 0.5}

    def __init__(self, high_confidence_threshold: float = 0.7):
        self.high_confidence_threshold = high_confidence_threshold

    def fuse(self, state: StateObject, metric: str) -> FusionResult:
        """Run 3-step arbitration for a single metric.

        Args:
            state: Current sensing StateObject
            metric: One of "hr", "rr", "temp"

        Returns:
            FusionResult with estimates, checks, and verdict
        """
        # Parse or synthesize per-modality estimates
        raw_mods = parse_modalities(synthesize_modalities(state))
        
        # Extract per-modality estimates
        if metric == "temp":
            wifi_val = get_modality_estimate(raw_mods, "wifi", "temp")
            mmwave_val = get_modality_estimate(raw_mods, "mmwave", "temp")
            thermal_val = get_modality_estimate(raw_mods, "thermal", "temp")
        else:
            wifi_val = get_modality_estimate(raw_mods, "wifi", metric)
            mmwave_val = get_modality_estimate(raw_mods, "mmwave", metric)
            thermal_val = None  # thermal doesn't estimate hr/rr
        
        # Confidence from state or estimates
        if metric == "temp":
            wifi_conf = estimate_wifi_confidence(state) if state.wifi_confidence is None else state.wifi_confidence
            mmwave_conf = estimate_mmwave_confidence(state) if state.mmwave_confidence is None else state.mmwave_confidence
            thermal_conf = estimate_thermal_confidence(state) if state.thermal_confidence is None else state.thermal_confidence
        else:
            wifi_conf = estimate_wifi_confidence(state) if state.wifi_confidence is None else state.wifi_confidence
            mmwave_conf = estimate_mmwave_confidence(state) if state.mmwave_confidence is None else state.mmwave_confidence
            thermal_conf = 0.0  # not applicable

        # Step 1: Confidence evaluation
        wifi_reliable = wifi_conf >= self.high_confidence_threshold
        mmwave_reliable = mmwave_conf >= self.high_confidence_threshold and not state.nlos_flag
        thermal_reliable = thermal_conf >= self.high_confidence_threshold

        estimates = {}
        if wifi_val is not None:
            estimates["wifi"] = {"value": wifi_val, "confidence": round(wifi_conf, 2), "nlos_affected": False}
        if mmwave_val is not None:
            estimates["mmwave"] = {"value": mmwave_val, "confidence": round(mmwave_conf, 2), "nlos_affected": bool(state.nlos_flag)}
        if thermal_val is not None:
            estimates["thermal"] = {"value": thermal_val, "confidence": round(thermal_conf, 2), "nlos_affected": False}

        # Step 2: Consistency check
        mod_values = [v for v in [wifi_val, mmwave_val, thermal_val] if v is not None]
        consistent = True
        delta = 0.0
        if len(mod_values) >= 2:
            delta = max(mod_values) - min(mod_values)
            threshold = self.CONSISTENCY_DELTA.get(metric, 5.0)
            consistent = delta <= threshold

        # Step 3: Arbitration
        fused_value = None
        dominant = "none"
        rationale = "No reliable modality available"

        if state.nlos_flag:
            # NLOS → WiFi dominates (mmWave known-bad)
            fused_value = wifi_val if wifi_val is not None else mmwave_val
            if fused_value is not None:
                dominant = "wifi"
                rationale = "mmWave degraded by NLOS, trusting WiFi"
        elif wifi_reliable and mmwave_reliable and consistent:
            # Both reliable and consistent → confidence-weighted average
            available = [(v, c) for v, c in [(wifi_val, wifi_conf), (mmwave_val, mmwave_conf), (thermal_val, thermal_conf)] if v is not None]
            if available:
                total_weight = sum(c for _, c in available)
                if total_weight > 0:
                    fused_value = round(sum(v * c for v, c in available) / total_weight, 1)
                    dominant = "fusion"
                    rationale = f"Confidence-weighted fusion across {len(available)} modalities"
        elif wifi_reliable and not mmwave_reliable:
            fused_value = wifi_val
            if fused_value is not None:
                dominant = "wifi"
                rationale = "WiFi reliable, mmWave insufficient"
        elif mmwave_reliable and not wifi_reliable:
            fused_value = mmwave_val
            if fused_value is not None:
                dominant = "mmwave"
                rationale = "mmWave reliable, WiFi insufficient"
        else:
            # Both unreliable or conflict → best-effort fallback
            available = [(v, c) for v, c in [(wifi_val, wifi_conf), (mmwave_val, mmwave_conf), (thermal_val, thermal_conf)] if v is not None]
            if available:
                best = max(available, key=lambda x: x[1])
                fused_value = best[0]
                dominant = "best_effort"
                rationale = f"All modalities low-confidence, picked best available"

        return FusionResult(
            metric=metric,
            estimates=estimates,
            checks={
                "delta": round(delta, 2),
                "consistent": consistent,
                "wifi_reliable": wifi_reliable,
                "mmwave_reliable": mmwave_reliable,
                "thermal_reliable": thermal_reliable,
                "nlos_flag": bool(state.nlos_flag),
                "rr_source": state.rr_source,
                "hr_source": state.hr_source,
                "has_per_modality": state.hr_wifi is not None or state.rr_wifi is not None,
            },
            verdict={
                "fused_value": fused_value,
                "dominant_modality": dominant,
                "rationale": rationale,
                "rr_source": state.rr_source,
                "hr_source": state.hr_source,
            },
        )

    def fuse_all(self, state: StateObject) -> dict[str, FusionResult]:
        """Run fusion for all relevant metrics."""
        results = {}
        for metric in ["hr", "rr", "temp"]:
            results[metric] = self.fuse(state, metric)
        return results
