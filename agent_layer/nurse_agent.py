"""Nurse Agent — 非 LLM 规则引擎 + 秒级反射弧"""

from datetime import datetime, timedelta
from typing import Optional
from agent_layer.event_bus import EventBus
from agent_layer.state_objects import StateObject, HealthEvent, NurseRuleContext
from agent_layer.baseline_provider import BaselineProvider
from agent_layer.confidence import (
    estimate_wifi_confidence, estimate_mmwave_confidence,
    estimate_thermal_confidence,
)
from agent_layer.tiered_action import ActionLevel
from agent_layer.fusion_engine import FusionEngine
from storage.db import get_db
import uuid


class NurseAgent:
    """规则引擎: 正常状态保持静默, 异常状态发布事件"""

    def __init__(
        self,
        event_bus: EventBus,
        hr_deviation: float = 10.0,
        temp_deviation: float = 1.0,
        observe_duration: int = 300,
        baseline_provider: Optional[BaselineProvider] = None,
        z_threshold: float = 2.0,
        resident_id: str = "resident_01",
    ):
        self.bus = event_bus
        self.hr_deviation = hr_deviation
        self.temp_deviation = temp_deviation
        self.observe_duration = observe_duration
        self.baseline_provider = baseline_provider
        self.z_threshold = z_threshold
        self.resident_id = resident_id
        self._baseline = self._default_baseline()
        # DurationTracker inline: (event_type, resident_id) -> {"start": ts, "last": ts, "elapsed": float}
        self._durations: dict[tuple[str, str], dict] = {}
        self.fusion_engine = FusionEngine()

    def _default_baseline(self) -> dict:
        return {
            "hr_mean": 72, "hr_std": 3,
            "rr_mean": 16, "rr_std": 2,
            "temp_mean": 36.5, "temp_std": 0.3,
        }

    # ── Duration Tracker ──────────────────────────────────────────────────
    def _update_duration(self, event_type: str, resident_id: str, ts: datetime) -> int:
        """Track elapsed seconds since first consecutive occurrence."""
        # Strip timezone to avoid naive-aware comparison errors
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        key = (event_type, resident_id)
        if key not in self._durations:
            self._durations[key] = {"start": ts, "last": ts, "elapsed": 0.0}
            return 0
        entry = self._durations[key]
        delta = (ts - entry["last"]).total_seconds()
        entry["elapsed"] += delta
        entry["last"] = ts
        return int(entry["elapsed"])

    def _reset_duration(self, event_type: str, resident_id: str):
        """Reset duration if anomaly is no longer detected."""
        self._durations.pop((event_type, resident_id), None)

    def _cleanup_stale_durations(self, current_ts: datetime, max_age_sec: int = 3600):
        """Remove entries that haven't been updated in max_age_sec."""
        if current_ts.tzinfo is not None:
            current_ts = current_ts.replace(tzinfo=None)
        stale = [k for k, v in self._durations.items()
                 if (current_ts - v.get("last", current_ts)).total_seconds() > max_age_sec]
        for k in stale:
            self._durations.pop(k, None)

    # ── Main Evaluate ────────────────────────────────────────────────────
    async def evaluate(self, state: StateObject):
        """主入口: 评估一个 StateObject (async — EventBus.publish 是 async)"""
        self._cleanup_stale_durations(state.timestamp)
        events = []
        rule_markers = {"activity_state": state.activity_state}

        # 规则 1: 跌倒检测
        if state.fall_status == "fall":
            phy_change = (
                (state.heart_rate and state.heart_rate > 100) or
                (state.body_temp and state.body_temp > 38.0)
            )
            events.append(self._make_event(
                "fall_detected" if phy_change else "fall_no_physiological_change",
                state, "Fall detected", rule_markers,
            ))

        # ── Build real personal baseline dict (used by NurseRuleContext) ──
        personal_baseline = {}

        # 规则 2: 心率异常 (z_score 优先, fallback 到固定偏差)
        if state.heart_rate:
            z_score_used = None
            deviation = abs(state.heart_rate - self._baseline["hr_mean"])

            if self.baseline_provider:
                try:
                    bl = self.baseline_provider.compute_metric(
                        self.resident_id, "hr", state.heart_rate, state.timestamp
                    )
                    if bl:
                        z_score_used = abs(bl["z_score"])
                        deviation = abs(state.heart_rate - bl["mean"])
                        personal_baseline["hr"] = {
                            "value": state.heart_rate,
                            "mean": bl["mean"],
                            "std": bl["std"],
                            "z_score": bl["z_score"],
                            "source": "RESIDENT_HISTORY",
                        }
                except Exception:
                    pass

            trigger = z_score_used > self.z_threshold if z_score_used is not None else deviation > self.hr_deviation * 2

            if trigger:
                rule_markers["hr_deviation"] = round(deviation, 1)
                if z_score_used is not None:
                    rule_markers["hr_z_score"] = round(z_score_used, 2)
                dur = self._update_duration("hr_abnormal", self.resident_id, state.timestamp)
                if dur > 0:
                    rule_markers["duration_sec"] = dur
                events.append(self._make_event(
                    "hr_abnormal", state,
                    f"HR {state.heart_rate:.0f}, deviation {deviation:.0f} bpm",
                    rule_markers,
                ))
            else:
                self._reset_duration("hr_abnormal", self.resident_id)

        # 规则 3: 体温异常 (z_score 优先, fallback 到固定偏差)
        if state.body_temp:
            z_score_used = None
            dev = abs(state.body_temp - self._baseline["temp_mean"])

            if self.baseline_provider:
                try:
                    bl = self.baseline_provider.compute_metric(
                        self.resident_id, "temp", state.body_temp, state.timestamp
                    )
                    if bl:
                        z_score_used = abs(bl["z_score"])
                        dev = abs(state.body_temp - bl["mean"])
                        personal_baseline["temp"] = {
                            "value": state.body_temp,
                            "mean": bl["mean"],
                            "std": bl["std"],
                            "z_score": bl["z_score"],
                            "source": "RESIDENT_HISTORY",
                        }
                except Exception:
                    pass

            trigger = z_score_used > self.z_threshold if z_score_used is not None else dev > self.temp_deviation

            if trigger:
                rule_markers["temp_deviation"] = round(dev, 1)
                if z_score_used is not None:
                    rule_markers["temp_z_score"] = round(z_score_used, 2)
                dur = self._update_duration("temp_abnormal", self.resident_id, state.timestamp)
                if dur > 0:
                    rule_markers["duration_sec"] = dur
                events.append(self._make_event(
                    "temp_abnormal", state,
                    f"Temp {state.body_temp:.1f}°C, deviation {dev:.1f}°C",
                    rule_markers,
                ))
            else:
                self._reset_duration("temp_abnormal", self.resident_id)

        # 规则 4: RR 呼吸过缓 (bradypnea)
        if state.respiration_rate is not None and state.respiration_rate < 8:
            rule_markers["rr_deviation"] = 8 - state.respiration_rate
            events.append(self._make_event(
                "rr_bradypnea", state,
                f"RR {state.respiration_rate:.0f} < 8, 呼吸过缓",
                rule_markers,
            ))

        # 规则 5: RR 呼吸急促 (tachypnea)
        if state.respiration_rate is not None and state.respiration_rate > 30:
            rule_markers["rr_deviation"] = state.respiration_rate - 30
            events.append(self._make_event(
                "rr_tachypnea", state,
                f"RR {state.respiration_rate:.0f} > 30, 呼吸急促",
                rule_markers,
            ))

        # 规则 5b: RR 个人基线偏离 (z-score, 8 <= RR <= 30)
        if state.respiration_rate is not None and self.baseline_provider:
            try:
                bl = self.baseline_provider.compute_metric(
                    self.resident_id, "rr", state.respiration_rate, state.timestamp
                )
                if bl:
                    rr_z = abs(bl["z_score"])
                    personal_baseline["rr"] = {
                        "value": state.respiration_rate,
                        "mean": bl["mean"],
                        "std": bl["std"],
                        "z_score": bl["z_score"],
                        "source": "RESIDENT_HISTORY",
                    }
                    if (
                        rr_z > self.z_threshold
                        and 8 <= state.respiration_rate <= 30
                    ):
                        rule_markers["rr_z_score"] = round(rr_z, 2)
                        dur = self._update_duration(
                            "rr_baseline_deviation", self.resident_id, state.timestamp
                        )
                        if dur > 0:
                            rule_markers["duration_sec"] = dur
                        events.append(self._make_event(
                            "rr_baseline_deviation", state,
                            f"RR {state.respiration_rate:.0f}, personal baseline {bl['mean']:.1f}±{bl['std']:.1f}, z={bl['z_score']:.2f}",
                            rule_markers,
                        ))
                    else:
                        self._reset_duration("rr_baseline_deviation", self.resident_id)
            except Exception:
                pass

        # 规则 6: 低置信度 (双模态 < 0.5)
        if (state.wifi_confidence is not None and state.mmwave_confidence is not None
                and state.wifi_confidence < 0.5 and state.mmwave_confidence < 0.5):
            rule_markers["wifi_conf"] = state.wifi_confidence
            rule_markers["mmwave_conf"] = state.mmwave_confidence
            events.append(self._make_event(
                "low_confidence", state,
                f"WiFi={state.wifi_confidence:.2f}, mmWave={state.mmwave_confidence:.2f}, 双模态置信度过低",
                rule_markers,
            ))

        # 规则 7: NLOS 遮挡
        if state.nlos_flag is True:
            low_conf_any = (
                (state.wifi_confidence is not None and state.wifi_confidence < 0.3) or
                (state.mmwave_confidence is not None and state.mmwave_confidence < 0.3) or
                (state.thermal_confidence is not None and state.thermal_confidence < 0.3)
            )
            if low_conf_any:
                wifi_str = f"{state.wifi_confidence:.2f}" if state.wifi_confidence is not None else "N/A"
                mmwave_str = f"{state.mmwave_confidence:.2f}" if state.mmwave_confidence is not None else "N/A"
                thermal_str = f"{state.thermal_confidence:.2f}" if state.thermal_confidence is not None else "N/A"
                rule_markers["wifi_conf"] = state.wifi_confidence
                rule_markers["mmwave_conf"] = state.mmwave_confidence
                rule_markers["thermal_conf"] = state.thermal_confidence
                events.append(self._make_event(
                    "nlos_occlusion", state,
                    f"NLOS遮挡, Wi-Fi={wifi_str}, mmWave={mmwave_str}, 红外={thermal_str}",
                    rule_markers,
                ))

        # 规则 8: 模态冲突 — WiFi 与 mmWave 估计差异过大
        if state.heart_rate is not None or state.respiration_rate is not None:
            try:
                for metric in ["hr", "rr"]:
                    result = self.fusion_engine.fuse(state, metric)
                    if result.checks.get("consistent") is False:
                        delta_val = result.checks.get("delta", 0)
                        rule_markers[f"{metric}_modality_delta"] = round(delta_val, 1)
                        rule_markers[f"{metric}_dominant"] = result.verdict.get("dominant_modality", "unknown")
                        rule_markers[f"{metric}_rationale"] = result.verdict.get("rationale", "")
                        events.append(self._make_event(
                            "modality_conflict", state,
                            f"{metric}: WiFi vs mmWave delta={delta_val:.1f}, trusting {result.verdict.get('dominant_modality', 'none')}",
                            rule_markers,
                        ))
            except Exception:
                pass  # fusion is best-effort

        # ── Phase H: Build NurseRuleContext for downstream ──
        ctx_duration = max(
            (v.get("elapsed", 0) for k, v in self._durations.items()
             if k[0] in ("hr_abnormal", "temp_abnormal", "rr_bradypnea", "rr_tachypnea")),
            default=0.0,
        )

        # Query recent sensing windows for context enrichment
        recent_windows = []
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM sensing_windows WHERE resident_id=? ORDER BY timestamp DESC LIMIT 10",
                    (self.resident_id,)
                ).fetchall()
            recent_windows = [dict(r) for r in rows]
        except Exception:
            pass

        nurse_context = NurseRuleContext(
            current_state=state,
            personal_baseline=personal_baseline or None,
            duration_sec=int(ctx_duration),
            recent_windows=recent_windows,
        )

        # Publish events with NurseRuleContext attached
        for event in events:
            event.context = nurse_context
            await self.bus.publish(event)

    def update_baseline(self, new_baseline: dict):
        self._baseline.update(new_baseline)

    def _make_event(self, event_type: str, state: StateObject,
                    reason: str, markers: dict) -> HealthEvent:
        return HealthEvent(
            event_id=f"{event_type}_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            timestamp=state.timestamp,
            state=state,
            trigger_reason=reason,
            rule_markers=dict(markers),
        )
