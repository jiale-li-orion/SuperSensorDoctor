"""Nurse Agent — 非 LLM 规则引擎 + 秒级反射弧"""

from datetime import datetime, timedelta
from typing import Optional
from agent_layer.event_bus import EventBus
from agent_layer.state_objects import StateObject, HealthEvent
from agent_layer.confidence import (
    estimate_wifi_confidence, estimate_mmwave_confidence,
    estimate_thermal_confidence,
)
from agent_layer.tiered_action import ActionLevel
import uuid


class NurseAgent:
    """规则引擎: 正常状态保持静默, 异常状态发布事件"""

    def __init__(
        self,
        event_bus: EventBus,
        hr_deviation: float = 10.0,
        temp_deviation: float = 1.0,
        observe_duration: int = 300,
    ):
        self.bus = event_bus
        self.hr_deviation = hr_deviation
        self.temp_deviation = temp_deviation
        self.observe_duration = observe_duration
        self._baseline = self._default_baseline()

    def _default_baseline(self) -> dict:
        return {
            "hr_mean": 72, "hr_std": 3,
            "rr_mean": 16, "rr_std": 2,
            "temp_mean": 36.5, "temp_std": 0.3,
        }

    async def evaluate(self, state: StateObject):
        """主入口: 评估一个 StateObject (async — EventBus.publish 是 async)"""
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

        # 规则 2: 心率异常
        if state.heart_rate:
            deviation = abs(state.heart_rate - self._baseline["hr_mean"])
            if deviation > self.hr_deviation * 2:
                rule_markers["hr_deviation"] = deviation
                events.append(self._make_event(
                    "hr_abnormal", state,
                    f"HR {state.heart_rate:.0f}, deviation {deviation:.0f} bpm",
                    rule_markers,
                ))

        # 规则 3: 体温异常
        if state.body_temp:
            dev = abs(state.body_temp - self._baseline["temp_mean"])
            if dev > self.temp_deviation:
                rule_markers["temp_deviation"] = dev
                events.append(self._make_event(
                    "temp_abnormal", state,
                    f"Temp {state.body_temp:.1f}°C, deviation {dev:.1f}°C",
                    rule_markers,
                ))

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

        for event in events:
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
