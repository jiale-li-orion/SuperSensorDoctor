"""Sensor Hub — 组合 StateObject 并持久化到数据库"""

from datetime import datetime
from typing import Optional

from agent_layer.state_objects import StateObject
from agent_layer.nurse_agent import NurseAgent
from agent_layer.modality_synthesizer import synthesize_modalities
from storage.models import insert_sensing_window


class SensorHub:
    """数据汇聚: 接收 StateObject → 持久化 → 触发 NurseAgent"""

    def __init__(self, nurse: Optional[NurseAgent] = None):
        self.nurse = nurse

    async def compose(self, window_id: str, timestamp: datetime, data: dict):
        """组合并处理一个时间窗口的数据"""
        state = self._to_state_object(window_id, timestamp, data)

        # ── Per-modality synthesis ──
        modalities_json = synthesize_modalities(state)

        # 持久化到 DB
        result = insert_sensing_window(
            window_id=window_id, timestamp=timestamp,
            heart_rate=data.get("heart_rate"),
            respiration_rate=data.get("respiration_rate"),
            body_temp=data.get("body_temp"),
            wifi_confidence=data.get("wifi_confidence", 1.0),
            mmwave_confidence=data.get("mmwave_confidence", 1.0),
            thermal_confidence=data.get("thermal_confidence"),
            nlos_flag=data.get("nlos_flag", False),
            activity_state=data.get("activity_state", "unknown"),
            fall_status=data.get("fall_status"),
            sensor_contact=data.get("sensor_contact"),
            posture=data.get("posture"),
            missing_modalities=data.get("missing_modalities"),
            modalities_json=modalities_json,
            source=data.get("source", "replay"),
            # ── Per-modality portable_v2 fields ──
            rr_wifi=data.get("rr_wifi"),
            rr_mm=data.get("rr_mm"),
            hr_wifi=data.get("hr_wifi"),
            hr_mm=data.get("hr_mm"),
            rr_conf=data.get("rr_conf"),
            hr_conf=data.get("hr_conf"),
            quality_event=data.get("quality_event", False),
            rr_source=data.get("rr_source"),
            hr_source=data.get("hr_source"),
            rr_truth=data.get("rr_truth"),
            hr_truth=data.get("hr_truth"),
        )

        # 通知 NurseAgent
        if self.nurse:
            await self.nurse.evaluate(state)

        return result

    def _to_state_object(self, window_id: str, timestamp: datetime,
                         data: dict) -> StateObject:
        return StateObject(
            window_id=window_id,
            timestamp=timestamp,
            heart_rate=data.get("heart_rate"),
            respiration_rate=data.get("respiration_rate"),
            body_temp=data.get("body_temp"),
            wifi_confidence=data.get("wifi_confidence"),
            mmwave_confidence=data.get("mmwave_confidence"),
            thermal_confidence=data.get("thermal_confidence"),
            nlos_flag=data.get("nlos_flag", False),
            fall_status=data.get("fall_status"),
            activity_state=data.get("activity_state", "unknown"),
            sensor_contact=data.get("sensor_contact"),
            posture=data.get("posture"),
            missing_modalities=data.get("missing_modalities"),
            source=data.get("source", "replay"),
            hr_wifi=data.get("hr_wifi"),
            hr_mm=data.get("hr_mm"),
            rr_wifi=data.get("rr_wifi"),
            rr_mm=data.get("rr_mm"),
            rr_conf=data.get("rr_conf"),
            hr_conf=data.get("hr_conf"),
            quality_event=data.get("quality_event", 0),
            rr_source=data.get("rr_source"),
            hr_source=data.get("hr_source"),
            rr_truth=data.get("rr_truth"),
            hr_truth=data.get("hr_truth"),
        )
