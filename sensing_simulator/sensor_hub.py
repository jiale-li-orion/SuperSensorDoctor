"""Sensor Hub — 组合 StateObject 并持久化到数据库"""

from datetime import datetime
from typing import Optional

from agent_layer.state_objects import StateObject
from agent_layer.nurse_agent import NurseAgent
from storage.models import insert_sensing_window


class SensorHub:
    """数据汇聚: 接收 StateObject → 持久化 → 触发 NurseAgent"""

    def __init__(self, nurse: Optional[NurseAgent] = None):
        self.nurse = nurse

    async def compose(self, window_id: str, timestamp: datetime, data: dict):
        """组合并处理一个时间窗口的数据"""
        state = self._to_state_object(window_id, timestamp, data)

        # 持久化到 DB
        insert_sensing_window(
            window_id=window_id, timestamp=timestamp,
            heart_rate=data.get("heart_rate"),
            respiration_rate=data.get("respiration_rate"),
            body_temp=data.get("body_temp"),
            wifi_confidence=data.get("wifi_confidence", 1.0),
            mmwave_confidence=data.get("mmwave_confidence", 1.0),
            fall_status=data.get("fall_status"),
            source=data.get("source", "replay"),
        )

        # 通知 NurseAgent
        if self.nurse:
            await self.nurse.evaluate(state)

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
            fall_status=data.get("fall_status"),
            activity_state=data.get("activity_state", "unknown"),
            source=data.get("source", "replay"),
        )
