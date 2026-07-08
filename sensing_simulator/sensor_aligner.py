"""多传感器数据对齐器 — 按时间窗口对齐不同来源的数据"""

from datetime import datetime, timedelta
from typing import Optional

from agent_layer.state_objects import StateObject


class SensorAligner:
    """将来自不同传感器的 CSV/XLSX 行对齐到统一时间窗口"""

    def __init__(self, window_sec: float = 1.0):
        self.window_sec = window_sec

    def align(
        self,
        hr_rows: Optional[list[dict]] = None,
        fall_rows: Optional[list[dict]] = None,
    ) -> list[StateObject]:
        """按时间窗口对齐多源数据, 返回对齐后的 StateObject 列表"""
        windows: dict[str, dict] = {}

        if hr_rows:
            for row in hr_rows:
                ts = row.get("timestamp")
                if not ts:
                    continue
                key = self._window_key(ts)
                windows.setdefault(key, {})
                windows[key]["heart_rate"] = row.get("heart_rate")
                windows[key]["wifi_confidence"] = row.get("wifi_conf")

        if fall_rows:
            for row in fall_rows:
                ts = row.get("timestamp")
                if not ts:
                    continue
                key = self._window_key(ts)
                windows.setdefault(key, {})
                windows[key]["body_temp"] = row.get("body_temp")
                windows[key]["fall_status"] = row.get("fall_status")

        # 转成 StateObject
        result = []
        for key, data in sorted(windows.items()):
            ts = datetime.fromisoformat(key)
            result.append(StateObject(
                window_id=f"aligned_{key}",
                timestamp=ts,
                heart_rate=data.get("heart_rate"),
                body_temp=data.get("body_temp"),
                fall_status=data.get("fall_status"),
                wifi_confidence=data.get("wifi_confidence"),
                source="csv_import",
            ))
        return result

    def _window_key(self, ts: datetime) -> str:
        """将时间戳对齐到 window_sec 粒度的窗口"""
        epoch = ts.timestamp()
        aligned = int(epoch // self.window_sec) * self.window_sec
        return datetime.fromtimestamp(aligned).isoformat()
