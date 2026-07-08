"""Agent 层核心数据模型: StateObject → HealthEvent → EpisodeLog"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StateObject:
    """S_t: 单个时间窗口的统一多维健康状态"""
    window_id: str
    timestamp: datetime

    # 生命体征 (Optional — 不存在的模态为 None)
    heart_rate: Optional[float] = None
    respiration_rate: Optional[float] = None
    body_temp: Optional[float] = None

    # 传感器置信度
    wifi_confidence: Optional[float] = None
    mmwave_confidence: Optional[float] = None
    thermal_confidence: Optional[float] = None
    sensor_contact: Optional[bool] = None

    # 场景标记
    nlos_flag: bool = False           # 非视距遮挡
    missing_modalities: list[str] = field(default_factory=list)

    # WiFi 感知 (姿势/活动)
    activity_state: str = "unknown"
    posture: Optional[str] = None

    # 毫米波感知
    fall_status: Optional[str] = None  # "fall" | "no_fall" | None

    # 元数据
    source: str = "replay"             # "replay" | "csv" | "xlsx"


@dataclass
class HealthEvent:
    """E_t: Nurse Agent 检测到异常时发出的事件"""
    event_id: str
    event_type: str           # "hr_abnormal", "temp_abnormal", "fall_detected"
    timestamp: datetime
    state: StateObject
    trigger_reason: str       # 触发原因原文
    rule_markers: dict = field(default_factory=dict)


@dataclass
class EpisodeLog:
    """D_t: Diagnosis Agent 的完整诊断记录"""
    episode_id: str
    resident_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    evidence: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)    # {"level": "L2", "label": "...", ...}
    action: dict = field(default_factory=dict)      # {"channel": "screen", ...}
    audit: dict = field(default_factory=dict)       # {"tools_called": [...], "step_count": 3}
