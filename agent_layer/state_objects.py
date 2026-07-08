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


@dataclass
class TriageDecision:
    """Agent 输出协议 — 固定 schema 使决策可测、可审计。
    
    level:       L0-L4 必填 (严格校验)
    label:       对应级别的人类可读标签
    event_interpretation: 当前事件的医学解读
    evidence_used:  本次诊断引用的工具名列表
    uncertainty:    不确定性声明 (可选, missing 时用默认值)
    action:         推荐行动方案 (可选)
    safety_boundary: 安全边界声明 (固定为 care_support_only)
    
    NOTE: uncertainty 是 LLM 自评估, 暂不经过 ground truth 校验。
    """
    level: str
    label: str = ""
    event_interpretation: str = ""
    evidence_used: list[str] = field(default_factory=list)
    uncertainty: dict = field(default_factory=lambda: {
        "sensing_quality": "unknown",
        "missing_evidence": [],
        "needs_recheck": True,
    })
    action: dict = field(default_factory=dict)
    safety_boundary: str = "care_support_only"


@dataclass
class FusionResult:
    """三步链式跨模态仲裁的标准证据对象。
    
    estimates: 各模态的独立估计值 (Phase 6 前使用同一融合值)
    checks:    置信度一致性检查结果
    verdict:   最终的仲裁输出 (模态选择 + 融合值 + 理由)
    
    NOTE: Until Phase 6 provides per-modality sensor data, the same fused
    value is used for all modalities in estimates — only confidence differs.
    """
    metric: str
    estimates: dict
    checks: dict
    verdict: dict
