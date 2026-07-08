"""基于确定性规则的置信度估计器"""

from agent_layer.state_objects import StateObject


def estimate_wifi_confidence(state: StateObject) -> float:
    """估计 WiFi/BLE 模态置信度"""
    score = 1.0

    # 数据存在性
    has_rr = state.respiration_rate is not None
    has_hr = state.heart_rate is not None
    if not (has_rr or has_hr):
        return 0.0

    # 生理合理性
    hr = state.heart_rate
    if hr is not None and not (40 <= hr <= 180):
        score -= 0.5
    rr = state.respiration_rate
    if rr is not None and not (8 <= rr <= 30):
        score -= 0.5

    # 传感器接触质量 (仅 BLE)
    if state.sensor_contact is not None:
        score *= (0.9 if state.sensor_contact else 0.3)

    return max(0.0, min(1.0, score))


def estimate_mmwave_confidence(state: StateObject) -> float:
    """估计 mmWave 模态置信度"""
    if state.nlos_flag:
        return 0.3  # 遮挡时大幅降分

    if state.fall_status is None:
        return 0.5  # 无姿态数据

    return 0.85


def estimate_thermal_confidence(state: StateObject) -> float:
    """估计红外模态置信度"""
    if state.body_temp is None:
        return 0.0
    if not (30 <= state.body_temp <= 42):
        return 0.0
    return 0.9
