"""分级行动策略 — L0 到 L4"""

from enum import Enum


class ActionLevel(Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


ACTIONS = {
    "L0": {
        "label": "静默记录",
        "channel": "none",
        "message": "状态在个人基线范围内，仅写入数据库。",
        "recheck_after_sec": None,
    },
    "L1": {
        "label": "持续观察",
        "channel": "none",
        "message": "轻度偏离基线，设置定时复查。",
        "recheck_after_sec": 300,
    },
    "L2": {
        "label": "居民提醒",
        "channel": "screen",
        "message": "检测到异常，建议调整姿势或确认状态。",
        "recheck_after_sec": 600,
    },
    "L3": {
        "label": "家属告警",
        "channel": "family_push",
        "message": "持续偏离基线，已通知家属关注。",
        "recheck_after_sec": 1800,
    },
    "L4": {
        "label": "紧急告警",
        "channel": "emergency",
        "message": "⚠️ 紧急情况！请立即联系居民确认状态。",
        "recheck_after_sec": 60,
    },
}


def resolve_action(level: str) -> dict:
    """根据级别返回对应的行动配置"""
    action = ACTIONS.get(level, ACTIONS["L0"]).copy()
    action["level"] = level if level in ACTIONS else "L0"
    return action
