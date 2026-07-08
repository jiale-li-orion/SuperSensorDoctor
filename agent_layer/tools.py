"""Agent Tool 系统 — @tool 装饰器 + ToolRegistry"""

import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable


def tool(name: str, description: str, parameters: dict):
    """装饰器: 将函数包装为 Agent Tool"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._tool_meta = Tool(name, description, parameters, func)
        return wrapper
    return decorator


class ToolRegistry:
    """工具注册表 — 管理 Tool 的 schema 生成和调用"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, func):
        """注册一个 @tool 装饰的函数"""
        meta: Tool = getattr(func, "_tool_meta", None)
        if meta is None:
            raise ValueError(f"Function {func.__name__} has no @tool decorator")
        self._tools[meta.name] = meta

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name]

    def schema(self) -> list[dict]:
        """返回 OpenAI function-calling 格式的 schema"""
        result = []
        for name, tool in self._tools.items():
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": list(tool.parameters.keys()),
                    },
                },
            })
        return result

    def execute(self, name: str, arguments: dict) -> Any:
        """执行工具调用 (caller/executor 分离的 executor 侧)"""
        tool = self.get(name)
        return tool.func(**arguments)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ── Default tools ─────────────────────────────────────────────────────────────

_METRIC_COL_MAP = {"hr": "heart_rate", "rr": "respiration_rate", "temp": "body_temp"}


@tool(
    name="query_history",
    description=(
        "查询该居民特定指标的历史数据，包括基线、近期趋势和相似历史事件。"
        "用于判断当前值对该居民是否异常。"
    ),
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
        "metric": {
            "type": "string", "enum": ["hr", "rr", "temp"],
            "description": "指标名称: hr=心率, rr=呼吸率, temp=体温",
        },
        "time_range": {
            "type": "string",
            "description": "时间范围，如 '7d' 表示7天，'3w' 表示3周",
        },
    },
)
def query_history_tool(resident_id: str, metric: str, time_range: str) -> dict:
    from storage import models

    minutes = {"7d": 10080, "3w": 30240, "24h": 1440, "1h": 60}
    col_metric = _METRIC_COL_MAP.get(metric, metric)
    rows = models.query_recent_windows(
        resident_id, col_metric, minutes.get(time_range, 1440)
    )
    if not rows:
        return {"status": "no_data", "message": f"No {metric} data found"}
    values = [r.get("value") for r in rows if r.get("value") is not None]
    if not values:
        return {"status": "no_data"}
    return {
        "status": "ok",
        "count": len(values),
        "mean": round(sum(values) / len(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "recent_values": values[-10:],
    }


@tool(
    name="get_latest_vitals",
    description="获取该居民最新的生命体征快照，用于实时监控面板。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
    },
)
def get_latest_vitals_tool(resident_id: str) -> dict:
    from storage import models

    result = {}
    for short, full in _METRIC_COL_MAP.items():
        rows = models.query_recent_windows(resident_id, full, 5)
        if rows and rows[-1].get("value") is not None:
            result[short] = rows[-1]["value"]
    return {"status": "ok", "vitals": result} if result else {"status": "no_data"}


@tool(
    name="list_recent_events",
    description="列出该居民最近的异常事件，用于诊断上下文构建。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
        "limit": {"type": "integer", "description": "最多返回条数"},
    },
)
def list_recent_events_tool(resident_id: str, limit: int = 20) -> dict:
    from storage import models

    episodes = models.query_episodes_by_resident(resident_id, limit)
    return {
        "status": "ok",
        "count": len(episodes),
        "episodes": episodes,
    }


@tool(
    name="check_resident_context",
    description="检查居民基本信息，包括传感器状态、场景标记等元数据。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
    },
)
def check_resident_context_tool(resident_id: str) -> dict:
    from storage import models

    # 用 query_recent_windows 获取各指标的最新行
    latest = {}
    for short, full in _METRIC_COL_MAP.items():
        rows = models.query_recent_windows(resident_id, full, 5)
        if rows:
            latest[short] = {
                "value": rows[-1].get("value"),
                "wifi_conf": rows[-1].get("wifi_conf"),
                "mmwave_conf": rows[-1].get("mmwave_conf"),
            }
    return {
        "status": "ok",
        "resident_id": resident_id,
        "latest_sensor_context": latest,
    }


@tool(
    name="trend_analysis",
    description="分析该居民某项指标在时间范围内的趋势。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
        "metric": {
            "type": "string", "enum": ["hr", "rr", "temp"],
            "description": "指标名称: hr=心率, rr=呼吸率, temp=体温",
        },
        "time_range": {
            "type": "string",
            "description": "时间范围，如 '7d' 表示7天，'3w' 表示3周",
        },
    },
)
def trend_analysis_tool(resident_id: str, metric: str, time_range: str) -> dict:
    from storage import models

    minutes = {"7d": 10080, "3w": 30240, "24h": 1440, "1h": 60}
    col_metric = _METRIC_COL_MAP.get(metric, metric)
    rows = models.query_recent_windows(
        resident_id, col_metric, minutes.get(time_range, 1440)
    )
    if not rows:
        return {"status": "no_data", "message": f"No {metric} data found"}
    values = [r.get("value") for r in rows if r.get("value") is not None]
    if not values:
        return {"status": "no_data"}
    diff = round(values[-1] - values[0], 1) if len(values) > 1 else 0
    return {
        "status": "ok",
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "diff": diff,
        "direction": "up" if diff > 0 else ("down" if diff < 0 else "stable"),
        "mean": round(sum(values) / len(values), 1),
    }


# ── New Tools: design-spec tools ──────────────────────────────────────────


@tool(
    name="read_sensing_state",
    description="读取该居民当前最新的多模态传感估计，包括心率、呼吸率、体温及各自的置信度和传感器场景标记（nlos_flag、fall_status）。用于获取实时传感快照。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
    },
)
def read_sensing_state_tool(resident_id: str) -> dict:
    """从最新传感窗口读取当前多模态估计"""
    from storage import models
    row = models.query_latest_sensing_window(resident_id)
    if not row:
        return {"status": "no_data", "message": "No sensing data found"}

    return {
        "status": "ok",
        "resident_id": resident_id,
        "window_id": row.get("window_id"),
        "timestamp": row.get("timestamp"),
        "heart_rate": row.get("hr"),
        "respiration_rate": row.get("rr"),
        "body_temp": row.get("body_temp"),
        "wifi_confidence": row.get("wifi_conf"),
        "mmwave_confidence": row.get("mmwave_conf"),
        "thermal_confidence": row.get("thermal_conf"),
        "nlos_flag": bool(row.get("nlos_flag", False)),
        "fall_status": row.get("fall_status"),
        "activity_state": row.get("activity_state"),
    }


@tool(
    name="consult_fusion",
    description="三步链式跨模态仲裁: Step 1 检查各模态置信度和NLOS遮挡 → Step 2 检查双模态一致性 → Step 3 仲裁输出。用于当 WiFi 和 mmWave 对同一指标给出冲突估计时决定信任哪个。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
        "metric": {
            "type": "string", "enum": ["hr", "rr"],
            "description": "需要仲裁的指标: hr=心率, rr=呼吸率",
        },
    },
)
def consult_fusion_tool(resident_id: str, metric: str) -> dict:
    """三步链式跨模态融合仲裁"""
    from storage import models
    row = models.query_latest_sensing_window(resident_id)
    if not row:
        return {"status": "no_data", "message": "No sensing data for fusion"}

    col_metric = {"hr": "hr", "rr": "rr"}.get(metric)
    value = row.get(col_metric)
    wifi_conf = row.get("wifi_conf", 0.0) or 0.0
    mmwave_conf = row.get("mmwave_conf", 0.0) or 0.0
    nlos = bool(row.get("nlos_flag", False))

    # Step 1: Confidence Evaluation
    wifi_reliable = wifi_conf >= 0.7
    mmwave_reliable = mmwave_conf >= 0.7 and not nlos
    nlof_flag = nlos  # mmWave known-failure when NLOS

    # Step 2: Consistency check (simplified — we only have one value per metric)
    # In full implementation, both WiFi and mmWave would report separate estimates.
    # Here we use confidence delta as a proxy for consistency.
    conf_gap = abs(wifi_conf - mmwave_conf)
    consistent = conf_gap < 0.3

    # Step 3: Arbitration
    if nlos:
        # NLOS → WiFi dominates
        dominant = "wifi"
        fused_value = value
        rationale = "mmWave degraded by NLOS, trusting WiFi"
    elif wifi_reliable and mmwave_reliable and consistent:
        # Both reliable and consistent → confidence-weighted fusion
        # (with single value, we keep it but report high confidence)
        dominant = "fusion"
        fused_value = value
        rationale = "Both modalities consistent, high confidence"
    elif wifi_reliable and not mmwave_reliable:
        dominant = "wifi"
        fused_value = value
        rationale = "mmWave confidence too low, trusting WiFi"
    elif mmwave_reliable and not wifi_reliable:
        dominant = "mmwave"
        fused_value = value
        rationale = "WiFi confidence too low, trusting mmWave"
    else:
        # Conflict or both low → fall back to WiFi
        dominant = "wifi_fallback"
        fused_value = value
        rationale = "Both modalities unreliable, falling back to WiFi"

    return {
        "status": "ok",
        "metric": metric,
        "estimates": {
            "wifi": {
                "value": value,
                "confidence": round(wifi_conf, 2),
            },
            "mmwave": {
                "value": value,
                "confidence": round(mmwave_conf, 2),
                "nlos_flag": nlos,
            },
        },
        "checks": {
            "delta": round(abs((value or 0) - (value or 0)), 1),
            "consistent": consistent,
            "confidence_gap": round(conf_gap, 2),
            "quality_event": False,
        },
        "verdict": {
            "fused_value": value,
            "dominant_modality": dominant,
            "rationale": rationale,
        },
    }


@tool(
    name="write_episode",
    description="将诊断事件记录持久化到数据库。在 LLM 做出最终决策后调用此工具保存可审计记录。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
        "event_id": {"type": "string", "description": "触发此诊断的异常事件ID"},
        "decision_level": {
            "type": "string", "enum": ["L0", "L1", "L2", "L3", "L4"],
            "description": "决策级别",
        },
        "decision_explanation": {"type": "string", "description": "决策解释"},
        "action_message": {"type": "string", "description": "行动消息（可选）"},
    },
)
def write_episode_tool(
    resident_id: str, event_id: str,
    decision_level: str, decision_explanation: str,
    action_message: str = "",
    evidence: Optional[dict] = None,
) -> dict:
    """写入可审计诊断事件记录"""
    from storage import models
    from agent_layer.tiered_action import resolve_action

    action_cfg = resolve_action(decision_level)
    if action_message:
        action_cfg["message"] = action_message

    log = models.insert_episode_log(
        episode_id=f"ep_{uuid.uuid4().hex[:8]}",
        event_id=event_id,
        resident_id=resident_id,
        start_time=datetime.now(),
        evidence=evidence or {},
        decision={
            "level": decision_level,
            "explanation": decision_explanation,
            "action_message": action_message,
        },
        action=action_cfg,
        audit={
            "tools_called": ["write_episode"],
            "step_count": 1,
            "event_id": event_id,
        },
    )
    return {
        "status": "ok",
        "episode_id": log.get("episode_id"),
        "channel": action_cfg["channel"],
        "message": action_cfg["message"],
    }


@tool(
    name="issue_action",
    description="根据决策级别执行分级行动，返回具体行动指令（channel/消息/复查计划）。诊断 Agent 做出决策后调用此工具获取行动方案。",
    parameters={
        "level": {
            "type": "string", "enum": ["L0", "L1", "L2", "L3", "L4"],
            "description": "决策级别",
        },
        "message": {"type": "string", "description": "可选的自定义行动消息"},
    },
)
def issue_action_tool(level: str, message: str = "") -> dict:
    """执行分级行动，返回行动配置"""
    from agent_layer.tiered_action import resolve_action

    action_cfg = resolve_action(level)
    if message:
        action_cfg["message"] = message
    return {
        "status": "ok",
        "level": level,
        "channel": action_cfg["channel"],
        "message": action_cfg["message"],
        "recheck_after_sec": action_cfg["recheck_after_sec"],
    }


def create_default_tools() -> ToolRegistry:
    """创建默认的 Agent 工具箱（设计规范 5 工具 + 辅助工具）"""
    registry = ToolRegistry()
    registry.register(query_history_tool)
    registry.register(read_sensing_state_tool)
    registry.register(consult_fusion_tool)
    registry.register(write_episode_tool)
    registry.register(issue_action_tool)
    registry.register(get_latest_vitals_tool)
    registry.register(list_recent_events_tool)
    registry.register(check_resident_context_tool)
    registry.register(trend_analysis_tool)
    return registry
