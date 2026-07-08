"""Agent Tool 系统 — @tool 装饰器 + ToolRegistry"""

import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable
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


def create_default_tools() -> ToolRegistry:
    """创建默认的 Agent 工具箱"""
    registry = ToolRegistry()
    registry.register(query_history_tool)
    registry.register(get_latest_vitals_tool)
    registry.register(list_recent_events_tool)
    registry.register(check_resident_context_tool)
    registry.register(trend_analysis_tool)
    return registry
