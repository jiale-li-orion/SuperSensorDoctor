"""Agent Tool 系统 — @tool 装饰器 + ToolRegistry"""

import json
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from agent_layer.baseline_provider import BaselineProvider
from agent_layer.fusion_engine import FusionEngine


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
            properties = {
                key: {k: v for k, v in meta.items() if k != "optional"}
                for key, meta in tool.parameters.items()
            }
            required = [
                key for key, meta in tool.parameters.items()
                if not meta.get("optional", False)
            ]
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
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
        "reference_timestamp": {
            "type": "string",
            "description": "可选。事件发生时间 ISO 字符串；历史查询不得读取该时间之后的数据。",
            "optional": True,
        },
    },
)
def query_history_tool(
    resident_id: str,
    metric: str,
    time_range: str,
    reference_timestamp: Optional[str] = None,
) -> dict:
    from storage import models

    minutes = {"7d": 10080, "3w": 30240, "24h": 1440, "1h": 60}
    col_metric = _METRIC_COL_MAP.get(metric, metric)
    ref_dt = datetime.now()
    if reference_timestamp:
        try:
            ref_dt = datetime.fromisoformat(reference_timestamp)
            if ref_dt.tzinfo is not None:
                ref_dt = ref_dt.replace(tzinfo=None)
        except ValueError:
            ref_dt = datetime.now()
    rows = models.query_recent_windows(
        resident_id, col_metric, minutes.get(time_range, 1440),
        reference_timestamp=ref_dt,
    )
    if not rows:
        return {"status": "no_data", "message": f"No {metric} data found"}
    values = [r.get("value") for r in rows if r.get("value") is not None]
    if not values:
        return {"status": "no_data"}

    # Basic stats
    n = len(values)
    mean = round(sum(values) / n, 1)
    result = {
        "status": "ok",
        "count": n,
        "mean": mean,
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "recent_values": values[-10:],
    }

    # Baseline enrichment: z_score and std from BaselineProvider
    try:
        bp = BaselineProvider()
        baseline = bp.compute_metric(
            resident_id=resident_id,
            metric=col_metric,
            value=values[-1],  # most recent value
            at_timestamp=ref_dt,
        )
        if baseline is not None:
            result["z_score"] = baseline.get("z_score")
            result["std"] = baseline.get("std")
    except Exception:
        pass  # baseline enrichment is best-effort

    return result


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
        "posture": row.get("posture"),
        "sensor_contact": bool(row.get("sensor_contact")) if row.get("sensor_contact") is not None else None,
        "missing_modalities": json.loads(row.get("missing_mods", "[]")),
        "hr_wifi": row.get("hr_wifi"),
        "hr_mm": row.get("hr_mm"),
        "rr_wifi": row.get("rr_wifi"),
        "rr_mm": row.get("rr_mm"),
        "hr_conf": row.get("hr_conf"),
        "rr_conf": row.get("rr_conf"),
        "quality_event": int(row.get("quality_event") or 0),
        "hr_source": row.get("hr_source"),
        "rr_source": row.get("rr_source"),
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
    """三步链式跨模态融合仲裁 (delegates to FusionEngine)"""
    from storage import models
    from agent_layer.modality_synthesizer import parse_modalities

    row = models.query_latest_sensing_window(resident_id)
    if not row:
        return {"status": "no_data", "message": "No sensing data for fusion"}

    # Build StateObject from DB row
    from datetime import datetime
    from agent_layer.state_objects import StateObject

    state = StateObject(
        window_id=row.get("window_id", ""),
        timestamp=datetime.fromisoformat(row["timestamp"]) if isinstance(row.get("timestamp"), str) else datetime.now(),
        heart_rate=row.get("hr"),
        respiration_rate=row.get("rr"),
        body_temp=row.get("body_temp"),
        wifi_confidence=row.get("wifi_conf"),
        mmwave_confidence=row.get("mmwave_conf"),
        thermal_confidence=row.get("thermal_conf"),
        nlos_flag=bool(row.get("nlos_flag", False)),
        fall_status=row.get("fall_status"),
        activity_state=row.get("activity_state", "unknown"),
        posture=row.get("posture"),
        sensor_contact=row.get("sensor_contact"),
        missing_modalities=json.loads(row.get("missing_mods", "[]")),  # list from JSON string
        hr_wifi=row.get("hr_wifi"),
        hr_mm=row.get("hr_mm"),
        rr_wifi=row.get("rr_wifi"),
        rr_mm=row.get("rr_mm"),
        hr_conf=row.get("hr_conf"),
        rr_conf=row.get("rr_conf"),
        quality_event=int(row.get("quality_event") or 0),
        hr_source=row.get("hr_source"),
        rr_source=row.get("rr_source"),
    )

    # Delegate to FusionEngine
    engine = FusionEngine()
    result = engine.fuse(state, metric)

    # Detect if we had real per-modality data
    mods = parse_modalities(row.get("modalities_json"))
    has_pm = any(row.get(k) is not None for k in ("hr_wifi", "hr_mm", "rr_wifi", "rr_mm"))
    data_mode = "per_modality" if (mods or has_pm) else "fused_value_proxy"

    return {
        "status": "ok",
        "metric": metric,
        "estimates": result.estimates,
        "checks": {
            **result.checks,
            "confidence_gap": round(
                abs((row.get("wifi_conf", 0.0) or 0.0) - (row.get("mmwave_conf", 0.0) or 0.0)), 2
            ),
            "quality_event": result.checks.get("quality_event", False),
            "data_mode": data_mode,
        },
        "verdict": result.verdict,
    }


# NOTE: write_episode is NOT registered in create_default_tools().
# It is only called directly by DiagnosisAgent (orchestrator-only).
# The @tool decorator is used for tool metadata/schema, but the LLM never sees it.
@tool(
    name="write_episode",
    description="[Orchestrator Internal] 将诊断事件记录持久化到数据库。不对外暴露。",
    parameters={
        "resident_id": {"type": "string", "description": "居民ID"},
        "event_id": {"type": "string", "description": "触发此诊断的异常事件ID"},
    },
)
def write_episode_tool(
    resident_id: str,
    event_id: str,
    decision: Optional[dict] = None,
    *,
    evidence: Optional[dict] = None,
    tools_called: Optional[list[str]] = None,
    step_count: int = 1,
    decision_level: str = "L0",        # kept for backward compat
    decision_explanation: str = "",
    action_message: str = "",
) -> dict:
    """写入可审计诊断事件记录"""
    from storage import models
    from agent_layer.tiered_action import resolve_action

    # decision dict takes priority, fallback to old scalar params
    if decision is not None:
        level = decision.get("level", "L0")
        label = decision.get("label", "")
        event_interpretation = decision.get("event_interpretation", "")
        evidence_used = decision.get("evidence_used", [])
        clinical_basis = decision.get("clinical_basis", [])
        uncertainty = decision.get("uncertainty", {})
        action = decision.get("action", {})
        safety_boundary = decision.get("safety_boundary", "care_support_only")
    else:
        # Backward compat: construct from old scalar params
        level = decision_level
        label = ""
        event_interpretation = decision_explanation
        evidence_used = []
        clinical_basis = []
        uncertainty = {"sensing_quality": "unknown", "missing_evidence": [], "needs_recheck": True}
        action = {}
        safety_boundary = "care_support_only"

    action_cfg = resolve_action(level)
    if action_message:
        action_cfg["message"] = action_message

    log = models.insert_episode_log(
        episode_id=f"ep_{uuid.uuid4().hex[:8]}",
        event_id=event_id,
        resident_id=resident_id,
        start_time=datetime.now(),
        evidence=evidence or {},
        decision={
            "level": level,
            "label": label,
            "event_interpretation": event_interpretation,
            "evidence_used": evidence_used,
            "clinical_basis": clinical_basis,
            "uncertainty": uncertainty,
            "action": action,
            "safety_boundary": safety_boundary,
        },
        action=action_cfg,
        audit={
            "tools_called": tools_called or ["write_episode"],
            "step_count": step_count,
            "event_id": event_id,
        },
    )
    return {
        "status": "ok",
        "episode_id": log.get("episode_id"),
        "level": level,
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
    registry.register(get_latest_vitals_tool)
    registry.register(list_recent_events_tool)
    registry.register(check_resident_context_tool)
    registry.register(trend_analysis_tool)
    return registry
