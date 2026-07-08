# SuperSenseDoctor Agent Layer — 设计规范

## 概要

实现 SuperSenseDoctor 论文中的 MultiAgent Collaboration Layer。三个 Agent（Nurse/Diagnosis/Report）通过 Event Bus 通信，Nurse Agent 做确定性规则引擎 + 秒级反射弧，Diagnosis Agent 做 Think/Act 分离的 ReAct LLM 循环，Report Agent 生成周报和问答。

**核心理念**: 底层算法持续运行输出结构化状态 → 规则引擎提供秒级反射弧 → LLM 仅在事件出现后介入，完成高层证据整合和分级决策。

---

## 1. 项目结构

```
ubicomp/
├── agent_layer/
│   ├── __init__.py
│   ├── state_objects.py       # StateObject, HealthEvent, EpisodeLog dataclass
│   ├── llm_provider.py        # DeepSeek/Ollama API 抽象层
│   ├── event_bus.py           # 进程内 typed pub/sub
│   ├── nurse_agent.py         # 规则引擎 + 反射弧 + 置信度估计
│   ├── confidence.py          # 各模态置信度的规则计算
│   ├── diagnosis_agent.py     # Think/Act ReAct LLM Agent
│   ├── report_agent.py        # 周报 / Q&A 生成
│   ├── tools.py               # @tool 装饰器 + ToolRegistry + 5 个工具
│   └── tiered_action.py       # L0-L4 分级行动策略
├── sensing_simulator/
│   ├── __init__.py
│   ├── sensor_hub.py          # 组装 S_t → SQLite
│   ├── sensor_aligner.py      # 多文件时间窗口对齐
│   └── replay_engine.py       # 回放引擎 (速度/暂停/跳转)
├── storage/
│   ├── __init__.py
│   ├── db.py                  # SQLite 连接 + 建表 (约 50 行)
│   └── models.py              # 数据访问函�数
├── web/
│   ├── __init__.py
│   ├── app.py                 # FastAPI 入口
│   ├── routes.py              # API 路由
│   └── templates/             # Jinja2 HTML 模板
├── config.yaml                # 配置文件
├── main.py                    # 启动入口
├── requirements.txt
└── tests/
    ├── test_nurse_agent.py
    ├── test_diagnosis_agent.py
    ├── test_tools.py
    └── test_event_bus.py
```

---

## 2. 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 语言 | Python 3.12 | 与感知管道同语言 |
| Web | FastAPI + Jinja2 | 轻量, 够展示 demo |
| 存储 | SQLite (WAL 模式) | 零配置边缘部署 |
| LLM | DeepSeek V4 Flash/Pro | OpenAI 兼容 API, 后续切 Ollama |
| Agent 循环 | 裸 ReAct ~50 行 | 不引入 LangChain/LangGraph |
| 依赖 | pydantic, httpx, openpyxl, python-docx | 最小化 |

**为什么不引入 LangChain/LangGraph**: ReAct 循环仅 ~50 行 while 循环, 引入框架反而增加样板代码和版本风险。论文中可展示完整的 Agent 决策伪代码, 比引用框架名更有说服力。

---

## 3. 核心设计

### 3.1 数据模型 (state_objects.py)

三个单向传递的 dataclass: `StateObject → HealthEvent → EpisodeLog`

```python
@dataclass
class StateObject:
    """S_t: 每个时间窗口的统一多维健康状态"""
    window_id: str
    timestamp: datetime

    # 生命体征 (Optional — 不存在的模态为 None)
    heart_rate: Optional[float] = None
    respiration_rate: Optional[float] = None
    body_temp: Optional[float] = None

    # 置信度
    wifi_confidence: Optional[float] = None
    mmwave_confidence: Optional[float] = None
    thermal_confidence: Optional[float] = None

    # 故障标志
    nlos_flag: Optional[bool] = None
    missing_modalities: list[str] = field(default_factory=list)

    # 活动上下文
    activity_state: Optional[str] = None
    posture: Optional[str] = None      # "standing"|"sitting"|"lying"|"falling"
    fall_status: Optional[str] = None  # "fall"|"no_fall" (mmWave简化版)

    # BLE 可穿戴特有
    sensor_contact: Optional[bool] = None

    source: str = "replay"


@dataclass
class HealthEvent:
    """E_t: Nurse Agent 发布的异常事件"""
    event_id: str
    event_type: str
    timestamp: datetime
    state: StateObject
    trigger_reason: str
    rule_markers: dict = field(default_factory=dict)


@dataclass
class EpisodeLog:
    """Diagnosis Agent 产生的可审计事件记录"""
    episode_id: str
    resident_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    evidence: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
```

### 3.2 数据库 (storage/db.py)

3 张表, JSON 列存嵌套结构:

- `sensing_windows` — SensorHub 写入, Nurse Agent 读取
- `health_events` — Nurse Agent 写入, Diagnosis Agent 读取 (含 handled 标志)
- `episode_logs` — Diagnosis Agent 写入, Report Agent 读取 (evidence/decision/action/audit 均为 JSON 列)

### 3.3 Event Bus (event_bus.py)

```python
class EventBus:
    def publish(self, event: HealthEvent): ...
    def subscribe(self, event_type: str, handler: Callable): ...
```

进程内 pub/sub, 松耦合 Agent 间通信。后续可扩展为 Redis/MQTT (但论文 demo 阶段不需要)。

### 3.4 Nurse Agent (nurse_agent.py + confidence.py)

**职责**: 确定性规则引擎, 非 LLM, ms 级响应。

**逻辑**:
```
for each StateObject S_t:
    if within_baseline(S_t):
        continue  # 静默, 不发布事件
    if mild_deviation(S_t):
        observe(S_t)  # 记录, 等待持续时间超过阈值再触发
    if high_risk_combo(S_t):
        publish(L4_emergency_event)  # 直接触发反射弧
    if boundary_violation(S_t):
        publish(event)  # 发布异常事件
```

**检查条件**: 边界越界、持续时间、置信度衰减、NLOS、跨模态偏差、高风险组合(跌倒+心率突变+体温异常)。

**置信度估计 (confidence.py)**: 基于可解释规则的确定性计算:
- 数据存在性 (缺失 → 0)
- 生理合理性 (心率 40-180 bpm → 正常)
- 传感器接触质量 (仅 BLE)
- 短期方差 (相邻窗口突变 → 降分)

### 3.5 Diagnosis Agent (diagnosis_agent.py)

**核心循环 — Think/Act 分离的 ReAct**:

```python
async def handle_event(self, event: HealthEvent) -> EpisodeLog:
    messages = [system_prompt, event_message(event)]

    for step in range(MAX_STEPS):
        # === Think 节点: LLM 无 tools, 强制先分析 ===
        think = await self.llm.chat(messages)  # no tools
        messages.append(think)

        # === 条件边: 判断是否结束 ===
        if (decision := self._parse_decision(think)):
            return self._finalize(decision, messages, event)

        # === Act 节点: LLM 含 tools, 并行执行 ===
        act = await self.llm.chat_with_tools(messages, self.tools.schema())
        results = await asyncio.gather(*[
            self.tools.execute(c) for c in act.tool_calls
        ])
        messages.extend(tool_results(act.tool_calls, results))

    raise MaxStepsExceeded
```

**5 个工具** (对应技术报告 Table 10):
1. `query_history` — 查历史基线和趋势
2. `read_sensing_state` — 读当前多模态估计
3. `consult_fusion` — 三步链式仲裁 (置信度→一致性→仲裁)
4. `write_episode` — 写入可审计事件记录
5. `issue_action` — 执行分级行动

### 3.6 Report Agent (report_agent.py)

**职责**: 周报生成 + 家属自然语言 Q&A。接收 Diagnosis Agent 的 episode logs, 整合趋势摘要, 生成可读报告。

### 3.7 分级行动策略 (tiered_action.py)

| 级别 | 名称 | 触发 | 行动 |
|------|------|------|------|
| L0 | 静默记录 | 基线范围内 | 仅写数据库 |
| L1 | 持续观察 | 轻�度偏离 | 5-10 分钟复查 |
| L2 | 居民提醒 | 轻中度持续 | 本地屏幕/语音提醒 |
| L3 | 家属告警 | 持续 + 多模态一致 | 推送摘要+证据+建议 |
| L4 | 紧急告警 | 跌倒+突变/高烧+异常呼吸 | 立即通知+反射弧 |

### 3.8 LLM Provider (llm_provider.py)

```python
class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages, tools=None) -> ChatResponse: ...
    @abstractmethod
    async def chat_with_tools(self, messages, tool_schemas) -> ChatResponse: ...

class DeepSeekProvider(LLMProvider):   # OpenAI 兼容
class OllamaProvider(LLMProvider):     # 本地模型(后续)
```

温度默认 0 (健康决策需要可复现性), Think 调用不带 tools, Act 调用带 tools。

### 3.9 Sensor Aligner (sensing_simulator/sensor_aligner.py)

**职责**: 将队友提供的零散 CSV/XLSX 文件按时间戳对齐为统一窗口流。

```python
def align_modalities(files: list[Path], window_sec: float = 1.0) -> Iterator[StateObject]:
```

- 读取所有文件, 按 timestamp 排序
- 按 window_sec 分桶
- 桶内 merge 同一窗口的所有模态值
- 缺失模态字段为 None
- 支持最近邻填充 (温度采样率低时)

---

## 4. 调用流程图

```
CSV/XLSX Files (队友提供)
     │
     ▼
Sensor Aligner ──► 统一 StateObject 窗口流
     │
     ▼
SensorHub ──► 写入 SQLite (sensing_windows 表)
     │
     ▼
Nurse Agent (规则引擎) ──► 正常: 静默, 异常: publish HealthEvent
     │
     ▼
Event Bus ──► Diagnosis Agent (ReAct LLM + 5 Tools)
     │                  │
     │                  ├── query_history (查 SQLite)
     │                  ├── read_sensing_state (读状态)
     │                  ├── consult_fusion (三步仲裁)
     │                  ├── write_episode (写 episode_logs)
     │                  └── issue_action (执行分级行动)
     │
     ▼
Report Agent ──► 周报 / Q&A
     │
     ▼
Web UI (FastAPI + Jinja2) ──► 医生工作站 / 家属界面
```

---

## 5. 测试策略

**不依赖队友数据, 自己构造**:

1. construct 正常状态的 StateObject 流 (心率 72±3, 体温 36.5±0.3)
2. construct 异常事件序列:
   - 轻度偏高 (心率 95 持续 10s) → 期望 L1
   - 跌倒 + 心率突变 (80→120) → 期望 L4
   - 持续偏高 + 体温略升 → 期望 L3
   - 单次波动后恢复 → 期望 L0
3. Nurse Agent 测试: falsification (给定构造输入, 验证触发/不触发)
4. Diagnosis Agent 测试: mock LLM 响应, 验证 tool 调用序列正确
5. 端到端测试: 从对齐到 Web 展示全链路

---

## 6. 配置文件 (config.yaml)

```yaml
llm:
  provider: deepseek           # deepseek | ollama | openai
  model: deepseek-chat         # deepseek-chat | deepseek-reasoner
  api_key: ${DEEPSEEK_API_KEY}
  base_url: https://api.deepseek.com
  temperature: 0

storage:
  db_path: data/supersense.db

sensing:
  window_sec: 1.0              # 时间窗口粒度

agents:
  diagnosis:
    max_steps: 8               # ReAct 最大步数
  nurse:
    threshold_hr_deviation: 10 # 心率偏�离阈值(bpm)
    threshold_temp_deviation: 1.0
    observe_duration_sec: 300  # L1�观察持续时间

web:
  host: 0.0.0.0
  port: 8000
```

---

## 7. 限制与假设

- 当前不实现完整 MCP Server / FHIR / OAuth2 (论文 6.1 远景)
- 感知管道结果由 CSV/XLSX replay 模拟, 不接真实硬件
- 单居民模式 (resident_01)
- 无持久化 Agent 状态 (每次重启从 SQLite 恢复)
- LLM 调用为同步阻塞 (不做 streaming)

---

## 8. 待队友确认后微调

- StateObject 字段根据队友实际提供的数据调整
- SensorAligner 的解析逻辑根据文件格式调整
- 置信度规则根据队友管道实际输出的质量信号调整
- 如果队友能提供同步的多模态数据, 简化 Aligner

---

## 附录: 借鉴的框架模式

| 来源 | 模式 | 用在哪里 |
|------|------|---------|
| AutoGen/FinRobot | caller/executor tool 分离 | ToolRegistry.execute |
| CrewAI | typed EventBus | event_bus.py |
| LangGraph | Think/Act 双节点 | diagnosis_agent.py |
| LangGraph | asyncio.gather 并行 tool | tools.py |
| FinRobot | temperature=0 | llm_provider.py |
| Trading agents | Event→Rule→Agent Loop | 整体架构 |

