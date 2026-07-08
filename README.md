# SuperSenseDoctor — MultiAgent Collaboration Layer

Ubicomp 论文投稿项目。多模态无接触健康监测系统的 Agent 协作层实现。

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│ SensorAligner│───>│   SensorHub      │───>│   NurseAgent   │
│  (多源对齐)   │    │  (数据汇聚+持久化)│    │  (规则引擎)     │
└─────────────┘    └──────────────────┘    └───────┬────────┘
                                                   │ EventBus
                                                   ▼
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│ ReportAgent │    │  DiagnosisAgent  │<───│  HealthEvent   │
│ (周报/Q&A)   │    │  (ReAct Think/Act)│   │  (异常事件)     │
└─────────────┘    └────────┬─────────┘    └────────────────┘
                            │
                     ┌──────┴──────┐
                     │  ToolRegistry│
                     │  (5 tools)   │
                     └─────────────┘
```

## Tech Stack

- **Python 3.12** — 裸 ReAct 循环 (无 LangChain/LangGraph)
- **FastAPI + Jinja2** — Web 医生工作站
- **SQLite (WAL 模式)** — 持久化存储
- **DeepSeek V4 API** — LLM Provider (OpenAI 兼容)
- **pytest + pytest-asyncio** — 测试 (41 tests)

## Project Structure

```
agent_layer/              # Agent 核心层
├── state_objects.py      # StateObject / HealthEvent / EpisodeLog
├── event_bus.py          # 异步 pub/sub 事件总线 (通配符支持)
├── confidence.py         # 三模态确定性置信度估计
├── nurse_agent.py        # 规则引擎 (跌倒/心率/体温检测)
├── diagnosis_agent.py    # ReAct Think/Act 循环 (8步 max)
├── tools.py              # @tool 装饰器 + ToolRegistry + 5 工具
├── tiered_action.py      # L0-L4 分级行动策略
├── llm_provider.py       # DeepSeekProvider + MockProvider
└── report_agent.py       # 周报生成 + 关键词 Q&A

sensing_simulator/        # 感知模拟器
├── sensor_aligner.py     # 多文件时间窗对齐
├── sensor_hub.py         # 数据汇聚 + DB 持久化 + Nurse 触发
└── replay_engine.py      # 合成数据生成器

storage/                  # 持久化层
├── db.py                 # SQLite 连接管理 + 建表
└── models.py             # 数据访问层 (3 表 CRUD)

web/                      # Web UI
├── app.py                # FastAPI 入口 (3 路由)
└── templates/dashboard.html

tests/                    # 测试 (41 用例)
```

## Data Flow

1. **SensorHub.compose()** → 接收 StateObject → 写入 SQLite `sensing_windows`
2. **NurseAgent.evaluate()** async → 规则检测 (跌倒/心率/体温) → 发布 HealthEvent 到 EventBus
3. **DiagnosisAgent.handle_event()** async → ReAct 循环: Think(LLM) → Act(Tool) → Decide(JSON)
4. **TieredAction.resolve_action()** → L0(静默) ~ L4(紧急) → 写入 `episode_logs`
5. **ReportAgent** → 按 EpisodeLog 统计周报 + 关键词 Q&A

## Tiered Action Levels

| Level | Channel | Description |
|-------|---------|-------------|
| L0 | none | 静默记录，正常范围 |
| L1 | none | 轻度偏离，持续观察 |
| L2 | screen | 异常，提醒居民 |
| L3 | family_push | 持续异常，通知家属 |
| L4 | emergency | 紧急，立即通知 |

## Getting Started

```bash
pip install -r requirements.txt

# 配置环境变量
export DEEPSEEK_API_KEY=your_key_here

# 运行全量测试
make test

# 启动 Web 服务
make run
# http://localhost:8000
```

## Key Design Decisions

- **零框架 Agent**: 裸 ReAct 循环，无 LangChain/LangGraph 依赖
- **确定性置信度**: 生理合理性 + 数据存在性 + 传感器接触质量，不做 ML
- **异步贯穿**: NurseAgent → EventBus → DiagnosisAgent 全异步链
- **@tool 装饰器**: AutoGen 启发的 caller/executor 分离
- **SQLite WAL**: 单居民场景 (resident_01)，配置驱动
