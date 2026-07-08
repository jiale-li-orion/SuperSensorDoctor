# SuperSenseDoctor — MultiAgent Collaboration Layer

SuperSenseDoctor is a multimodal and contactless intelligent health-guarding system for home-based elderly care. On a home edge device, the system uses **WiFi Beamforming Feedback Information (BFI)**, **mmWave radar**, and **an infrared thermal array** to continuously track respiration, heart rate, body temperature, posture, and fall events — without requiring the elderly person to wear or operate any device.

This repository implements the **MultiAgent Collaboration Layer**: the orchestration layer that transforms continuous low-level sensing estimates into auditable, actionable long-term health tracking decisions.

## Paper

The full technical report covers the complete end-to-end system. This repo focuses on the Agent layer (Section 4 of the report).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Multimodal Sensing Layer                 │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │ WiFi BFI   │  │ mmWave     │  │ IR Thermal Array │   │
│  │ (呼吸/心率) │  │ (呼吸/跌倒) │  │ (体温 32×24)     │   │
│  └─────┬──────┘  └──────┬─────┘  └────────┬─────────┘   │
│        │                │                  │             │
│        └────────────────┴──────────────────┘             │
│                         │ SensorAligner                   │
│                         ▼                                │
│                 ┌───────────────┐                        │
│                 │  SensorHub    │                        │
│                 │ (StateObject) │──→ SQLite (sensing)    │
│                 └───────┬───────┘                        │
└─────────────────────────┼────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              MultiAgent Collaboration Layer              │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              EventBus (pub/sub)                   │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────┐  │   │
│  │  │ NurseAgent  │  │DiagnosisAgent│  │Report   │  │   │
│  │  │ (规则引擎)   │  │(ReAct Think  │  │Agent    │  │   │
│  │  │             │  │ /Act/Decide) │  │(周报/Q&A)│  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └─────────┘  │   │
│  │         │                │                        │   │
│  │         ▼                ▼                        │   │
│  │  HealthEvent       ToolRegistry                   │   │
│  │  (异常事件)       (5 tools)                        │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│                 SQLite (episode_logs)                     │
│                                                          │
└─────────────────────────┼────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│            Output Layer (Human-Computer Interaction)     │
│  ┌──────────────────────────────────────────────────┐   │
│  │     FastAPI Web → Doctor Workstation Dashboard    │   │
│  │     Report Agent → Weekly Summary / Q&A            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## End-to-End Data Flow

1. **WiFi BFI Pipeline** — Raw BFM matrix stream from OpenWRT Sniffer via SSH pipe → BFM-ratio normalization → dual-peak tracking for respiration/heart-rate estimation.
2. **mmWave Pipeline** — Point-cloud frames (x, y, z, v, I) from TI AWR1843 → vital signs extraction + fall detection via height/velocity thresholding.
3. **IR Thermal Pipeline** — MLX90640 32×24 array → 8×8 zonal reduction → peak body-surface temperature.
4. **SensorAligner** — Aligns all three pipelines to the same time window (configurable, default 1s) → produces unified `StateObject`.
5. **SensorHub** — Assembles `StateObject` → writes to SQLite `sensing_windows` → forwards to NurseAgent.
6. **NurseAgent** — Deterministic rule engine. Silent for normal states; publishes `HealthEvent` for boundary violations, falls, or cross-modal anomalies.
7. **DiagnosisAgent** — Receives event → Think (LLM without tools) / Act (call ToolRegistry tools) / Decide (output JSON `{"level": "L2", "explanation": "..."}`) ReAct loop. Max 8 steps, falls back to L0.
8. **TieredAction** — Maps decision level to channel: L0(none) → L1(none) → L2(screen) → L3(family_push) → L4(emergency).
9. **ReportAgent** — Generates weekly summaries from `EpisodeLog`; answers natural-language questions via keyword matching.

---

## Cross-Modal Fusion Arbitration

When the Diagnosis Agent needs to resolve conflicting estimates across modalities, it uses a 3-step chain (implemented as callable tools):

| Step | Name | Logic |
|------|------|-------|
| 1 | Confidence Evaluation | Check each modality's confidence score; check `nlos_flag` (mmWave known-failure indicator) |
| 2 | Consistency Check | Compute cross-modal delta: Δf = |f_wifi − f_mm|; check historical trend deviation |
| 3 | Arbitration | NLOS → WiFi dominates; both reliable → confidence-weighted fusion; conflict → trust the reliable branch |

**Arbitration priority**: fall alert > NLOS switching > confidence arbitration > semantic merging.

---

## Agent System Design

### Nurse Agent (Non-LLM Rule Engine)

- **Principle**: Remain silent at normal states; publish event for anomalies; trigger reflex arc for high-risk combinations (e.g., fall + elevated HR).
- **Rules**: Fall detection, heart-rate deviation (>2σ), temperature deviation (>threshold), cross-modal confidence degradation.
- **Reflex arc**: Certain high-risk patterns bypass LLM deliberation and publish L3/L4 events directly.

### Diagnosis Agent (ReAct Think/Act Loop)

- **Think phase**: Calls LLM without tools → analyzes event + resident baseline → attempts to output JSON decision.
- **Act phase**: If no decision parsed, calls LLM with tools → executes `tool_call` via ToolRegistry → appends result to message history → repeats.
- **Tool system**: 5 registered tools — `query_history`, `get_latest_vitals`, `list_recent_events`, `check_resident_context`, `trend_analysis`.
- **Fallback**: Returns L0 (silent record) if `max_steps` exceeded without parseable decision.

### Report Agent

- Generates weekly reports by aggregating `EpisodeLog` records (last 7 days, severity counts).
- Answers natural-language questions via keyword match (发烧/心率/跌倒/血压). *Stub — replaceable with LLM in production.*

### Tiered Action Strategy

| Level | Label | Channel | Message | Recheck |
|-------|-------|---------|---------|---------|
| L0 | 静默记录 | none | 正常范围，仅写数据库 | — |
| L1 | 持续观察 | none | 轻度偏离，定时复查 | 300s |
| L2 | 居民提醒 | screen | 建议调整姿势/确认状态 | 600s |
| L3 | 家属告警 | family_push | 持续异常，已通知家属 | 1800s |
| L4 | 紧急告警 | emergency | 立即联系居民确认状态 | 60s |

### Auditable Episode Record

Every Diagnosis Agent interaction is recorded as an `EpisodeLog` containing:

- `decision`: `{"level": "L2", "explanation": "...", "action_message": "..."}`
- `action`: `{"channel": "screen", "message": "...", "recheck_after": 600}`
- `evidence`: `{"sensing_summary": {"heart_rate": 120, ...}}`
- `audit`: `{"tools_called": ["query_history"], "step_count": 3, "event_id": "..."}`

The LLM's private reasoning chain is not stored; only the auditable summary is persisted.

---

## Project Structure

```
ubicomp/
├── agent_layer/                # Agent 核心层
│   ├── state_objects.py        # StateObject / HealthEvent / EpisodeLog (dataclass)
│   ├── event_bus.py            # 异步 pub/sub 事件总线 (通配符 fnmatch)
│   ├── confidence.py           # 三模态确定性置信度估计 (WiFi/mmWave/IR)
│   ├── nurse_agent.py          # 规则引擎 (跌倒/心率/体温检测, async evaluate)
│   ├── diagnosis_agent.py      # ReAct Think/Act/Decide 循环 (max 8 steps)
│   ├── tools.py                # @tool 装饰器 + ToolRegistry + 5 default tools
│   ├── tiered_action.py        # L0-L4 分级行动策略
│   ├── llm_provider.py         # DeepSeekProvider + MockProvider (OpenAI 兼容)
│   └── report_agent.py         # 周报生成 + 关键词 Q&A
├── sensing_simulator/          # 感知模拟器 (论文演示用)
│   ├── sensor_aligner.py       # 多文件时间窗对齐 (HR + fall + temp)
│   ├── sensor_hub.py           # 数据汇聚 + SQLite 持久化 + NurseAgent 触发
│   └── replay_engine.py        # 合成数据生成器 (可注入异常模式)
├── storage/                    # 持久化层
│   ├── db.py                   # SQLite WAL 模式连接 + 3 表 schema
│   └── models.py               # 数据访问层 (CRUD: sensing_windows/health_events/episode_logs)
├── web/                        # Web UI
│   ├── app.py                  # FastAPI 入口 (3 routes: /, /api/replay/start, /api/health)
│   └── templates/dashboard.html # 医生工作站 Jinja2 模板
├── tests/                      # 测试 (41 cases, pytest + pytest-asyncio)
│   ├── test_state_objects.py   # 4 tests
│   ├── test_db.py              # 3 tests
│   ├── test_event_bus.py       # 4 tests
│   ├── test_nurse_agent.py     # 9 tests
│   ├── test_tools.py           # 4 tests
│   ├── test_diagnosis_agent.py # 4 tests
│   ├── test_report_agent.py    # 3 tests
│   ├── test_tiered_action.py   # 3 tests
│   ├── test_llm_provider.py    # 1 test
│   └── test_replay.py          # 6 tests
├── main.py                     # 集成入口 (init_db → EventBus → Agents → FastAPI)
├── config.yaml                 # 配置 (LLM/DB/Nurse/Diagnosis/Web)
├── requirements.txt
├── Makefile                    # install / run / test / clean
└── README.md
```

---

## Tech Stack

- **Python 3.12** — Bare ReAct loop, no LangChain/LangGraph dependency
- **FastAPI + Jinja2** — Web doctor workstation
- **SQLite (WAL mode)** — Local persistent storage with PRAGMA foreign_keys
- **DeepSeek V4 API** — LLM Provider via OpenAI-compatible HTTP endpoint
- **pytest + pytest-asyncio** — 41 tests across 10 test files

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No LangChain/LangGraph** | Raw ReAct loop with `@tool` decorator + ToolRegistry. Full control over prompt structure and tool dispatch. |
| **Async event chain** | NurseAgent.evaluate() → EventBus.publish() → DiagnosisAgent.handle_event() is fully async. Both sync and async subscribers supported via `iscoroutinefunction` detection. |
| **Deterministic confidence** | Three heuristics: data existence + physiological plausibility + sensor contact quality. No ML model needed — confidence is an explainable signal, not a black box. |
| **SQLite JSON columns** | `evidence`, `decision`, `action`, `audit` stored as TEXT (JSON). Parsed to dict at the web layer before template rendering. |
| **Single-resident mode** | `resident_id = "resident_01"` for paper demo. Schema supports multi-resident via indexed columns. |
| **Deterministic replay** | Seeded `random.Random(42)` for reproducible synthetic data generation with configurable anomaly injection windows. |

## Getting Started

```bash
# Install dependencies
make install

# Set your API key
export DEEPSEEK_API_KEY=sk-your-key-here

# Run all tests (requires no API key — mock provider used)
make test

# Start the doctor workstation
make run
# → http://localhost:8000
```

### Test Suite

```
pytest tests/ -v
# 41 passed in ~32s
```

## Data Privacy & MCP Vision

All sensing data, baselines, episode records, and audit logs are stored **locally** on the edge device. External systems do not directly retrieve raw data. The internal conversational query interface already uses the same retrieval tools as the agent layer — registering them as an MCP server is only a protocol-level encapsulation step for future authorization-gated access.
