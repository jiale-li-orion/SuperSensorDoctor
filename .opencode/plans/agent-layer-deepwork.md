# SuperSenseDoctor Agent Layer — Deepwork Progress

## Goal
实现 SuperSenseDoctor 论文中的 MultiAgent Collaboration Layer：完整 Agent 层 + 感知模拟器 + Web 界面。

## Decisions Made

| 决策 | 结论 |
|------|------|
| 实现范围 | 完整 Agent 协作层 (Nurse + Diagnosis + Report + EventBus + TieredAction) |
| 语言/框架 | Python 3.12 + FastAPI + SQLite, 不引入 LangChain/LangGraph |
| LLM | DeepSeek V4 Flash/Pro (OpenAI 兼容), 后续切换 Ollama 本地模型 |
| Agent 循环 | 裸 ReAct Think/Act 分离, ~50 行 while 循环, 无框架依赖 |
| Tool 定义 | @tool 装饰器 + ToolRegistry, 借鉴 AutoGen caller/executor 分离 |
| 事件总线 | 进程内 typed EventBus, 借鉴 CrewAI |
| 感知数据 | 队友给零散文件 → SensorAligner 对齐 → ConfidenceEstimator 算置信度 |
| 测试数据 | 自己构造 |
| 数据库 | SQLite 3 表, JSON 列存嵌套 dataclass |

## Borrowed Patterns (Research)
- **AutoGen/FinRobot**: caller/executor tool separation
- **CrewAI**: typed event bus with Pydantic models
- **LangGraph**: Think/Act dual-node ReAct, conditional edges, parallel tool execution
- **Trading agents**: Event → Rule Filter → Agent Loop → Action pipeline

## Current Status
- [x] 项目上下文探索
- [x] 数据格式分析 + 与论文对齐
- [x] 技术方案讨论 (3 方案对比, 选定混合架构)
- [x] 模块设计逐节确认
- [ ] 完整设计规范文档
- [ ] 实现计划
- [ ] 分阶段实现
