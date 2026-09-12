# SuperSenseDoctor — MultiAgent Collaboration Layer

**English** · 中文说明见各章节末尾的可折叠区块（点击 **中文说明** 展开）

SuperSenseDoctor is a multimodal and contactless intelligent health-guarding system for home-based elderly care. On a home edge device, the system uses **WiFi Beamforming Feedback Information (BFI)**, **mmWave radar**, and **an infrared thermal array** to continuously track respiration, heart rate, body temperature, posture, and fall events — without requiring the elderly person to wear or operate any device.

This repository implements the **MultiAgent Collaboration Layer**: the orchestration layer that transforms continuous low-level sensing estimates into auditable, actionable long-term health tracking decisions. It is the Agent-layer artifact accompanying the published paper below.

<details>
<summary><b>中文说明</b></summary>

SuperSenseDoctor 是一套面向居家养老的多模态非接触式智能健康守护系统。系统在家庭边缘设备上使用 **WiFi 波束成形反馈信息（BFI）**、**毫米波雷达** 与 **红外热成像阵列**，持续跟踪呼吸、心率、体温、体态与跌倒事件，无需老人佩戴或操作任何设备。

本仓库实现其中的 **MultiAgent Collaboration Layer（多智能体协作层）**：把连续的底层传感估计，转化为可审计、可行动的长期健康跟踪决策。它也是下方已发表论文所对应的 Agent 层工程产物。

</details>

---

## Paper

> **SuperSenseDoctor: A Multimodal and Contactless Agent for Health Tracking**
> Xuwen Zhang, Zijian Lu, Yicheng Lei, Rui Qiu, **Jiale Li**, Yiping Zuo, Weibei Fan, and Fu Xiao.
> *Companion of the 2026 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp Companion '26)*, October 11–15, 2026, Shanghai, China. ACM, New York, NY, USA, 5 pages.

**DOI:** [10.1145/3798063.3837323](https://doi.org/10.1145/3798063.3837323) · **CCS:** Human-centered computing → Ubiquitous and mobile computing systems and tools

### Cite

```bibtex
@inproceedings{zhang2026supersensedoctor,
  author    = {Zhang, Xuwen and Lu, Zijian and Lei, Yicheng and Qiu, Rui and
               Li, Jiale and Zuo, Yiping and Fan, Weibei and Xiao, Fu},
  title     = {SuperSenseDoctor: A Multimodal and Contactless Agent for Health Tracking},
  booktitle = {Companion of the 2026 ACM International Joint Conference on
               Pervasive and Ubiquitous Computing (UbiComp Companion '26)},
  year      = {2026},
  pages     = {1--5},
  location  = {Shanghai, China},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3798063.3837323}
}
```

### Presentation companion

[`showcase/index.html`](showcase/index.html) is a self-contained single-page case narrative for this work — one fictional resident walked end to end through the pipeline, with the paper's Table 2 and validation figures alongside. Open it directly in a browser; it needs no server and no build step. See [`showcase/README.md`](showcase/README.md).

### Reported results

The paper validates the end-to-end pipeline; this repository covers the Agent layer (§2.2) and its evaluation (§4.2).

| Layer | Metric | Result |
|-------|--------|--------|
| Sensing (§4.1) | Heart rate MAE / RMSD vs. Huawei Watch GT 3 | **1.994 / 3.142 bpm** (−51.6% MAE vs. mmWave, −52.9% vs. WiFi BFI) |
| Sensing (§4.1) | Respiratory rate MAE / RMSD vs. respiration belt | **0.197 / 0.263 bpm** (−87.9% vs. mmWave, −92.7% vs. WiFi BFI) |
| Sensing (§4.1) | Fall recognition accuracy | **96.5%** |
| Coverage (§4.1) | Multi-interval state validation | **9 intervals, 2686 one-second state rows** |
| **Agent (§4.2)** | Rule screening + hidden-answer triage checklist | **206 / 213 criteria = 96.7%** |
| Agent (§4.2) | Deterministic rule screening | 58/60 event matches; fusion arbitration 23/24 |
| Agent (§4.2) | Hidden-answer triage (GPT-5.4) | tier + channel 22/24; trace preserved 23/24 |
| Agent (§4.2) | P95 latency | deterministic **0.56 ms** · LLM triage **6560.7 ms** |

The §4.2 numbers are also exposed at runtime through `agent_layer/validation_results.py`, which both report rendering paths read so they cannot drift from the paper.

### Prototype hardware

Comfast WU785AC (WiFi BFI) · Texas Instruments AWR1843 (mmWave radar) · MLX90640 (32×24 IR thermal array) · Intel NUC (local hub and Agent runtime).

<details>
<summary><b>中文说明 — 论文与实验结果</b></summary>

**论文**：SuperSenseDoctor: A Multimodal and Contactless Agent for Health Tracking，发表于 *UbiComp Companion '26*（2026 年 10 月 11–15 日，中国上海），ACM 出版，共 5 页。

**DOI**：[10.1145/3798063.3837323](https://doi.org/10.1145/3798063.3837323) —— BibTeX 见上方英文区。

**已报告结果**（论文验证的是端到端流水线，本仓库覆盖 §2.2 Agent 层与 §4.2 其评测）：

| 层次 | 指标 | 结果 |
|------|------|------|
| 感知（§4.1） | 心率 MAE / RMSD（对比 Huawei Watch GT 3） | **1.994 / 3.142 bpm**（相比 mmWave MAE 降低 51.6%，相比 WiFi BFI 降低 52.9%） |
| 感知（§4.1） | 呼吸率 MAE / RMSD（对比呼吸带） | **0.197 / 0.263 bpm**（相比 mmWave 降低 87.9%，相比 WiFi BFI 降低 92.7%） |
| 感知（§4.1） | 跌倒识别准确率 | **96.5%** |
| 覆盖面（§4.1） | 多区间状态验证 | **9 个区间、2686 行一秒状态数据** |
| **Agent（§4.2）** | 规则筛查 + 隐藏答案分诊检查项 | **206 / 213 项 = 96.7%** |
| Agent（§4.2） | 确定性规则筛查 | 事件匹配 58/60；融合仲裁 23/24 |
| Agent（§4.2） | 隐藏答案分诊（GPT-5.4） | 分级与渠道 22/24；证据链保留 23/24 |
| Agent（§4.2） | P95 延迟 | 确定性 **0.56 ms** · LLM 分诊 **6560.7 ms** |

上述 §4.2 数字在运行时由 `agent_layer/validation_results.py` 统一提供，两条报告渲染路径都从它读取，因此不会与论文脱节。

**原型硬件**：Comfast WU785AC（WiFi BFI）· 德州仪器 AWR1843（毫米波雷达）· MLX90640（32×24 红外热阵列）· Intel NUC（本地中枢与 Agent 运行时）。

</details>

---

## Getting Started

### Requirements

| | |
|---|---|
| Python | **3.12** (the code uses `list[str]` / `X \| None` syntax). Windows: `py -3.12`, Linux/WSL: `python3.12` |
| OS | Linux, **WSL (recommended)**, or Windows 10/11 with PowerShell |
| Git | Any recent version |
| Hardware | **None.** The sensing layer replays configured CSV / derived files — no WiFi card, radar, or thermal array is needed to run the Agent layer |
| LLM key | **Optional.** Without it the report falls back to the deterministic renderer and everything still runs |

---

### Step 1 — Clone

Both transports work; pick one.

**SSH** (needs an SSH key registered on GitHub):

```bash
git clone git@github.com:jiale-li-orion/SuperSensorDoctor.git
cd SuperSensorDoctor
```

**HTTPS** (works everywhere, may prompt for a username/token):

```bash
git clone https://github.com/jiale-li-orion/SuperSensorDoctor.git
cd SuperSensorDoctor
```

> The checkout directory in this workspace is named `ubicomp/`. Substitute your
> own path wherever the commands below say `<repo>`.

---

### Step 2 — Bootstrap and run

<details open>
<summary><b>WSL / Linux (bash)</b></summary>

```bash
cd <repo>

# 1. Confirm the interpreter
python3 --version                 # expect 3.12.x

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: enable live LLM triage
export DEEPSEEK_API_KEY=sk-...

# 5. Run
python main.py                    # → http://127.0.0.1:8000
```

If step 2 fails with `No module named venv` / `ensurepip is not available`:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To deactivate the environment later: `deactivate`.

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd <repo>

# 1. Confirm the interpreter (use the py launcher, not bare `python`)
py -3.12 --version                # expect 3.12.x

# 2. Create and activate a virtual environment
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: enable live LLM triage (session-scoped)
$env:DEEPSEEK_API_KEY = "sk-..."

# 5. Run
python main.py                    # → http://127.0.0.1:8000
```

**If `Activate.ps1` is blocked** with *"running scripts is disabled on this system"*,
allow it for the current shell only (no permanent policy change):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**If the console shows mojibake**, or you hit
`UnicodeDecodeError: 'gbk' codec can't decode byte ...`, switch the console to
UTF-8 for the session:

```powershell
$env:PYTHONUTF8 = "1"
chcp 65001
python main.py
```

To deactivate the environment later: `deactivate`.

Using **cmd.exe** instead of PowerShell? Same steps, but activate with
`venv\Scripts\activate.bat`.

</details>

<details>
<summary><b>Both platforms — one-liners</b></summary>

WSL / Linux:

```bash
cd <repo> && python3 -m venv .venv && source .venv/bin/activate \
  && pip install -r requirements.txt && python main.py
```

Windows PowerShell:

```powershell
cd <repo>; py -3.12 -m venv venv; .\venv\Scripts\Activate.ps1; `
  pip install -r requirements.txt; python main.py
```

The `Makefile` targets (`make install` / `make run` / `make test`) are
convenience wrappers for the `bash` path only — PowerShell has no `make` by
default, so use the explicit commands above.

</details>

---

### Step 3 — Use it

Open **<http://127.0.0.1:8000>**.

| Route | What it shows |
|-------|---------------|
| `/` | Live dashboard — care state, trend chart, vitals, confidence, recent events, weekly counts |
| `/episodes` | All triage episodes, filterable by tier and event type |
| `/episode/<episode_id>` | Full evidence chain for one episode: trigger → anchors → tier → channel → tools → audit |
| `/report` | Weekly care report |

**First launch seeds itself.** If `data/supersense.db` is empty the app
auto-seeds 10 demo episodes, 10 health events, and 400 sensing windows, so the
UI is never blank. There is no manual data-loading step.

**Report page parameters** (combinable):

| Parameter | Values | Effect |
|-----------|--------|--------|
| `view` | `report` (default) · `blocks` | `report` = Agent-written markdown narrative; `blocks` = structured HTML panels with charts and tables |
| `lang` | `zh` (default) · `en` | Switches the whole UI; also persisted in a `lang` cookie by the in-page `中` / `EN` toggle |
| `figure` | `1` | Print / figure mode — hides navigation and action buttons |
| `llm` | `1` | Regenerates the report through the LLM **if** `DEEPSEEK_API_KEY` is set; silently falls back otherwise |

```
http://127.0.0.1:8000/report?view=blocks&lang=en
http://127.0.0.1:8000/report?view=report&lang=zh&figure=1
```

---

### Step 4 — Run the tests

WSL / Linux:

```bash
source .venv/bin/activate
pytest tests/ -q
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pytest tests/ -q
```

Provider-free — no API key required. The Agent-layer suites run fully offline.

```bash
pytest tests/test_report_agent.py tests/test_report_data.py -v
# 37 passed — canonical blocks, evidence trail, quality-rate deduplication,
#              decision-path inference, bilingual output, LLM path,
#              provider-failure fallback, Q&A, report window selection
```

Browser end-to-end suite (optional, extra install — same on both platforms):

```bash
pip install playwright && playwright install chromium
pytest tests/test_web_e2e.py -q
```

> ### ⚠️ Running the test suite wipes `data/supersense.db`
>
> Seven test modules delete the **shared repository database** in their
> fixtures — `tests/test_main_integration.py` does so with `autouse=True`:
>
> ```python
> @pytest.fixture(autouse=True)
> def clean_db():
>     os.remove(DB_PATH)      # the real data/supersense.db, not a temp copy
> ```
>
> **This is a known defect, not intended behaviour.** Consequences:
>
> - Any demo or replayed data you had loaded is destroyed.
> - Later tests in the same run see a database an earlier test deleted — the
>   root cause of the current `14 failed / 7 errors` baseline. It is **not**
>   21 independent product defects.
> - On Windows the failure count is inflated further, because deleting a file
>   whose SQLite handle is still open raises `WinError 32`.
> - `make clean` deletes the same file.
>
> Recovery is automatic: delete `data/supersense.db` and restart; the app
> re-seeds the 10-episode demo set. Data imported from `portable_v2` or team
> CSVs must be re-imported (`scripts/load_portable_v2.py`, or the load buttons
> in the UI). Back up first if it matters:
>
> WSL / Linux:
>
> ```bash
> cp data/supersense.db /tmp/ssd-backup.db
> ```
>
> Windows PowerShell:
>
> ```powershell
> Copy-Item data\supersense.db $env:TEMP\ssd-backup.db
> ```
>
> The fix is a per-test `tmp_path` database with connections closed before
> teardown — tracked under *Known issues*.

---

### Makefile targets

Bash only (WSL / Linux):

```bash
make install   # pip install -r requirements.txt
make run       # python main.py
make test      # pytest tests/ -v        ⚠ deletes data/supersense.db
make clean     # removes __pycache__     ⚠ also deletes data/supersense.db
```

---

### Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `python: command not found` (WSL/Linux) | Use `python3`, or activate the venv first so `python` resolves inside it. |
| `python` opens the Microsoft Store (Windows) | The App Execution Alias is intercepting it. Use `py -3.12`, or disable the alias under *Settings → Apps → App execution aliases*. |
| `.\venv\Scripts\Activate.ps1 : running scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (current shell only). |
| `ModuleNotFoundError: No module named 'markdown'` | Dependencies not installed, or the venv is not active. `pip install -r requirements.txt` (declared as `Markdown==3.10.2`). |
| `ModuleNotFoundError: No module named 'yaml'` | Same — `PyYAML` is in `requirements.txt`. |
| `UnicodeDecodeError: 'gbk' codec can't decode byte ...` | Native Windows reading `config.yaml` without an explicit encoding. Set `$env:PYTHONUTF8 = "1"`, or use WSL. |
| `PermissionError: [WinError 32] The process cannot access the file` | A SQLite handle is still open — this is the test-isolation defect above. Close the app, or use WSL. |
| `/report` returns HTTP 500 with `'NoneType' object has no attribute 'get'` | Empty database with no sensing rows. Guarded in `web/app.py`; restart to trigger the auto-seed. |
| Report page unstyled, or `TemplateNotFound` | `base.html` links `/static/ssd.css`. Confirm the file exists under `web/static/`. |
| `sqlite3.IntegrityError: FOREIGN KEY constraint failed` when writing an episode | `episode_logs.event_id` references `health_events`. Any subscriber calling `DiagnosisAgent.handle_event()` must first persist the parent row with `insert_health_event(...)` — see the `on_diagnosis_event` subscriber in `main.py`. |
| Port 8000 already in use | Change `web.port` in `config.yaml`. |

> **WSL vs Windows interpreters.** The repo root may contain two environment
> directories: `.venv/` (Linux layout — use `.venv/bin/python`) and `venv/`
> (Windows layout — `venv\Scripts\python.exe`). **Under WSL always use `.venv`**;
> the Windows one cannot be activated from bash, and vice versa. Both are
> gitignored, so a fresh clone has neither.

<details>
<summary><b>中文说明 — 快速开始</b></summary>

### 环境要求

| | |
|---|---|
| Python | **3.12**（代码使用 `list[str]` / `X \| None` 语法）。Windows 用 `py -3.12`，Linux/WSL 用 `python3.12` |
| 操作系统 | Linux、**WSL（推荐）**，或 Windows 10/11 + PowerShell |
| Git | 任意较新版本 |
| 硬件 | **不需要**。感知层回放已配置的 CSV / 派生文件，运行 Agent 层无需任何 WiFi 网卡、雷达或热成像硬件 |
| LLM Key | **可选**。不设置时周报回退到确定性渲染器，其余功能照常 |

### 第一步 —— 克隆

两种方式任选。

**SSH**（需已在 GitHub 注册 SSH key）：

```bash
git clone git@github.com:jiale-li-orion/SuperSensorDoctor.git
cd SuperSensorDoctor
```

**HTTPS**（通用，可能要求输入用户名 / token）：

```bash
git clone https://github.com/jiale-li-orion/SuperSensorDoctor.git
cd SuperSensorDoctor
```

> 本工作区中的检出目录名为 `ubicomp/`。下文命令中的 `<repo>` 请替换为你自己的路径。

### 第二步 —— 安装并启动

**WSL / Linux（bash）**

```bash
cd <repo>
python3 --version                 # 应为 3.12.x
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...    # 可选，启用实时 LLM 分诊
python main.py                    # → http://127.0.0.1:8000
```

若第二步报 `No module named venv` / `ensurepip is not available`：

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

退出环境：`deactivate`。

**Windows（PowerShell）**

```powershell
cd <repo>
py -3.12 --version                # 应为 3.12.x
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "sk-..."  # 可选，仅当前会话有效
python main.py                    # → http://127.0.0.1:8000
```

若 `Activate.ps1` 被拦，报 *"running scripts is disabled on this system"*，
**只对当前 shell** 放开（不永久修改系统策略）：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

若控制台中文乱码，或报 `UnicodeDecodeError: 'gbk' codec can't decode byte ...`，
把当前会话切到 UTF-8：

```powershell
$env:PYTHONUTF8 = "1"
chcp 65001
python main.py
```

退出环境：`deactivate`。用 **cmd.exe** 的话步骤相同，激活命令换成
`venv\Scripts\activate.bat`。

### 第三步 —— 使用

打开 **<http://127.0.0.1:8000>**。

| 路由 | 内容 |
|------|------|
| `/` | 实时看板 —— 看护状态、趋势图、体征、置信度、最近事件、本周统计 |
| `/episodes` | 全部分诊事件，可按分级与事件类型筛选 |
| `/episode/<episode_id>` | 单条事件的完整证据链：触发 → 证据依据 → 分级 → 渠道 → 工具 → 审计 |
| `/report` | 周报 |

**首次启动会自动播种。** 若 `data/supersense.db` 为空，应用会自动写入 10 条演示
episode、10 条 health event 与 400 个传感窗口，界面不会空白，无需手动加载数据。

**周报页参数**（可组合）：

| 参数 | 取值 | 作用 |
|------|------|------|
| `view` | `report`（默认）· `blocks` | `report` = Agent 生成的 markdown 叙述；`blocks` = 结构化 HTML 区块（含图表与表格） |
| `lang` | `zh`（默认）· `en` | 切换整站语言；页面内 `中` / `EN` 按钮会写入 `lang` cookie 持久化 |
| `figure` | `1` | 出图模式，隐藏导航与操作按钮 |
| `llm` | `1` | **若**已设置 `DEEPSEEK_API_KEY` 则走 LLM 重新生成周报，否则静默回退 |

### 第四步 —— 运行测试

WSL / Linux：

```bash
source .venv/bin/activate
pytest tests/ -q
```

Windows PowerShell：

```powershell
.\venv\Scripts\Activate.ps1
pytest tests/ -q
```

无需 API Key，Agent 层测试可完全离线运行。浏览器端到端套件为可选项，两个平台一致：
先 `pip install playwright` 再 `playwright install chromium`。

> ### ⚠️ 跑测试会清空 `data/supersense.db`
>
> **七个测试模块**会在 fixture 中删除**共享的仓库数据库**，其中
> `tests/test_main_integration.py` 用的是 `autouse=True`：
>
> ```python
> @pytest.fixture(autouse=True)
> def clean_db():
>     os.remove(DB_PATH)      # 删的是真的 data/supersense.db，不是临时副本
> ```
>
> **这是已知缺陷，不是设计意图。** 后果：
>
> - 你已加载的演示数据或回放数据会被销毁。
> - 同一次运行中，后面的测试看到的是被前面测试删掉的库 —— 这才是当前
>   `14 failed / 7 errors` 基线的根因，**不是 21 个独立产品缺陷**。
> - Windows 下失败数会进一步放大：SQLite 句柄未释放时删除文件会抛 `WinError 32`。
> - `make clean` 会删除同一个文件。
>
> 恢复是自动的：删掉 `data/supersense.db` 后重启，应用会重新播种 10 条演示数据。
> 从 `portable_v2` 或队伍 CSV 导入的数据需重新导入（`scripts/load_portable_v2.py`，
> 或界面上的加载按钮）。数据重要时先备份：
>
> WSL / Linux：`cp data/supersense.db /tmp/ssd-backup.db`
>
> Windows PowerShell：`Copy-Item data\supersense.db $env:TEMP\ssd-backup.db`
>
> 正确修法是把测试库指向每个测试独立的 `tmp_path`，并在 teardown 前关闭连接，
> 已记录在「已知问题」中。

### Makefile 目标（仅 bash / WSL / Linux）

```bash
make install   # pip install -r requirements.txt
make run       # python main.py
make test      # pytest tests/ -v        ⚠ 会删除 data/supersense.db
make clean     # 清理 __pycache__        ⚠ 同时删除 data/supersense.db
```

PowerShell 默认没有 `make`，请直接使用上文列出的显式命令。

### 常见故障排查

| 现象 | 原因与处理 |
|------|-----------|
| `python: command not found`（WSL/Linux） | 改用 `python3`，或先激活 venv，使 `python` 指向环境内的解释器。 |
| 在 Windows 输入 `python` 却打开 Microsoft Store | 应用执行别名拦截。改用 `py -3.12`，或在 *设置 → 应用 → 应用执行别名* 中关闭。 |
| `.\venv\Scripts\Activate.ps1 : running scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`（仅当前 shell 生效）。 |
| `ModuleNotFoundError: No module named 'markdown'` | 依赖未安装或 venv 未激活。执行 `pip install -r requirements.txt`（已声明 `Markdown==3.10.2`）。 |
| `ModuleNotFoundError: No module named 'yaml'` | 同上，`PyYAML` 已在 `requirements.txt` 中。 |
| `UnicodeDecodeError: 'gbk' codec can't decode byte ...` | 原生 Windows 未显式指定编码读取 `config.yaml`。设置 `$env:PYTHONUTF8 = "1"`，或改用 WSL。 |
| `PermissionError: [WinError 32] The process cannot access the file` | SQLite 句柄未释放 —— 即上述测试隔离缺陷。关闭应用，或改用 WSL。 |
| `/report` 返回 HTTP 500，报 `'NoneType' object has no attribute 'get'` | 空数据库、无传感记录。`web/app.py` 已加保护；重启以触发自动播种。 |
| 周报页无样式，或报 `TemplateNotFound` | `base.html` 引用 `/static/ssd.css`，确认该文件存在于 `web/static/`。 |
| 写入 episode 时报 `sqlite3.IntegrityError: FOREIGN KEY constraint failed` | `episode_logs.event_id` 外键指向 `health_events`。任何调用 `DiagnosisAgent.handle_event()` 的订阅者都必须先用 `insert_health_event(...)` 落库父行 —— 参见 `main.py` 的 `on_diagnosis_event`。 |
| 8000 端口被占用 | 修改 `config.yaml` 中的 `web.port`。 |

> **WSL 与 Windows 解释器之别。** 仓库根目录可能同时存在两套环境：
> `.venv/`（Linux 结构，用 `.venv/bin/python`）与 `venv/`（Windows 结构，
> `venv\Scripts\python.exe`）。**在 WSL 下一律使用 `.venv`** —— Windows 那套无法从
> bash 激活，反之亦然。两者均被 gitignore 忽略，因此全新 clone 下都不存在。

</details>

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
│  │  (异常事件)       (9 tools)                        │   │
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

1. **WiFi BFI Pipeline** — Raw BFM matrix stream from OpenWRT Sniffer via SSH pipe → BFM-ratio normalization → dual-peak tracking for respiration / heart-rate estimation.
2. **mmWave Pipeline** — Point-cloud frames (x, y, z, v, I) from TI AWR1843 → vital-sign extraction + fall detection via height/velocity thresholding.
3. **IR Thermal Pipeline** — MLX90640 32×24 array → 8×8 zonal reduction → peak body-surface temperature.
4. **SensorAligner** — Aligns all three pipelines to the same time window (configurable, default 1 s) → produces a unified `StateObject`.
5. **SensorHub** — Assembles `StateObject` → writes to SQLite `sensing_windows` → forwards to NurseAgent.
6. **NurseAgent** — Deterministic rule engine. Silent for normal states; publishes `HealthEvent` for boundary violations, falls, or cross-modal anomalies.
7. **DiagnosisAgent** — Receives event → Think (LLM without tools) / Act (call ToolRegistry tools) / Decide (output `TriageDecision` JSON) ReAct loop. Max 8 steps, falls back to L0.
8. **TieredAction** — Maps decision level to channel: L0(none) → L1(none) → L2(screen) → L3(family_push) → L4(emergency).
9. **ReportAgent** — Generates weekly summaries from `EpisodeLog` records; answers natural-language questions.

---

## Cross-Modal Fusion Arbitration

When the Diagnosis Agent needs to resolve conflicting estimates across modalities it uses a 3-step chain returning a structured `FusionResult`:

| Step | Name | Logic | Output |
|------|------|-------|--------|
| 1 | Confidence Evaluation | Check each modality's confidence score; check `nlos_flag` (mmWave known-failure mode) | `estimates` — per-modality value/confidence |
| 2 | Consistency Check | Compute cross-modal delta + confidence gap | `checks` — delta, consistent, confidence_gap |
| 3 | Arbitration | NLOS → WiFi dominates; both reliable → fusion; conflict → trust the reliable branch | `verdict` — fused_value, dominant_modality, rationale |

**Arbitration priority:** fall alert > NLOS switching > confidence arbitration > semantic merging.

---

## Agent System Design

### Nurse Agent (Non-LLM Rule Engine)

- **Principle**: Remain silent at normal states; publish an event for anomalies; trigger the reflex arc for high-risk combinations (e.g. fall + elevated HR).
- **Rules**: Fall detection, heart-rate deviation (>2σ), temperature deviation (>threshold), cross-modal confidence degradation.
- **Reflex arc**: Certain high-risk patterns bypass LLM deliberation and publish L3/L4 events directly.

### Diagnosis Agent (ReAct Think/Act Loop)

- **Think phase**: Calls the LLM without tools → analyses event + resident baseline → attempts to output a `TriageDecision`.
- **Act phase**: If no decision is parsed, calls the LLM with tools → executes the `tool_call` via ToolRegistry → appends the result to message history → repeats. Tool results are captured in the evidence chain.
- **Tool system**: 9 registered tools in 3 categories (Evidence / Action / Persistence) — see *Tool Taxonomy*.
- **Decision schema**: `TriageDecision` — `{"level": "L2", "label": "resident_alert", "event_interpretation": "...", "evidence_used": [...], "clinical_basis": [...], "uncertainty": {"sensing_quality": "reliable", ...}, "action": {"channel": "screen", ...}, "safety_boundary": "care_support_only"}`
- **JSON parser**: Brace-counting extractor (handles nested `TriageDecision` objects, unlike a simple regex).
- **Fallback**: Returns L0 (silent record) if `max_steps` is exceeded without a parseable decision.

### Report Agent

The Report Agent closes the loop by turning triage records into longitudinal memory. Two rendering paths share **one** evidence context, so they cannot disagree on counts, tiers, or traces:

| Path | Entry point | Output |
|------|-------------|--------|
| **Primary — agent report** | `ReportAgent.generate_weekly_report(..., llm_provider=...)` | Markdown prose from the LLM, grounded in the canonical context |
| **Fallback — deterministic** | `render_fallback_report(context)` | Markdown with the same blocks, no provider required |

Both emit the blocks declared in `REPORT_BLOCKS`:

```
summary · evidence_trail · sensing_quality · action_routing · uncertainty
· vital_trends · personal_baseline · validation · indicator_coverage · privacy_boundary
```

- `build_report_context(episodes, events, reference_ts)` is the single source of truth — fully deterministic and provider-free.
- `agent_layer/report_data.py` owns **how a report window is loaded**: `ReportAgent.load_window(resident_id, ...)` and `generate_weekly_report_for(resident_id, ...)` select episodes by time range, never by "the newest N rows", so a busy week cannot be silently truncated. It is the only place in the Agent layer that reaches into storage for report data.
- The **evidence trail** block renders the paper's traceable chain explicitly: Nurse event → evidence anchors → triage tier → delivery channel → persisted `EpisodeLog`, including the reflex-path flag and the tools used.
- Provider failures and timeouts fall through to the deterministic renderer; empty or missing LLM prose never surfaces as report content.
- Wording is **care support**, not diagnosis.

### Bilingual report (EN / 中文)

Both rendering paths are fully bilingual and produce **pure** output in one language — never mixed:

| Entry point | Language control |
|-------------|------------------|
| `build_report_context(..., lang="zh")` | Resolves every label pair for that language; codes, counts, and identifiers stay language-neutral |
| `render_fallback_report(context, lang)` | Deterministic markdown in that language |
| `ReportAgent.generate_weekly_report(..., lang="zh")` | Picks `REPORT_PROMPT_ZH` and the Chinese fallback |
| `ReportAgent.answer_question(q, lang)` | Matches keyword lists in **both** languages, answers in `lang` |

- Every display label is a `(zh, en)` pair: `TIER_LABELS`, `CHANNEL_LABELS`, `SOURCE_LABELS`, `EVENT_LABELS`, `SENSING_QUALITY_LABELS`, `PRIVACY_BOUNDARY`.
- `report_prompt(lang)` selects `REPORT_PROMPT` (EN) or `REPORT_PROMPT_ZH` (中文); both constrain the model to the same ten `##` blocks.
- `validation_results.py` carries Chinese counterparts (`task_zh`, `ground_truth_zh`, `condition_zh`, `INDICATOR_COLUMNS_ZH`, `CLINICAL_BOUNDARY_ZH`) alongside the English fields.
- Unknown language tags fall back to English; `lang` is stored on the context so a renderer can default to the language the context was built with.

The web layer keeps its own inline convention in `web/i18n.py`: `L('中文', 'English')` on the server and `T('中文', 'English')` in the browser, switched with the `lang` cookie.

### Published validation results

`agent_layer/validation_results.py` holds the paper's reported numbers as a single typed source of truth:

- Table 1 — heart-rate / respiratory-rate MAE and RMSD per branch and fused, plus fall-recognition accuracy
- §4.1 — multi-interval validation (9 intervals, 2686 one-second state rows)
- §4.2 — Agent scenario evaluation (60 deterministic cases, 24 hidden-answer triage scenarios, 206/213 criteria, P95 latencies)
- Table 2 — disease-relevant indicator coverage matrix

These are **reported results, not live measurements**, and both rendering paths label them as such.

### Tiered Action Strategy

| Level | Label | Channel | Message | Recheck |
|-------|-------|---------|---------|---------|
| L0 | 静默记录 | none | 正常范围，仅写数据库 | — |
| L1 | 持续观察 | none | 轻度偏离，定时复查 | 300 s |
| L2 | 居民提醒 | screen | 建议调整姿势/确认状态 | 600 s |
| L3 | 家属告警 | family_push | 持续异常，已通知家属 | 1800 s |
| L4 | 紧急告警 | emergency | 立即联系居民确认状态 | 60 s |

### Tool Taxonomy

| Category | Tool | Function |
|----------|------|----------|
| **Evidence Tools** | `query_history` | 查询个人历史基线，计算均值/极值 |
| | `get_latest_vitals` | 最新生命体征快照 |
| | `read_sensing_state` | 最新多模态传感快照（含置信度 / NLOS / 跌倒 / 活动） |
| | `list_recent_events` | 最近的异常事件列表 |
| | `check_resident_context` | 居民传感器元数据上下文 |
| | `trend_analysis` | 指标在时间窗口内的趋势（升/降/稳定） |
| | `consult_fusion` | 三步链式跨模态仲裁（estimates → checks → verdict） |
| **Action Tools** | `issue_action` | 解析 L0–L4 分级行动方案（channel / recheck） |
| **Persistence Tools** | `write_episode` | 写入可审计诊断事件记录（含完整 evidence 链） |

### Auditable Episode Record

Every Diagnosis Agent interaction is recorded as an `EpisodeLog` containing:

- `evidence`: `{"event": {...}, "sensing_summary": {...}, "tool_results": [...]}` — full audit chain
- `decision`: `{"level": "L2", "label": "resident_alert", "event_interpretation": "...", "evidence_used": [...], "clinical_basis": [...], "uncertainty": {...}}` — TriageDecision schema
- `action`: `{"channel": "screen", "message": "...", "recheck_after": 600}`
- `audit`: `{"tools_called": ["query_history"], "step_count": 3, "reflex": false, "event_id": "..."}`

The LLM's private reasoning chain is not stored; only the auditable summary is persisted.

<details>
<summary><b>中文说明 — 架构与 Agent 设计</b></summary>

**端到端数据流**：WiFi BFI 流水线 → mmWave 点云流水线 → 红外热成像流水线 → `SensorAligner` 对齐到同一时间窗（默认 1 s）→ `SensorHub` 组装 `StateObject` 并落库 `sensing_windows` → 转交 `NurseAgent` → 规则引擎在正常状态保持静默、异常时发布 `HealthEvent` → `DiagnosisAgent` 走 Think/Act/Decide 的 ReAct 循环（最多 8 步，无法解析则回退 L0）→ `TieredAction` 把分级映射到渠道 → `ReportAgent` 从 `EpisodeLog` 生成周报并支持自然语言问答。

**跨模态融合仲裁**（三步链，输出结构化 `FusionResult`）：

| 步骤 | 名称 | 逻辑 | 输出 |
|------|------|------|------|
| 1 | 置信度评估 | 检查各模态置信度与 `nlos_flag`（mmWave 已知失效模式） | `estimates` — 各模态估计值/置信度 |
| 2 | 一致性检查 | 计算跨模态差值与置信度差 | `checks` — delta、consistent、confidence_gap |
| 3 | 仲裁 | NLOS → WiFi 主导；两者可靠 → 融合；冲突 → 采信可靠分支 | `verdict` — fused_value、dominant_modality、rationale |

**仲裁优先级**：跌倒告警 > NLOS 切换 > 置信度仲裁 > 语义合并。

**Agent 分工**：

- **NurseAgent（非 LLM 规则引擎）** —— 正常状态静默；异常发布事件；高风险组合（如跌倒 + 心率升高）走反射弧，绕过 LLM 直接发布 L3/L4。
- **DiagnosisAgent（ReAct 循环）** —— Think 阶段不带工具调用 LLM 尝试直接输出 `TriageDecision`；未解析出决策则进入 Act 阶段带工具调用，经 ToolRegistry 执行并把结果追加进消息历史后重试。使用括号配平解析器（可处理嵌套 JSON，优于正则），超步数则回退 L0。
- **ReportAgent** —— 把分诊记录转为纵向记忆。两条渲染路径共享**同一份**证据上下文，因此计数、分级与链路不可能彼此矛盾；`build_report_context()` 是唯一真相源，完全确定性、不依赖任何 provider。**证据链**区块显式渲染论文要求的可追溯链：护理事件 → 证据依据 → 分诊分级 → 投递渠道 → 落库 `EpisodeLog`，并含反射弧标志与调用过的工具。

**双语文档能力**：所有展示标签均为 `(zh, en)` 二元组；`report_prompt(lang)` 选择 `REPORT_PROMPT` 或 `REPORT_PROMPT_ZH`，两者约束模型输出同样的十个 `##` 区块；`validation_results.py` 为论文结果提供中文字段。未知语言标签回退英文。Web 层用 `web/i18n.py` 的内联约定：服务端 `L('中文','English')`，浏览器 `T('中文','English')`，以 `lang` cookie 切换。

**L0–L4 分级行动策略**：L0 静默记录（none）· L1 持续观察（none，300 s 复查）· L2 居民提醒（screen，600 s）· L3 家属告警（family_push，1800 s）· L4 紧急告警（emergency，60 s）。

**可审计流程记录**：每次 Diagnosis Agent 交互都落为 `EpisodeLog`，含 `evidence`（事件 + 传感摘要 + 工具结果）、`decision`（TriageDecision）、`action`（渠道/复查）、`audit`（调用工具、步数、反射弧标志）。**LLM 的私有推理链不落库**，只持久化可审计的摘要。

</details>

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
│   ├── tools.py                # @tool 装饰器 + ToolRegistry + 9 tools
│   ├── tiered_action.py        # L0-L4 分级行动策略
│   ├── llm_provider.py         # DeepSeekProvider + MockProvider (OpenAI 兼容)
│   ├── clinical_policy.py      # 临床参考层 (RCP NEWS2 reference / NICE NG249)
│   ├── validation_results.py   # 论文已发表结果 (Table 1 / §4.1 / §4.2 / Table 2)
│   ├── report_data.py          # 周报数据访问: 按时间窗取 episode (无行数上限)
│   └── report_agent.py         # 周报: 共享证据上下文 + LLM prose / 确定性回退
├── sensing_simulator/          # 感知模拟器 (论文演示用)
│   ├── sensor_aligner.py       # 多文件时间窗对齐 (HR + fall + temp)
│   ├── sensor_hub.py           # 数据汇聚 + SQLite 持久化 + NurseAgent 触发
│   └── replay_engine.py        # 合成数据生成器 (可注入异常模式)
├── storage/                    # 持久化层
│   ├── db.py                   # SQLite WAL 模式连接 + 3 表 schema
│   └── models.py               # 数据访问层 (CRUD + 时间窗查询 query_episodes_in_range)
├── web/                        # Web 工作站 (本地展示层)
│   ├── app.py                  # FastAPI 入口 (页面路由 + JSON API)
│   ├── i18n.py                 # 内联 i18n: L()/T() + lang cookie
│   ├── static/ssd.css          # 设计令牌与共享组件
│   └── templates/              # base / dashboard / episodes / episode_detail / report
├── showcase/                   # 单页案例演示 (自包含, 无需服务)
│   ├── index.html              # 个体化健康筛查报告
│   └── README.md
├── scripts/                    # load_portable_v2.py, team_data_demo.py
├── tests/                      # 测试 (188 cases, pytest + pytest-asyncio)
│   ├── test_report_agent.py    # 29 tests — 规范区块 / 证据链 / 双语 / 质量去重 / 决策路径
│   ├── test_report_data.py     # 8 tests — 时间窗选取 / 无行数上限 / JSON 解码
│   ├── test_nurse_agent.py     # 规则引擎 + z-score + 模态冲突
│   ├── test_triage_roundtrip.py# TriageDecision 落库往返 + 证据链
│   ├── test_tools.py           # 工具 schema + registry
│   ├── test_web_e2e.py         # Playwright 端到端 (需浏览器)
│   └── ...                     # baseline / fusion / replay / db / event_bus 等
├── main.py                     # 集成入口 (init_db → EventBus → Agents → FastAPI)
├── config.yaml                 # 配置 (LLM/DB/Nurse/Diagnosis/Web)
├── requirements.txt
├── Makefile                    # install / run / test / clean
└── README.md
```

---

## Tech Stack

- **Python 3.12** — Bare ReAct loop, no LangChain / LangGraph dependency
- **FastAPI + Jinja2** — Web doctor workstation
- **SQLite (WAL mode)** — Local persistent storage with `PRAGMA foreign_keys`
- **DeepSeek V4 API** — LLM provider via an OpenAI-compatible HTTP endpoint
- **pytest + pytest-asyncio** — 188 tests across 19 test files
- **Playwright** — Optional browser end-to-end suite

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No LangChain / LangGraph** | Raw ReAct loop with a `@tool` decorator + ToolRegistry. Full control over prompt structure and tool dispatch. |
| **Async event chain** | `NurseAgent.evaluate()` → `EventBus.publish()` → `DiagnosisAgent.handle_event()` is fully async. Sync and async subscribers both supported via `iscoroutinefunction` detection. |
| **Deterministic confidence** | Three heuristics: data existence + physiological plausibility + sensor contact quality. No ML model — confidence is an explainable signal, not a black box. |
| **SQLite JSON columns** | `evidence`, `decision`, `action`, `audit` stored as TEXT (JSON), parsed to dict at the web layer. |
| **Single-resident mode** | `resident_id = "resident_01"` for the paper demo. The schema supports multi-resident via indexed columns. |
| **Deterministic replay** | Seeded `random.Random(42)` for reproducible synthetic data with configurable anomaly injection windows. |
| **No diagnosis language** | Output is care support and risk triage. `safety_boundary` is forced to `care_support_only`; the system never claims a complete NEWS2 score. |

---

## Known issues

Ranked by impact. None block a local demo run.

| Severity | Issue | Detail |
|----------|-------|--------|
| **P1** | Test suite is not database-isolated | Seven test modules remove the shared `data/supersense.db` (see the warning under *Run the tests*). Root cause of the `14 failed / 7 errors` baseline, and it destroys loaded demo data. Fix: per-test `tmp_path` DB with connections closed before teardown. |
| **P1** | `test_main_integration.py::test_modality_conflict_reaches_diagnosis` can never pass | It asserts `len(episodes) == 1`, but `storage/db.py` auto-seeds 10 demo episodes into any empty database, so the count is always 11. Independent of database state. Fix: assert on the event id rather than the total, or disable auto-seed under test. |
| **P2** | `main.py` log line renders the tier twice | `print(f"...: L{decision.get('level')}")` yields `LL1`, because `level` is already stored as `L1`. |
| **P2** | `fusion_json` has no dedicated column | `FusionResult` is persisted inside each episode's `evidence.tool_results`. Auditable, but not queryable per modality. |
| **P2** | Live LLM path is unverified | The `?llm=1` route and the real `DeepSeekProvider` path are exercised only with `MockProvider`; no test runs against a live endpoint. |

## 悬置事项 / Open items

| 事项 | 状态 | 说明 |
|------|------|------|
| per-modality `modalities_json` | 已落地 | `sensing_windows.modalities_json` 已建成，由 `scripts/load_portable_v2.py` 写入每模态估计值（`hr_wifi` / `hr_mm` / `rr_wifi` / `rr_mm`）。 |
| `fusion_json` 独立列 | **未落地** | 见上方 Known issues。 |

---

## Data Privacy & MCP Vision

All sensing data, baselines, episode records, and audit logs are stored **locally** on the edge device. External systems do not directly retrieve raw data. When an external OpenAI-compatible model is selected, the Diagnosis Agent sends only a **minimized structured event context** — raw streams never leave the host. A local model can replace that endpoint without changing the Agent contract. The same contract carries access control, retention policy, encryption, and explicit resident consent.

The internal conversational query interface already uses the same retrieval tools as the Agent layer — registering them as an MCP server is only a protocol-level encapsulation step for future authorization-gated access.

<details>
<summary><b>中文说明 — 数据隐私与 MCP</b></summary>

所有传感数据、个人基线、episode 记录与审计日志都**本地存储**在边缘设备上，外部系统不直接获取原始数据。选用外部 OpenAI 兼容模型时，诊断 Agent 只发送**最小化的结构化事件上下文**，原始数据流不出本机；本地模型可在不改动 Agent 契约的前提下替换该端点。同一契约承载访问控制、留存策略、加密与明确的居民知情同意。

内部对话式查询接口已复用 Agent 层的同一套检索工具 —— 将其注册为 MCP server 只是协议层的封装工作，用于未来的授权访问。

</details>
