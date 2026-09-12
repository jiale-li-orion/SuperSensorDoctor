# Showcase — 个体化健康筛查报告

A single self-contained HTML page that presents the SuperSenseDoctor screening
pipeline as a **case narrative** for the published paper. It is the
presentation companion to the engineering artifact in the parent repository:
where the FastAPI workstation is the operating surface, this is the story.

**Open it directly — no server, no build step, no network:**

```bash
# Linux / WSL
xdg-open showcase/index.html

# macOS
open showcase/index.html

# Windows PowerShell
start showcase\index.html
```

Or serve it if you prefer a URL:

```bash
python3 -m http.server 8000 --directory showcase
# → http://127.0.0.1:8000
```

Everything — layout, palette, typography, the trend chart (inline SVG), and
both languages — lives in this one file. It has no external assets, no fonts
to download, and no JavaScript dependencies.

### Language

The page is bilingual. The **EN / 中文** button in the navigation switches the
entire document, including `<title>` and `<html lang>`:

| How | Effect |
|-----|--------|
| Click **EN / 中文** | Toggles the page; the choice is remembered in `localStorage` |
| `?lang=en` / `?lang=zh` | Forces a language for that load, and then sticks |

Translation is declarative: every translatable node carries
`data-zh` / `data-en`, and the switcher writes the active one into
`textContent`. Nothing is fetched and no key catalogue is needed, so the file
stays a single artifact you can email or drop on a USB stick.

---

## What it argues

The page walks one fictional resident through two scenes, so the architecture
is legible without reading the code:

| Scene | Claim |
|-------|-------|
| **Hero** | 7 days of continuous observation, 3 contactless sources, 2,686 valid state points, 6 screening directions |
| **`#observation`** | How a weak signal becomes a joint deviation: sleep breaks first (Day 1–3), daytime activity falls 26% (Day 4–5), then respiration and temperature cross the personal threshold together (Day 6–7) |
| **`#screening`** | One primary direction (acute respiratory infection, 78%) is explained in full on the left — why it ranks first, what still has to be ruled out, and what to do — while the right column keeps the whole probability spectrum visible |


**Fictional case:** 林国安 · 74 · living alone · hypertension history · 28-day
personal baseline · Case SSD-CR-01 · 2026.09.05–09.11.

> The case is synthetic. Every number on the page is either derived from that
> synthetic case or quoted from the published paper — never from a real
> resident.

---

## Design notes

The visual language is derived from the paper's own figures rather than
imported from a UI kit.

- **Colour carries meaning.** One accent per architectural layer, matching
  Figure 1: blue = sensing and state, orange = agent and action, green =
  quality and coverage, red = alert.
- **Numbers are set in a sans face with tabular figures** (`font-variant-numeric: tabular-nums`)
  so digits align in columns and read as data, not as decoration. Headings keep
  a CJK-native serif for editorial contrast.
- **One type floor: 13px.** No informational text renders smaller, at any
  viewport.
- **No horizontal overflow** at 390px.
- **The language toggle survives on mobile.** Below 660px the section anchors
  are dropped but the toggle is kept, centred in the nav — it is the one
  control that must always be reachable.

---

## Provenance

This page was selected from a set of design variants produced in a
brainstorming session, and then rebuilt here: the type scale was raised, the
display numbers moved off the serif face to tabular sans figures, the plain
cards (timeline phases, metric strip, why-strip) were given the same
pill/accent-bar/tinted-zone treatment as the case card, and three layout defects
were fixed — a clipped section anchor under the sticky nav, a 21px mobile
overflow caused by the unbreakable hero wordmark, and a wrapping KPI value.

The original variant set is not part of the repository; only this page is.

---

## 中文说明

本目录是论文的**演示用单页**：把 SuperSenseDoctor 的筛查流程讲成一个案例故事。
父目录里的 FastAPI 工作站是**操作界面**，这一页是**叙事界面**。

**直接打开即可**，无需服务器、无需构建、不依赖网络：

```bash
# Linux / WSL
xdg-open showcase/index.html
# Windows PowerShell
start showcase\index.html
```

**双语切换**：导航栏的 **EN / 中文** 按钮切换整页，连 `<title>` 和 `<html lang>`
一起换；选择记在 `localStorage`，也支持 `?lang=en` / `?lang=zh` 强制指定。翻译是声明式的
——每个可译节点带 `data-zh` / `data-en`，切换时写入 `textContent`，不发起任何请求、也不
需要词典文件，页面始终是单个可离线分发的文件。

样式、配色、排版、趋势图（内联 SVG）、双语内容全部在这一个文件里，无外部资源、
无字体下载、无 JS 依赖。

**页面主张**：Hero 交代 7 日观察 / 3 类非接触传感 / 2,686 个有效状态点 / 6 个筛查方向；
`#observation` 说明弱信号如何演化为联合偏离（睡眠先变 → 活动下降 26% → 呼吸与体温同步越阈）；
`#screening` 左侧完整解释首要方向（急性呼吸道感染 78%）——为什么优先、仍需排除什么、该怎么
做——右侧保留完整概率谱；

**案例为虚构**：林国安 · 74 岁 · 独居 · 高血压史 · 28 天个人基线 · Case SSD-CR-01。
页面上每个数字要么来自该虚构案例，要么引用自已发表论文，**不来自任何真实居民**。

**设计说明**：配色语义取自论文图 1 的分层配色（蓝=感知与状态、橙=Agent 与行动、
绿=质量与覆盖、红=告警）；数字改用**等宽数位（tabular-nums）的无衬线字体**，标题保留
CJK 衬线形成对比；**最小字号 13px**；390px 视口下无横向溢出。
