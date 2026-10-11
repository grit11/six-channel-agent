# 六经辨证 Agent Console

面向医生的六经辨证临床辅助工作台。使用 **FastAPI + 原生 HTML/CSS/JavaScript 单页界面**，提供病例录入、结构化证据、Top-K 辨证路径、来源查证、补证和医生审核。

Console 是既有“六经辨证临床辅助系统”的独立副本，替换展示层并沿用真实后端。运行页面无需 React、Vite、Node.js、npm 或 CDN；Node.js 和 Playwright 仅用于浏览器测试。

**当前状态：研发与功能展示阶段。** 最近一次历史全量验收为 10/11 项通过，pytest 364 项通过；后续浏览器回归修复尚未获得最终通过记录。工程验收不等同于临床准确性验证，六经与方证 Top-3 的专家标注评测仍待完成。

## 目录

- [功能](#功能)
- [技术架构](#技术架构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置](#配置)
- [使用流程](#使用流程)
- [项目结构](#项目结构)
- [测试与评测](#测试与评测)
- [部署](#部署)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证与数据来源](#许可证与数据来源)

## 功能

- **病例结构化**：解析自由文本，核对症状、脉象及阳性、阴性、未知三值事实。
- **辨证工作流**：八纲分析 → 六经候选 → 病证关系 → 方证候选 → 鉴别与证据缺口 → Top-K 路径。
- **证据溯源**：通过 KnowledgeService 查询经典条文、方证、鉴别规则、历史医案和核心框架，查看来源详情。
- **补证与状态恢复**：补充信息后在同一会诊线程继续推理；SSE 显示进度，权威会诊快照支持页面刷新恢复。
- **医生审核**：审核当前轮结果并记录决定；系统保留支持证据、反证、缺失证据和来源。
- **知识查询与会诊历史**：在统一工作台中查询知识、浏览当前存储支持的会诊记录。
- **可替换基础设施**：通过配置和适配器接入 LLM、Embedding、Reranker、Milvus、PostgreSQL 和 LangGraph checkpoint。

候选排序分用于比较路径，不代表临床概率。未知信息不会自动转为阴性，最终决策由医生审核。

## 技术架构

```text
原生 Console
    │ HTTP / SSE
FastAPI → ConsultationService → LangGraph / Domain
                                  ├─ KnowledgeService → 检索 / 证据来源
                                  ├─ 模型适配器 → LLM / Embedding / Reranker
                                  └─ Repository / Checkpointer → 持久化
```

医学逻辑位于后端业务层，前端负责展示与交互。SSE 传递进度提示，最终结果以会诊快照为准。外部服务不可用时明确返回失败，不生成固定演示候选。

## 环境要求

| 环境 | 要求 | 用途 |
| --- | --- | --- |
| Python | 3.11 及以上；本机已验证环境为 Conda `langgraph` / Python 3.13 | 后端运行与 Python 测试 |
| 浏览器 | 现代 Edge / Chrome 等 | 打开工作台 |
| Node.js | 支持内置 `fetch`，建议 20 及以上 | 仅语法检查与浏览器回归 |
| Playwright | Node 模块；可使用本机 Edge / Chrome | 仅浏览器回归 |
| PostgreSQL / Remote Milvus / Docker | 按需配置 | 持久化与远程基础设施 |

Python 依赖以 [backend/pyproject.toml](backend/pyproject.toml) 为准。本机开发复用已有 `langgraph` 环境。

## 快速开始

以下命令均在 `SixChannel_Agent_Console` 项目根目录执行。

### 1. 安装依赖

```powershell
conda activate langgraph
# 已安装项目依赖时可跳过这一行
python -m pip install -e "./backend[dev]"
```

### 2. 准备配置

```powershell
# 仅在不存在 .env 时复制，保留已有配置
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

模板提供离线规则与本地存储基线。在线模型配置默认留空，不需要先配置 API Key。

### 3. 启动

```powershell
python scripts/start_console.py
```

Windows 也可双击 [start_console_windows.bat](start_console_windows.bat)。启动脚本优先读取项目 `.env`，不存在时读取 `.env.example`，监听 `127.0.0.1:8000`。

| 入口 | 地址 |
| --- | --- |
| 工作台 | [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) |
| API 文档 | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| 就绪检查 | [http://127.0.0.1:8000/api/v1/health/ready](http://127.0.0.1:8000/api/v1/health/ready) |
| Console 配置 | [http://127.0.0.1:8000/api/console/config](http://127.0.0.1:8000/api/console/config) |

端口被占用时使用 `python scripts/start_console.py --port 8001`。

## 配置

完整变量及说明见 [.env.example](.env.example)，配置校验入口为 [backend/app/core/config.py](backend/app/core/config.py)。

| 变量 | 本地模板值 | 说明 |
| --- | --- | --- |
| `CLINICAL_PARSER_MODE` | `rules` | 规则解析；`llm` 接入结构化模型 |
| `REASONING_MODE` | `rules` | 确定性工程基线；`llm` 使用模型推理 |
| `KNOWLEDGE_QA_MODE` | `extractive` | 基于来源摘录的知识查询 |
| `RETRIEVAL_MODE` | `bm25` | 本地词法检索；混合检索另行配置模型与索引 |
| `MILVUS_MODE` | `lite` | 本地 Lite；`remote` 或 `auto` 使用远程适配 |
| `MILVUS_LITE_AUTO_INDEX` | `true` | 实际 Lite 的 dense/hybrid 索引缺失时，启动自动建立 |
| `CHECKPOINT_BACKEND` | `sqlite` | 本地图状态；生产可配置 PostgreSQL |
| `DATABASE_ENABLED` | `false` | 本地默认不启用业务 PostgreSQL |
| `CONSULTATION_STORAGE` | `memory` | 开发会诊存储；持久会诊需 `postgres` |

`memory` 模式下服务重启会丢失会诊历史。SQLite checkpoint 保存图状态，不能替代业务数据库，也不能单独提供持久会诊与任务恢复。需要重启恢复时，配置 PostgreSQL 业务存储与 checkpoint，并执行数据库迁移。

模型、向量检索与重排器通过配置注入；切换模型或向量维度后需重新检查索引兼容性。不要将真实密钥、患者资料或数据库密码写入 README、测试样本或提交记录。

使用 Lite（包括远程不可用后自动降级）时，dense/hybrid 缺少当前知识、模型版本或维度对应的索引，启动会通过已配置的 Embedding 服务自动建立五类知识索引，完整发布并校验后才进入就绪状态。首次启动可能较慢并产生 Embedding 调用费用；已有完整索引直接复用，不重复向量化。无需手动执行索引命令。

BM25 不建立向量索引。Remote 保持预建索引要求；可设置 `MILVUS_LITE_AUTO_INDEX=false` 禁用 Lite 自动建立。索引服务或写入失败时明确报错并释放连接，下次启动重试；不会伪造向量或自动改用 BM25。

## 使用流程

1. 录入病例并解析，核对结构化事实及未知项。
2. 创建会诊，观察实时进度并查看 Top-K 路径。
3. 打开来源详情，核对支持证据、反证和证据缺口。
4. 补充区分性信息，等待同一会诊的新轮结果。
5. 医生审核当前版本；按已配置存储查看会诊历史。

## 项目结构

各目录的工程职责、保留与清理建议见 [项目目录说明](docs/PROJECT_STRUCTURE.md)。

```text
SixChannel_Agent_Console/
├── backend/
│   ├── app/                 # API、业务服务、图、知识、检索与持久化
│   │   └── static/          # dashboard.html、console.js、labels.js、favicon.svg
│   ├── tests/              # unit / integration / api，可执行测试源代码
│   ├── alembic/            # 业务数据库迁移
│   ├── pyproject.toml      # Python 依赖与质量工具配置
│   └── Dockerfile
├── knowledge/v1.2/         # 五类 JSONL 知识数据
├── scripts/                # 启动、索引、校验与验收入口
├── eval/                   # 检索、系统评测、工作流验收与样本集
├── docs/                   # 项目目录与维护说明
├── data/                   # 本地运行数据库，不进入版本管理
├── .env.example
├── docker-compose.yml
├── Makefile
├── AGENTS.md
├── CODEX_DEVELOPMENT_STANDARD_V1.1.md
├── start_console_windows.bat
└── README.md

../Console版本测试文件/      # 本地集中归档，不是运行时依赖
├── console版本测试文件.md   # 测试说明、报告索引与整理范围
├── source/                 # 测试代码及评测样本快照
├── reports/                # 历史报告及新验收结果
├── screenshots/            # 单独浏览器回归的截图
├── legacy/                 # 已不适用的旧工具备份
└── manifest.json           # 归档文件 SHA-256 校验清单
```

## 测试与评测

### 全量工程验收

```powershell
conda activate langgraph
$env:PYTHONUTF8 = "1"
python scripts/engineering_acceptance.py
```

执行 Ruff、Black、mypy、pytest、知识校验、检索评测、系统评测、工作流验收、两项 JavaScript 语法检查与 Console E2E，共 11 项。全部通过才返回退出码 0，部分执行、跳过或失败均不算整体通过。

报告默认写入 `../Console版本测试文件/reports/<时间戳_编号>/`，包含 Markdown、JSON、逐项日志、JUnit 和评测结果。全量验收产生的截图保存在对应报告的 `screenshots/`，Windows 测试数据库使用短临时路径，位置记录在 `runtime.txt`。

```powershell
# 自定义测试资料根目录
python scripts/engineering_acceptance.py --output-dir "D:\Console测试资料"
# 仅定位某项；其他检查记为 not_run，退出码仍为非零
python scripts/engineering_acceptance.py --only console_e2e
```

也可通过 `CONSOLE_TEST_ARTIFACTS` 设置资料根目录。有效测试源文件保留项目原路径；归档的 `source/` 是整理时的快照，执行测试请使用项目内入口。

### 单项检查

```powershell
conda run -n langgraph python -m pytest backend/tests -c backend/pyproject.toml -q
conda run -n langgraph python -m ruff check backend scripts eval
conda run -n langgraph python -m black --config backend/pyproject.toml --check backend scripts eval
conda run -n langgraph python -m mypy --config-file backend/pyproject.toml backend/app scripts/engineering_acceptance.py eval/system_eval.py eval/workflow_acceptance.py
node --check backend/app/static/console.js
node --check backend/app/static/labels.js
```

### 浏览器回归

测试需要 Node.js、Playwright 模块和可启动的浏览器。已有模块可通过 `CONSOLE_PLAYWRIGHT_MODULES` 指定其 `node_modules` 目录；没有模块时可按需安装测试工具：

```powershell
npm install --no-save --package-lock=false playwright
conda activate langgraph
$env:E2E_PYTHON = (Get-Command python).Source
node scripts/console_e2e.cjs
```

Windows 优先使用已安装 Edge / Chrome。可用 `CONSOLE_BROWSER_CHANNEL=msedge` 或 `CONSOLE_BROWSER_EXECUTABLE` 指定浏览器；没有系统浏览器时可按需执行 `npx playwright install chromium`。Node 不在 PATH 时使用其可执行文件绝对路径。

回归覆盖真实会诊、SSE、来源、刷新恢复、补证、医生审核、移动布局与历史入口。看到 `Native Console E2E passed` 且退出码为 0 才确认通过；已有截图不能单独证明完整回归成功。

### 结果与范围

历史全量报告原样保存在本机父目录 `Console版本测试文件/`，不属于分发仓库，也不是运行依赖。2026-10-10 最近一次完整报告记录 pytest 364/364 通过、总计 10/11 项通过；后续本机 E2E 已完成补证与审核，仍因移动视口横向溢出失败，需修复后重新确认。

离线样本集用于工程回归，当前只有 6 个固定病例。缺少专家标注的六经 Top-3、方证 Top-3 和语义忠实度指标标记为 `unavailable`，不能据此宣称达到上线临床精度。真实 PostgreSQL、Remote Milvus、在线模型与部署恢复需分别验收；`verify_real_hybrid.py`、`verify_real_reranker.py` 是可选在线验证，不包含在默认离线验收中。

## 部署

本地展示使用上述启动入口。Docker 配置提供 PostgreSQL、etcd、MinIO、Milvus 和单个 FastAPI 服务，不需要单独部署前端。

接入持久化环境前，先填写 `.env` 中的数据库、MinIO 与模型配置，启动基础设施并确认就绪，再使用匹配目标数据库的配置执行 `backend/alembic.ini` 迁移，最后启动应用：

```powershell
docker compose up -d postgres etcd minio milvus
# 此时 .env 中 DATABASE_URL 必须指向待部署的 PostgreSQL
conda run -n langgraph python -m alembic -c backend/alembic.ini upgrade head
docker compose --profile app up -d --build backend
```

配置和恢复步骤以实际部署环境为准。目前尚未确认真实 Docker 部署验收；安全、权限与隐私工作（S6）按既有项目计划后置。

## 常见问题

**启动提示缺少模块**：检查 Python 是否来自 `langgraph`，按 `backend/pyproject.toml` 安装依赖，并在项目根目录执行命令。

**重启后会诊记录消失**：检查 `CONSULTATION_STORAGE`。`memory` 是开发存储，持久会诊需要 PostgreSQL。

**服务就绪但无法生成候选**：查看就绪接口、终端日志和页面错误；核对当前模式、知识数据、索引和模型配置。

**E2E 缺少 Chromium / Playwright**：使用本机 Edge / Chrome，或显式指定现有 Playwright 模块与浏览器位置，参见浏览器回归步骤。

**Windows 测试目录过长**：优先使用全量验收脚本，其测试数据库使用短临时路径；报告和运行数据分别保存。

**Windows 启动报 psycopg / ProactorEventLoop 错误**：使用 `python scripts/start_console.py`；该入口显式使用兼容 PostgreSQL 异步连接的 Selector 循环。直接使用 Uvicorn CLI 的默认循环可能再次触发此错误。修复循环后若仍报 PostgreSQL 不可用，应核对数据库连通性与配置；强制 `postgres` 模式不会静默降级。

## 贡献指南

修改前阅读 [AGENTS.md](AGENTS.md) 与 [开发规范](CODEX_DEVELOPMENT_STANDARD_V1.1.md)。保持中文职责说明、类型标注、Black 行宽 100 和架构边界：医学知识访问通过 KnowledgeService，前端不复制医学规则，未知信息不当作阴性。

提交问题时提供复现步骤、脱敏输入、运行模式和相关日志。提交变更时说明行为变化和验证结果；涉及 API、图或检索时运行对应测试，涉及关键用户流程时运行浏览器回归。

## 许可证与数据来源

仓库目前未声明开源许可证，尚无 `LICENSE` 文件；公开分发前需确定代码和数据的授权范围。本 README 不授予额外使用或再分发权利。

`knowledge/v1.2/` 包含经典条文、方证、鉴别规则、范中林医案和核心框架五类数据。数据来源及整理版本以各 JSONL 的来源字段和项目资料为准，引用时保留条文编号、来源与版本信息。
