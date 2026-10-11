# 项目目录说明与 GitHub 整理指南

这份说明对应当前 FastAPI + 原生单页 Console 工程。发布源码的边界是 `SixChannel_Agent_Console/`，其父目录的历史资料、压缩包、迁移副本和 `Console版本测试文件/` 都不属于该仓库。

## 1. 从一次会诊理解工程

```text
浏览器：backend/app/static/
    ↓ HTTP 病例解析 / 创建会诊 / 补证 / 审核；SSE 进度
API：backend/app/api/v1/
    ↓ 校验请求，交给应用服务
应用服务：backend/app/services/
    ↓ 协调会诊、存储、后台任务与图执行
工作流：backend/app/graph/ → agents/ → nodes/
    ↓ 结构化事实、八纲、六经、病证关系、方证、鉴别、补证和 Top-K
知识边界：backend/app/knowledge/
    ↓ 原始知识读取、精确查证与检索策略
检索适配：backend/app/rag/ → knowledge/v1.2/ 与 Milvus
持久化：backend/app/persistence/ → PostgreSQL / checkpoint
```

`schemas/` 定义这些模块之间交换的数据。`core/` 统一配置、依赖注入和模型端口。前端不计算医学规则，KnowledgeService 是医学知识访问边界，医生审核是最终决策入口。

## 2. 根目录：哪些保留、哪些可清理

| 目录 | 当前内容与用途 | 处理建议 | 上传 GitHub |
| --- | --- | --- | --- |
| `backend/` | FastAPI 服务、医学工作流、原生页面、数据库迁移、有效测试、Python 依赖 | 必须保留 | 是；排除内部缓存 |
| `knowledge/` | 五类经过整理的运行知识 JSONL，供 KnowledgeService 加载 | 必须保留，不能与生成的向量索引混淆 | 授权确认后随源码提供 |
| `scripts/` | 启动、索引管理、校验、验收与开发工具 | 保留有效工具，见下表 | 是 |
| `eval/` | 检索评测、系统评测、工作流验收与固定样本 | 保留，便于别人复现工程结果 | 是 |
| `docs/` | 项目结构及维护说明；本文件位于此目录 | 保留读者需要的文档 | 是 |
| `data/` | 本地 Lite 索引、SQLite checkpoint 和历史运行产物 | 保留运行数据；移除已归档测试输出 | 否，已忽略 |
| `.vscode/` | 本机编辑器设置，与后端运行无关 | 保留现有个人设置；不属于运行必需项 | 否，已忽略 |
| `.mypy_cache/` | 类型检查缓存，加快下一次检查 | 可删除，会自动重建 | 否 |
| `.ruff_cache/` | Ruff 缓存 | 可删除，会自动重建 | 否 |
| `.audit-tests/` | 之前审查时产生的临时测试目录 | 可删除；权限受阻时不会影响启动 | 否 |
| `.local-test-temp/` | 之前本地测试临时目录，存在时可清理 | 可删除 | 否 |
| 各处 `__pycache__/`、`.pytest_cache/` | Python 字节码与 pytest 缓存 | 可删除，运行或测试后会再出现 | 否 |

“目录干净”不要求本地永远没有缓存，而是让提交内容只有有效源码和文档。服务器正在运行时不要删除运行数据库或更改其权限。

## 3. backend 内部职责

| 路径 | 用途与边界 |
| --- | --- |
| `backend/app/main.py` | 应用入口、lifespan、静态页面与路由装配 |
| `backend/app/core/` | 配置读取、基础设施装配、LLM 接口、日志与异常；外部服务通过这里注入 |
| `backend/app/api/v1/` | clinical、consultation、knowledge、reasoning、review、stream、health 等 HTTP 接口，不编写医学判断 |
| `backend/app/services/` | 会诊、推理、知识查询、审核、SSE 与任务恢复的应用服务 |
| `backend/app/graph/` | LangGraph 图、节点顺序、状态路由与恢复 |
| `backend/app/agents/` | 六经推理与独立知识问答的编排入口 |
| `backend/app/nodes/` | 解析、八纲、候选、时间证据、关系、评分、鉴别、补证、路径摘要等具体步骤 |
| `backend/app/knowledge/` | 原始知识加载、精确查询、索引身份与统一 KnowledgeService |
| `backend/app/rag/` | BM25、Embedding、向量检索、RRF 融合、重排、证据分级及 Remote/Lite 适配 |
| `backend/app/persistence/` | 业务 ORM、Repository、任务协调与 checkpoint；SQLite 图状态不能替代业务 PostgreSQL |
| `backend/app/schemas/` | 请求、响应、事实、证据、候选、图状态与索引报告的数据契约 |
| `backend/app/prompts/` | 已实现模型节点使用的提示词；属于运行资源，不能作为普通说明文档删除 |
| `backend/app/observability/` | 公开进度与模型调用阶段记录，不保存隐藏推理 |
| `backend/app/safety/` | 目前只有包初始化文件，预留安全模块目录；安全阶段仍后置，不能解释为已完成 |
| `backend/app/static/` | `dashboard.html`、`console.js`、`labels.js`、`favicon.svg`；整个 Console 展示层，无独立 React 工程 |
| `backend/tests/unit/` | 算法、配置、状态、适配器与启动兜底等小范围回归 |
| `backend/tests/integration/` | 跨模块图流程、真实 Lite、checkpoint、补证和任务恢复验证 |
| `backend/tests/api/` | HTTP 行为、状态码、响应契约和页面入口验证 |
| `backend/alembic/` | 业务 PostgreSQL 版本迁移；`0001`、`0002` 是迁移历史，不应删除或随意改写 |
| `backend/alembic.ini` | Alembic 路径与入口配置 |
| `backend/pyproject.toml` | Python 依赖、开发工具及格式、类型、测试配置 |
| `backend/Dockerfile` | FastAPI 容器构建入口 |
| 各处 `__init__.py` | Python 包标记；内容为空也有结构作用，不能用“空文件”规则统一删除 |

`safety/` 的预留目录可暂时保持，避免为了表面整洁打破既有目录契约。真正删除模块前，应同时核对导入、容器构建、包资源和测试引用。

## 4. scripts 与 eval

| 文件 | 用途 | 是否运行必需 |
| --- | --- | --- |
| `scripts/start_console.py` | 常规启动，选用 `.env` / 模板；Windows 使用兼容 psycopg 的事件循环 | 当前推荐启动入口 |
| `scripts/index_knowledge.py` | 版本化索引管理、试运行和显式重建；Lite 自动补建后仍用于 Remote 运维和受控重建 | 常规 Lite 启动无需手动执行，但应保留 |
| `scripts/validate_knowledge.py` | 校验五类知识结构与来源 | 工程质量工具 |
| `scripts/bootstrap_dev.py` | 环境依赖检查，可选探测向量后端 | 开发辅助 |
| `scripts/engineering_acceptance.py` | 汇总 11 项工程检查、退出码、日志和报告 | 完整验收入口 |
| `scripts/console_e2e.cjs` | Playwright 原生页面流程回归；需要 Node | 浏览器测试工具 |
| `scripts/run_e2e_backend.py` | 为浏览器回归启动隔离后端，不使用业务数据库 | E2E 依赖 |
| `scripts/ci.ps1` | Windows 验收命令包装 | 开发辅助，不是已配置的 GitHub Actions |
| `scripts/verify_real_hybrid.py` | 配置的真实检索服务联调 | 可选在线验证，不属于默认验收 |
| `scripts/verify_real_reranker.py` | 多候选重排模型联调 | 可选在线验证，不属于默认验收 |
| `eval/retrieval_eval.py` | 检索工程指标 | 评测源代码 |
| `eval/system_eval.py` | 解析、极性与已具备标注的指标评测 | 评测源代码 |
| `eval/workflow_acceptance.py` | 纵向工作流验收 | 评测源代码 |
| `eval/datasets/engineering_cases.jsonl` | 6 个固定工程回归病例 | 可复现样本，不代表独立临床验证 |

测试源代码和测试结果不是同一类文件。保留上述源代码，历史 `.log`、报告、截图和临时数据库放到仓库之外即可。`verify_real_*` 仍有联调用途，不因平时不运行就判定无用。

## 5. knowledge 与 data 的区别

`knowledge/v1.2/` 是原始运行知识：经典条文 398 条、六经方证、鉴别规则 53 条、范中林医案和核心框架，共五个 JSONL。

`data/milvus_lite.db/` 是由知识与 Embedding 服务生成的本地向量数据库，当前 milvus-lite 版本把它保存为目录。名称带 `.db` 不代表可当作普通文件删除。删掉它会丢失索引，hybrid/dense 下下次启动可能重新调用 Embedding 建库。

`data/langgraph_checkpoints.sqlite3` 及 `-wal`、`-shm` 是 SQLite 图状态和事务相关文件。不要在运行中单独删 WAL/SHM。即使当前部署使用 PostgreSQL，也保留这些历史数据，待确认不需要图恢复后再单独决定。

`data/s7_acceptance/`、`data/console-e2e/` 是历史测试输出，已经有外部归档。本次核验 36 个历史报告文件和 1 张截图后清理：旧截图目录已移除，旧验收目录的删除在受保护 pytest 临时目录处受阻，可能仍有残留。`.audit-tests/` 也受权限限制未能删除；二者均不参与运行且已忽略。详细记录在父目录归档的 `release-cleanup.json`、`cleanup.json`。新报告默认写到父目录 `Console版本测试文件/`。

## 6. 根目录文件

| 文件 | 用途 | 建议 |
| --- | --- | --- |
| `README.md` | GitHub 首页：功能、安装、配置、运行、测试和已知边界 | 保留 |
| `.env` | 本机真实配置，可能含密钥、数据库凭据 | 本地保留；不上传、不复制进测试归档 |
| `.env.example` | 可分发配置模板、占位符和注释 | 保留并上传 |
| `.gitignore` | 限定 Git 提交边界，排除配置、缓存、运行数据 | 保留 |
| `.dockerignore` | 排除 Docker 构建不需要的本机文件 | 保留；与 Git 忽略规则职责不同 |
| `docker-compose.yml` | PostgreSQL、Milvus 等基础设施与后端容器装配 | 保留 |
| `Makefile` | 开发、测试、格式、类型和索引命令快捷入口 | 可不使用，保留不影响运行 |
| `start_console_windows.bat` | Windows 双击启动入口 | 保留 |
| `AGENTS.md` | AI 协作时的仓库开发约束 | 保留，不是测试报告 |
| `CODEX_DEVELOPMENT_STANDARD_V1.1.md` | 项目工程规范，当前 AGENTS 与贡献说明引用 | 保留；不属于可删的旧测试文档 |

## 7. GitHub 发布边界

建议仓库包含 `backend/`、`knowledge/`、`scripts/`、`eval/`、`docs/`、README、配置模板和构建配置。真实 `.env`、`data/`、`.vscode/`、缓存、本机依赖、父目录归档不提交。

本目录尚未初始化 Git，也没有在此次整理中推送或创建远程仓库。正式上传时应在 Console 项目内建立 Git 仓库，避免在上一级混合资料目录执行提交。

目前没有 `LICENSE`。公开分发前需要由项目所有者决定代码许可证，并核实知识数据及历史医案的再分发授权；代码许可证不能自动授权第三方内容。此处不替用户选择许可证。

当前验收边界也应公开保留：历史全量工程检查为 10/11，通过记录中的 pytest 为 364 项；后续启动回归与自动索引回归已通过。浏览器 E2E 尚有移动视口横向溢出失败，独立临床标注与准确性验收尚未完成。上传 GitHub 不等于确认达到生产部署条件。
