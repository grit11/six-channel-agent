# SixChannel Agent Console V1.2

面向医生的六经辨证临床辅助系统。FastAPI 提供 API 与原生 Console，LangGraph 执行五个自动阶段，并在第六阶段等待医生补证或审核。默认展示 Top-3 候选、支持证据、反证、缺失信息及原始来源；排序分是工程比较值，不是诊断概率。

## 文档入口

- [六阶段架构、鉴别与评分](docs/REASONING_V1_2_SIX_STAGE.md)
- [目录与工具职责](docs/PROJECT_STRUCTURE.md)
- [V1.2 发布与已知限制](docs/RELEASE_V1_2.md)
- [Benchmark48 与 RAGAs 测评](eval/README_EVALUATION.md)
- [仓库协作约束](AGENTS.md) · [工程开发规范](CODEX_DEVELOPMENT_STANDARD_V1.1.md)

## 六阶段流程

| 阶段 | 稳定标识 | 主要职责 |
| --- | --- | --- |
| S1 | `clinical_parser` | 结构化事实、阳性/阴性/未知、时间、冲突；医生编辑优先 |
| S2 | `six_channel_screen` | 八纲与六经宽候选，保留多候选 |
| S3 | `formula_branch_retrieval` | 检索方证及单条经典分支，区分经典与历史归属 |
| S4 | `differential_verification` | 病证关系、双方来源条件、鉴别和反证 |
| S5 | `evidence_topk` | 来源核验、公开分项评分、Top-K、缺口与最多三条有效补问 |
| S6 | `doctor_review` | 主图人工中断，医生补证、采用、驳回、修改与记录 |

八纲、评分、时间和补证模块是阶段内部组件，不是额外 Graph 阶段。历史医案只供知识中心主动查询，不进入自动排序。新协议为 `1.2-six-stage`；旧九阶段活跃 Checkpoint 不直接恢复为新图。

## 安装与启动

在项目根目录执行。本机优先使用已有 Conda `langgraph` 环境，不创建另一个环境。

```powershell
conda run -n langgraph python --version
conda run -n langgraph python -m pip install -e './backend[dev,eval]'
# 仅首次配置且 .env 不存在时执行；已有文件不要覆盖。
Copy-Item -LiteralPath .env.example -Destination .env
conda run --no-capture-output -n langgraph python scripts/start_console.py
```

访问 `http://127.0.0.1:8000/dashboard`，API 文档为 `/docs`。Windows 可双击 `start_console_windows.bat`；推荐入口显式使用 PostgreSQL 异步客户端兼容的事件循环，并从本项目根目录加载 `.env`。

`.env.example` 是可运行的规则/BM25开发模板。配置模型地址和密钥不会自动开启模型模式；需分别设置：

```dotenv
CLINICAL_PARSER_MODE=llm
REASONING_MODE=llm
RETRIEVAL_MODE=hybrid
RERANKER_ENABLED=true
```

LLM、Embedding 与 Reranker 的 Provider、URL、模型和凭据分别由 `.env` 注入。模型请求失败明确返回错误，不用固定候选替代。规则模式只识别已支持的词项，医生应在启动辨证前核对结构化事实。

## 本地 PostgreSQL 与 Checkpoint

业务会话存储和图 Checkpoint 是两个独立配置。使用自建本地数据库时按实际账号填写，下面的密码只是占位符：

```dotenv
DATABASE_ENABLED=true
CONSULTATION_STORAGE=postgres
DATABASE_URL=postgresql+asyncpg://six_channel:CHANGE_ME@127.0.0.1:5432/six_channel
CHECKPOINT_BACKEND=auto
LANGGRAPH_DATABASE_URL=postgresql://six_channel:CHANGE_ME@127.0.0.1:5432/six_channel?sslmode=disable
CHECKPOINT_FALLBACK_TO_SQLITE=true
```

首次接入空业务数据库须显式迁移；应用不会自动创建业务表：

```powershell
conda run -n langgraph python -m alembic -c backend/alembic.ini upgrade head
```

业务迁移头为 `0002_reliable_runtime`。Checkpoint 表由官方 Saver 初始化。开发模式下业务 PostgreSQL 故障可降为进程内存，会诊记录会随进程消失；Checkpoint 在 auto 且允许 fallback 时独立降为 SQLite。生产模式不以开发兜底替代业务持久化。

`/api/v1/health` 报告实际 `business_database`、`checkpoint_backend` 和 `vector_backend`。Lite 导致 degraded 不等于 PostgreSQL 失败。保存 `.env` 后启动进程需重新加载配置；确认实际状态以健康接口为准。

## 向量与知识

五类原始 V1.2 数据：398 条经典、128 个方证、53 条鉴别、69 条历史医案、11 条核心框架。`knowledge/derived/fpeu_v12.jsonl` 是带原文来源和审校状态的方证分支索引，不改变原始五库。

`MILVUS_MODE=auto` 优先 Remote，失败且允许 fallback 时使用 Lite。Hybrid/Dense 模式需有效 Embedding 与匹配索引；Lite 缺失索引可自动构建，会产生在线调用，完整索引复用。Remote 需预建索引：

```powershell
conda run -n langgraph python scripts/validate_knowledge.py
conda run -n langgraph python scripts/index_knowledge.py --dry-run
# 配置和目标确认后执行真实索引建设。
conda run -n langgraph python scripts/index_knowledge.py
```

不要删除运行中的 `data/milvus_lite.db/` 或 SQLite 的 WAL/SHM 文件。评测与 Console 不应并发占用同一 Lite 索引。

## 工程验证

```powershell
conda run -n langgraph python scripts/engineering_acceptance.py
```

入口执行 Ruff、Black、mypy、pytest、知识/检索/离线病例/工作流验证、两项 JavaScript 语法检查与 Playwright E2E，共 11 项。报告默认写至项目父目录 `Console版本测试文件/`，可用 `--output-dir` 修改。`--only` 是定向定位，未执行项使整体报告保持未通过。

浏览器回归需要 Node、Playwright 和浏览器；可通过 `CONSOLE_PLAYWRIGHT_MODULES` 指定已有模块，通过 `CONSOLE_BROWSER_CHANNEL=msedge` 或 `CONSOLE_BROWSER_EXECUTABLE` 指定浏览器。它启动隔离服务，不连接用户会诊数据库。

在线 48 条与 RAGAs 命令、消融、续跑和指标分母见[测评说明](eval/README_EVALUATION.md)。48 条数据是研发合成银标，不代表专家独立临床验证。发布整理不会自动运行付费全量测评。

## 可选真实联调

以下命令会调用配置的服务，不属于默认离线验收：

```powershell
conda run -n langgraph python scripts/verify_real_hybrid.py
conda run -n langgraph python scripts/verify_real_hybrid.py --probe-reranker
```

后者仅验证配置的一个重排器，通过现有工厂注入，不枚举或自动更换其他模型。

## Docker

Compose 提供 PostgreSQL、etcd、MinIO、Milvus 和 FastAPI。填写 `.env` 中基础设施凭据，注意宿主已有 PostgreSQL 与 Compose 默认端口冲突。容器数据库 URL 应使用 Compose 服务名而非宿主 `127.0.0.1`。

```powershell
docker compose up -d postgres etcd minio milvus
# 迁移命令读取的 DATABASE_URL 必须指向目标数据库。
conda run -n langgraph python -m alembic -c backend/alembic.ini upgrade head
docker compose --profile app up -d --build backend
```

真实 Docker/Remote 部署、身份权限、隐私及安全分诊需另行验收。仓库未声明开源许可证；数据来源与审核状态保留在知识记录内。

## 发布边界

Git 包含代码、迁移、V1.2 知识、派生索引、评测数据、测试和当前文档。`.env`、数据库、向量索引、缓存、报告与本机编辑器设置仅留本地。当前已知解析、评分饱和与同分排序限制详见[发布说明](docs/RELEASE_V1_2.md)，工程通过不等于临床准确性通过。
