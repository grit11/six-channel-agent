# 六经辨证临床辅助系统工程规范 V1.1

本规范约束产品 V1.2 的工程实现。工程规范版本与运行协议版本独立。当前目录与工作流以 [V1.2 架构](docs/REASONING_V1_2_SIX_STAGE.md)和 [项目结构](docs/PROJECT_STRUCTURE.md)为依据，不再采用旧 React/九阶段文件契约。

## 1. 定位与开发流程

医生侧临床辅助，输出待审核 Top-K，不自动诊疗或处方。知识访问、模型适配、公开状态和医生决策职责必须分离。

开工先读 AGENTS、规范、相关架构/协议、知识结构、现有接口和测试。冲突先指出，用户最新明确要求优先，其次工程规范、当前设计、协议、现有代码。复杂功能或跨三个以上模块先说明计划、影响文件、接口/数据变化和风险，每次完成可验证纵向切片。

## 2. 目录与职责

| 层 | 约束 |
| --- | --- |
| `api/v1` | HTTP/SSE 校验与错误转换，不写医学逻辑 |
| `services` | 会诊、审核、任务、事务、恢复协调 |
| `graph` / `agents` / `nodes` | 状态、路由、可审计推理，不能感知 FastAPI 或厂商 SDK |
| `knowledge` | KnowledgeService 唯一医学访问边界，原始来源与派生分支校验 |
| `rag` | 检索、融合、重排、适用性评价，只提供证据 |
| `persistence` | 业务 ORM/Repository 与独立图 Saver，不混用存储 |
| `schemas` | 事实、来源、候选、状态和接口契约 |
| `core` | 配置、端口、依赖、异常、日志 |
| `static` | 原生 Console 展示和交互，不复制后端医学规则 |
| `prompts` | 当前模型节点使用的运行资源，不按旧版本文件统一删除 |

不得随意新增顶层运行目录、移动模块或建立新的医学知识通道。需要目录变化先说明原因、影响、迁移并更新文档。空包标记、迁移历史和共享模块不能以“无内容/旧名称”直接删除。

## 3. 注释、类型与格式

所有新建或实质修改代码顶部中文说明职责、边界和依赖；公开类、函数、方法中文 docstring/JSDoc；复杂私有逻辑、节点、RAG、评分、路由、事务解释“为什么”和业务约束，保留有价值注释。

Python 4 空格、无 Tab、UTF-8、全参数/返回类型、Black 行宽 100、Ruff import/lint、mypy。优先 guard clause 和小职责函数，不堆万能函数。JavaScript 2 空格；若未来增加 TypeScript，strict、避免 any、Prettier/ESLint。组件/页面不散布医学判断与 Provider 细节。

异步 I/O 使用 async；同步文件/SDK通过明确线程适配，不在异步请求直接 requests 或执行阻塞 I/O。客户端资源在 lifespan 或上下文管理器关闭。

## 4. 六经证据约束

顺序保持：事实 → 八纲 → 六经宽候选 → 病证关系 → 方证 → 鉴别/缺口 → Top-K → 医生审核。当前合并成五自动阶段加真实 S6，不通过改 UI 名称伪装图改造。

Unknown 永远不等于 Negative；多候选、多经、多关系，不做 softmax 六分类。每个候选保留支持、反证、未知、来源和版本。默认 Top-3，模型单一结论不得替代医生。来源召回与 rerank 相关性不等于患者适用性；重复来源不累加支持。历史医案不能覆盖经典，不进入自动排名。不同条文分支不交叉累加反证。

复杂自然语言条件无法完整规范化时保留 Unknown，不能截取局部词项冒充完整条件。硬排除须来源明确、原文对齐且经专家批准；普通阴性/禁忌字段不自动成为硬排除规则。禁止伪造知识、Evidence ID、来源或模型 usage。

## 5. 外部能力与配置

LLM、Embedding、Reranker、Milvus、PostgreSQL、PostgresSaver通过接口/工厂注入；禁止在业务层硬编码厂商 SDK、模型、URL、集合、密钥。

配置集中于 `core/config.py`，装配于 `core/dependencies.py`；Reranker/VectorStore适配在 `rag`，数据库与 Saver 在 `persistence`。测试可注入 Fake，业务节点不包含 Remote/Lite 或 Postgres/SQLite 条件。

`.env` 永不提交，模板只含占位符。模型模式、检索模式与服务配置分开；模型故障不能退为固定演示结果。关键配置启动校验、错误明确。

## 6. Milvus

生产 Remote。auto 先探 Remote，不可用且允许 fallback 时使用 `from pymilvus import MilvusClient` 的 Lite 文件模式。契约、metadata、RetrievedDocument保持一致；health 报实际 backend 和 degraded，Remote/Lite均测试。Lite 缺失索引按配置经 KnowledgeService/Embedding补建，完整发布校验后才就绪，失败不得假成功。索引身份包括知识、模型、维度、端点和检索参数，不混用旧索引。

## 7. 数据库、Checkpoint与事务

业务持久化目标：PostgreSQL + SQLAlchemy 2.x async + asyncpg + Alembic + Repository。业务表变化新增清晰迁移及升级/回退测试，应用不隐式创建业务表，不删除既有迁移链。

图状态优先 PostgreSQL，auto 未配置/不可访问且允许 fallback 时用 `AsyncSqliteSaver`。它只替代图状态，不当业务 ORM。Saver由 lifespan一次创建/关闭，每请求复用；health报实际后端，两分支测试。

开发业务 PostgreSQL故障可按已授权策略降进程内存，生产必须持久化；不得对外假装数据已落库。业务服务拥有事务边界，长时图锁不能耗尽业务连接池；任务恢复、幂等、并发版本和终态必须测试。

旧协议活跃线程不直接交新图恢复；新 namespace 隔离，协议/version mismatch明确拒绝。生产迁移先规划数据与恢复策略，不能用开发允许丢弃历史的授权替代后续生产判断。

## 8. API、公开状态与错误

Router保持薄，关键写Schema extra forbid、枚举和中文字段说明，响应不直接暴露 ORM。错误在 core统一转换、带业务码和 request_id，保留异常链；禁止 `except Exception: pass`。

SSE表达公开阶段、来源、补证、审核和失败，不是权威事实源；断线恢复 GET会诊快照。UI区分阳性/阴性/未知，展示支持、反证、缺口、来源、排名和公开分项；同分的展示不能被解释为新增医学证据。审核完成来自业务状态，不能由推理结束事件猜测。

## 9. 日志、隐私与版本

关键日志尽可能带 request_id、consultation_id、thread_id、stage和耗时。禁止密钥、数据库密码、完整患者信息和隐藏 Chain-of-Thought。审计只存结构化事实版本、Evidence ID、Top-K、人工动作、模型/知识/协议/评分版本、错误事件。

供应商 usage缺失保持 unavailable，记录重试和覆盖率，不估造真实账单。临床API与知识QA隔离，模型响应内容不当系统指令执行。身份、权限、隐私、安全分诊尚未验收，不默认宣称生产安全。

## 10. Python环境与质量门槛

本机必须优先已有 Conda `langgraph`，不擅建 venv/Conda或向 base装依赖。先确认版本和解释器，安装以 pyproject声明为准，新增依赖同步配置，不修改其他环境。

```powershell
conda run -n langgraph python --version
conda run -n langgraph python -c "import sys; print(sys.executable)"
conda run -n langgraph python -m ruff check backend scripts eval
conda run -n langgraph python -m black --config backend/pyproject.toml --check backend scripts eval
conda run -n langgraph python -m mypy --config-file backend/pyproject.toml backend/app
conda run -n langgraph python -m pytest backend/tests -c backend/pyproject.toml
```

功能变更必须补必要测试。Graph/API/RAG改动运行对应集成测试；关键用户流程Playwright，原生JS运行Node语法检查。统一入口为 `scripts/engineering_acceptance.py`，测试隔离真实.env和用户数据库。未运行/跳过/失败均不能算整体通过；只执行 `--only`不得假称完整验收。

测试保留三值、解析、评分、关系、鉴别、缺口、KnowledgeService、metadata、配置、Repository、Remote/Lite、Postgres/SQLite、interrupt/resume、审核、恢复、幂等及UI流程。知识评测保留398经典、53鉴别、研发病例。工程测试不等于临床准确性，银标和RAGAs分母、失败与unavailable必须披露。

## 11. Git、发布与完成定义

只改相关文件，不大规模无关重构、格式化、重命名，不自动添加许可证。配置变化更新模板，API变化更新Schema/UI，协议变化更新测试/评测口径。源码与生成报告分开，不提交密钥、数据库、索引、缓存、患者快照或临时输出。

交付必须说明修改、原因、文件、实际检查/通过失败、未解决问题。质量门槛通过、类型/异常/注释/架构完整、来源可追溯且风险披露，才能确认工程任务完成；不得把发布成功写成医学质量达标。
