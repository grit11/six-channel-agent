# 六经辨证临床辅助系统 - Codex 仓库级开发约束

## Console 副本的明确迁移约定（2026-10-10）

用户要求本副本改为 FastAPI + 原生单页 Console；原 React 项目保留在原目录。
原生展示文件位于 `backend/app/static/`，替代 React 前端目录契约。业务模块与医学知识边界沿用原架构。
不再执行 npm 构建；页面语法和关键流程由 Node 语法检查、原生 Playwright 回归验证。
部署只运行 FastAPI；Node/Playwright 仅是工程测试工具。S6 继续按原计划后置。

## Console 测试资料整理约定（2026-10-10）

用户要求重构 README，并将测试资料统一归档到父目录 `Console版本测试文件/`。
`backend/tests/`、`scripts/`、`eval/` 中有效测试源文件保留原路径，以保持导入和执行契约；归档保存可核验副本。
工程验收报告和浏览器截图默认输出到该归档目录，可通过 `CONSOLE_TEST_ARTIFACTS` 指定其他根目录。
历史报告保持原内容；缓存和临时测试文件可清理，真实配置、业务数据库和原 React 项目必须保留。

## Lite 启动索引兜底约定（2026-10-10）

用户明确要求实际向量后端为 Lite 时，启动自动补建缺失的当前知识/模型索引。
默认启用 `MILVUS_LITE_AUTO_INDEX`，包括 auto 模式降级后的 Lite；完整索引复用。
索引仍经 KnowledgeService 和配置化 Embedding 建立，完整发布并校验后才就绪；失败不伪造成功、不切换检索模式。
Remote 保持预建索引策略；BM25 不调用 Embedding。自动索引可调用已配置的在线服务。

## GitHub 发布目录整理约定（2026-10-10）

用户要求清理无用测试资料并提供目录讲解；允许 `docs/` 保存项目结构说明。
有效测试与评测源代码保留；已校验归档的历史测试输出可删除，运行索引、checkpoint、真实配置及编辑器配置保留本地。
仅以 Console 项目作为未来 Git 仓库根目录，父目录测试归档不纳入发布；不得自动选择许可证或上传代码。

> 本文件供 Codex 在仓库根目录自动读取。它是强制性开发约束，不是业务需求说明书。
> 详细规范见根目录 `CODEX_DEVELOPMENT_STANDARD_V1.1.md`。

## 1. 开工前必须阅读

在修改任何代码前，按以下顺序阅读并以其为实现依据：

1. `AGENTS.md`
2. `CODEX_DEVELOPMENT_STANDARD_V1.1.md`
3. `六经辨证临床辅助系统_最终开发文档_V1.0.md`（若仓库内存在）
4. `六经辨证_Reasoning_Protocol_V1.0.md`（若仓库内存在）
5. `knowledge/v1.2/` 下的知识库说明与数据结构

若文档与代码发生冲突，不得静默选择。先指出冲突，再按“用户最新明确要求 > 开发规范 > 最终开发文档 > Reasoning Protocol > 现有代码”的优先级处理。

## 2. 目录是契约

严格按照最终开发文档的目录实现。不得为了方便随意新增顶层目录、重命名既有目录或移动模块。

允许新增的仓库治理文件仅限：`AGENTS.md`、`CODEX_DEVELOPMENT_STANDARD_V1.1.md`、必要的 README/配置模板。

确需调整运行时代码目录时，必须先说明原因、影响范围和迁移方案，再修改文档后实施。

## 3. 代码注释是强制项

所有新建或实质修改的代码文件必须满足：

- 文件开头有中文说明，解释“当前脚本负责什么、边界是什么、依赖谁、不负责什么”。
- 每个公开函数、公开方法、类必须有中文 docstring/JSDoc。
- 复杂私有函数、LangGraph 节点、RAG 融合、评分、状态路由、事务边界等重要位置必须有中文注释。
- 注释解释“为什么”和“业务约束”，不要逐行翻译代码。
- 禁止删除原有有价值中文注释，除非同步替换为更准确的说明。

## 4. 格式与可读性

Python：
- 4 空格缩进，禁止 Tab。
- Black 风格，行宽 100。
- Ruff 管理 lint/import。
- 所有函数参数和返回值必须有类型标注。
- I/O 路径优先 async；禁止在 async 路径直接执行阻塞 I/O。

TypeScript/React：
- 2 空格缩进。
- Prettier + ESLint。
- TypeScript `strict` 开启。
- 禁止大面积使用 `any`。
- React 组件职责单一，页面不直接写 API 细节。

## 5. 架构边界不可破坏

- FastAPI Router 不写六经辨证医学逻辑。
- Agent/Domain 层不得 import FastAPI。
- 前端不得复制后端医学规则。
- `KnowledgeService` 是唯一医学知识访问边界。
- Retriever 只返回证据，不直接下诊断。
- 医生审核是最终决策入口。
- 不记录、不展示模型隐藏 Chain-of-Thought；只保存结构化可审计状态。

## 6. 六经推理不可被简化

必须遵守 Reasoning Protocol：

`Clinical Evidence -> Eight Principles -> Six-Channel Candidates -> Syndrome Relationship -> Formula Candidates -> Differential/Evidence Gap -> Top-K -> Doctor Review`

硬性规则：
- Unknown 永远不等于 Negative。
- 六经允许多候选、多关系，不做 softmax 六分类。
- 每个候选必须保留支持证据、反证、缺失证据和来源。
- 默认输出 Top-3，不以模型单一结论替代医生。

## 7. 外部模型与基础设施必须通过接口/工厂注入

严禁在业务代码中硬编码厂商 SDK、模型名、URL、密钥或集合名。

必须预留并通过配置注入：
- LLM / Chat Model
- Embedding Model
- Reranker Model
- Milvus Vector Store
- PostgreSQL
- LangGraph PostgresSaver

Provider 相关构造统一收口到现有目录中的配置/依赖工厂：
- `backend/app/core/config.py`
- `backend/app/core/dependencies.py`
- `backend/app/rag/reranker.py`
- `backend/app/rag/vector_store.py`
- `backend/app/persistence/database.py`
- `backend/app/persistence/checkpoint.py`

## 8. Milvus：生产 Remote，开发自动 Lite 降级

- 正式部署使用 Docker/Remote Milvus。
- `MILVUS_MODE=auto` 时先探测 Remote；不可用且允许 fallback 时，必须改用 `from pymilvus import MilvusClient` 的 Milvus Lite 文件模式。
- Remote/Lite 选择只允许出现在 `rag/vector_store.py` / 依赖工厂；Node、Agent、KnowledgeService、Router 不得写此分支。
- 不允许节点直接调用 pymilvus。
- Remote/Lite 必须保持同一 VectorStorePort、metadata 和 RetrievedDocument 契约。
- 降级后 `/health` 必须报告 `vector_backend=lite` 和 degraded 状态。
- 两条分支都必须有测试。

## 9. PostgreSQL 与 LangGraph Checkpoint 降级

- 业务数据库生产目标仍是 PostgreSQL：SQLAlchemy 2.x async + asyncpg + Alembic + Repository。
- LangGraph checkpoint 生产优先 PostgreSQL。
- `CHECKPOINT_BACKEND=auto` 时，PostgreSQL 未配置或不可访问且允许 fallback 时，必须使用 `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`。
- `AsyncSqliteSaver` 仅替代 LangGraph checkpoint，不得当作业务 ORM 数据库。
- Checkpointer 必须在应用 lifespan 创建/关闭，不得每请求新建。
- 降级后 `/health` 必须报告 `checkpoint_backend=sqlite`。
- Postgres/SQLite 两条分支必须有测试。

## 10. 错误处理与日志

- 禁止 `except Exception: pass`。
- 对外返回统一业务错误码；对内保留异常链。
- 日志必须带 `request_id`、`consultation_id`、`thread_id`（可用时）。
- 禁止日志输出 API Key、数据库密码、完整患者敏感信息和模型隐藏推理。

## 11. 测试是完成定义的一部分

任何功能变更都必须补测试。

至少运行：
- `ruff check`
- `black --check` 或项目统一格式检查
- `pytest`
- 前端 `npm run lint` / `npm run typecheck` / `npm run test`（若已配置）

涉及 API/Graph/RAG 时还必须运行对应 integration tests。
修改关键用户流程时运行 Playwright E2E。

## 12. Codex 工作方式

复杂功能或跨 3 个以上模块的改动：
1. 先列出实现计划和影响文件。
2. 确认现有接口和测试，不要盲目重写。
3. 每次只完成一个可验证纵向切片。
4. 修改后执行最小必要测试，再执行回归测试。
5. 最终交付必须说明：改了什么、为什么、测试结果、遗留风险。

## 13. 明确禁止

- 禁止伪造知识库条文、方证、医案或 Evidence ID。
- 禁止把缺失信息默认为“否”。
- 禁止把历史医案直接升级为 canonical 规则。
- 禁止自动向患者输出高风险药物剂量。
- 禁止把 Provider 细节散落在节点、Router、React 组件中。
- 禁止因为“能跑”而跳过类型、异常、测试、注释和结构边界。
- 禁止大规模无关重构。
