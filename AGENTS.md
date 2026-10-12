# 六经辨证 Console V1.2 仓库约束

详细工程规范见 [CODEX_DEVELOPMENT_STANDARD_V1.1.md](CODEX_DEVELOPMENT_STANDARD_V1.1.md)。规范版本与产品版本独立，V1.1 工程约束仍适用于产品 V1.2。

## 开工与优先级

修改前依次阅读本文件、工程规范、[当前六阶段架构](docs/REASONING_V1_2_SIX_STAGE.md)、相关知识说明及数据结构、代码与测试。若旧最终开发文档或 Reasoning Protocol 存在，阅读相关章节。冲突先说明，按用户最新明确要求 > 工程规范 > 当前架构/最终开发文档 > Reasoning Protocol > 代码处理。

## 当前目录与流程

- FastAPI + 原生 Console，前端为 `backend/app/static/`，不建立 React 工程或要求 npm 构建。
- 五自动阶段：`clinical_parser`、`six_channel_screen`、`formula_branch_retrieval`、`differential_verification`、`evidence_topk`；S6 为主图 `doctor_review` 人工审核。
- `nodes/` 的共享八纲、来源、关系、评分、补证模块保留；文件数量不等于 Graph 节点数量。
- 历史医案不自动检索、不回流排序；独立知识 QA 可以按需查询。
- 当前用户允许未上线开发数据不迁移，旧活跃 Checkpoint 不恢复为新图。新线程命名空间 `six-stage-v1.2:`。
- 目录遵循当前[项目结构](docs/PROJECT_STRUCTURE.md)。新增顶层运行目录或移动模块前先说明原因、影响和迁移方案，并修改文档。

## 注释、格式与边界

- 新建或实质修改代码顶部用中文说明职责、依赖、边界；公开类/函数/方法有中文 docstring/JSDoc，重要私有函数、节点、评分和事务有解释业务约束的中文注释。
- Python 4 空格、类型标注、Black 行宽 100、Ruff；异步路径不直接阻塞 I/O。JavaScript 2 空格，HTML/JS 原生展示不复制医学规则。
- Router 不写医学逻辑，Agent/Domain 不 import FastAPI。KnowledgeService 是唯一医学知识边界；Retriever 只返回证据。医生是最终决策入口。
- 不保存或展示隐藏 Chain-of-Thought，只保存结构化可审计状态。未知不等于阴性；支持、反证、缺失与来源均保留；默认 Top-3，多经多关系，禁止 softmax 单分类。

## 基础设施

- 模型、URL、密钥、集合名配置注入，构造收口于既有配置/依赖工厂与基础设施适配器。
- Milvus 生产 Remote；auto 在允许 fallback 时用 MilvusClient Lite 文件模式。节点不能调用 pymilvus；契约相同，health 报实际 backend 与 degraded，两分支有测试。
- 业务 PostgreSQL 用 SQLAlchemy async/asyncpg/Alembic/Repository。SQLite Saver 仅替代图 Checkpoint；不能当业务 ORM。
- Checkpoint auto 优先 PostgreSQL，允许 fallback 时用 AsyncSqliteSaver。lifespan 创建/关闭，不每请求新建；health 报实际后端，两分支有测试。
- 开发业务 PostgreSQL 故障允许进程内存兜底；生产仍要求业务 PostgreSQL。业务表只经显式 Alembic 迁移。
- Lite 缺失当前版本索引默认自动补建，完整索引复用；通过 KnowledgeService 与配置 Embedding 构建，失败不切换为虚假成功。

## 验证与交付

本机 Python 优先已有 Conda `langgraph`，禁止擅建环境或安装到 base。功能变更补必要测试；运行 Ruff、Black、mypy、pytest，涉及 Graph/API/RAG 的对应集成测试，关键流程 Playwright。原生页面用 Node 语法检查，不要求不存在的 npm lint/typecheck。

复杂功能或跨三个以上模块先列计划和影响文件，按可验证切片实现。交付说明变化、原因、验证结果和遗留风险。测试源代码与报告分开：有效 `backend/tests/`、`eval/`、`scripts/` 保留；报告默认父目录 `Console版本测试文件/` 或显式配置目录。

## 日志与禁止事项

统一业务错误码并保留异常链，不得 `except Exception: pass`。可用时日志带 request_id、consultation_id、thread_id；不输出密钥、密码、完整患者信息或隐藏推理。

禁止伪造条文/方证/医案/Evidence ID，将缺失默认阴性，将历史医案升级 canonical，自动输出高风险剂量，将 Provider 散入节点/Router/页面，跳过类型、异常、测试，或无关大规模重构。

## 发布约定

Console 是仓库根。`.env`、数据库、索引、报告、缓存和个人编辑器配置保留本地不上传。用户明确授权 V1.2 文档清理、提交当前“六经辨证推理优化”分支并更新 GitHub；不改写远程历史，不自动选择许可证，不清空运行数据库。
