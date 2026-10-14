# QA Agent V2 开发交付说明

本次 QA 专项重构以 `QA_Agent重构` 分支最新提交 `d4d74ae48c2616a2cb1ced05b66ba91b0d91e29f` 为实际基线。开发包文档固定的 `4f9c5a6` 仅作为方案参考；最新提交已有的临床 UI、事件分页和执行遥测变更全部保留。

## 运行链路

知识问答使用 LangChain `create_agent()` 和四个只读工具：`lookup_exact_source`、`search_knowledge`、`search_differential`、`get_evidence_details`。工具只能通过 `KnowledgeService` 访问五类 V1.2 知识，检索结果先按当前知识快照和真实 Evidence ID 精确复核，再登记到本轮引用表。模型只生成带本轮 `[n]` 标记的中文解释，引用不存在、跨版本、无来源或来源不足时降级为 `partial`/`no_evidence`。

QA 会话是进程内临时会话，TTL 30 分钟，最多 100 个活跃会话，每会话保留最近 4 轮，使用会话锁串行处理；不使用临床 checkpoint、不写 PostgreSQL 业务表、不接受 consultation/thread ID。

## 接口

- `POST /api/v2/knowledge/chat`：V2 生成式知识问答。
- `POST /api/v1/knowledge/qa`：返回 HTTP 410 `QA_ENDPOINT_DEPRECATED`，指向新接口。
- `/api/v1/knowledge/clauses/*`、`/api/v1/knowledge/formulas/*`、`/api/v1/knowledge/search/classic`、`/api/v1/knowledge/evidence/*`：继续保留精确来源和条文速查能力。

当前 `.env` 的 `KNOWLEDGE_QA_MODE` 是 `extractive`。按照方案，未显式设置为 `llm` 时新聊天返回 `QA_CHAT_NOT_CONFIGURED`，不会静默回退旧摘录链路；本次没有修改 `.env` 或现有配置语义。

## 页面

只重构 `[data-view="knowledge"]`：增加知识问答/条文速查 Tab、临时会话提示、聊天区域和来源栏。模型及来源使用安全 DOM `textContent` 渲染，条文速查仍由原有选择器和 `/api/v1/knowledge/search/classic` 驱动。工作台、执行轨迹和会诊历史 view 未改动。

## 质量验证

- QA 专项（`backend/tests/qa`、QA 单元/API）：`12 passed`，包含真实 `create_agent` 工具循环 fake 测试、参数校验、伪造 Evidence 拒绝和未配置 fail-closed。
- `ruff check --no-cache`（QA 范围）：通过；`black --check backend`：通过；`mypy --no-incremental backend/app`：67 个文件无错误。
- `node --check backend/app/static/knowledge_qa.js`：通过。
- `node --check backend/app/static/console.js`：通过。
- `scripts/engineering_acceptance.py`：10/10 通过，报告输出到任务可写目录，包含知识校验、检索校验和 Console E2E。
- 临床 V2/API/基础设施回归：28 项通过；2 项既有 Milvus Lite 集成测试因 Windows 文件锁和异步清理失败，失败堆栈仅位于既有 `backend/app/rag/vector_store.py`，与本次 QA diff 无关。
- 临床冻结黑名单相对 `d4d74ae`：零差异。

当前 `.env` 仍为 `KNOWLEDGE_QA_MODE=extractive`，因此本次 API 回归验证了 fail-closed 的 `QA_CHAT_NOT_CONFIGURED`。此前已用当前 OpenAI 兼容配置完成 `ChatOpenAI.bind_tools()` 结构化 tool-call 冒烟；线上生成式回答仍需显式切换到既有 `KNOWLEDGE_QA_MODE=llm` 后记录真实 `tool_calls`、usage 和延迟。

## 清理边界

旧 QA 分类/摘录运行逻辑和 StateGraph 已退役。由于 `backend/app/graph/graph_factory.py` 属于临床冻结黑名单且仍保留历史导出名，`knowledge_graph.py` 保留一个显式失败的兼容占位，防止旧消费者静默运行；不再含任何 QA 节点、checkpoint 或医学处理。物理删除该占位需要后续单独修改冻结工厂并运行全站消费者迁移。
