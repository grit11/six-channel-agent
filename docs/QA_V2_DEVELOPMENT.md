# QA Agent V2 开发交付说明

本次 QA 专项重构以 `QA_Agent重构` 分支最新提交 `d4d74ae48c2616a2cb1ced05b66ba91b0d91e29f` 为实际基线。开发包文档固定的 `4f9c5a6` 仅作为方案参考；最新提交已有的临床 UI、事件分页和执行遥测变更全部保留。

## 运行链路

知识问答使用 LangChain `create_agent()` 和四个只读工具：`lookup_exact_source`、`search_knowledge`、`search_differential`、`get_evidence_details`。工具只能通过 `KnowledgeService` 访问五类 V1.2 知识，检索结果先按当前知识快照和真实 Evidence ID 精确复核，再登记到本轮引用表。模型只生成带本轮 `[n]` 标记的中文解释，引用不存在、跨版本、无来源或来源不足时降级为 `partial`/`no_evidence`。

V2.1 增量基线为 `V2_S4优化@9be4567`。QA 历史由独立 `QARepository` 管理，`QA_STORAGE=auto` 复用已装配的 PostgreSQL；研发无数据库时采用明确标识的内存替身。仅新增 `qa_sessions`、`qa_messages`，不使用临床 checkpoint、不接受 consultation/thread ID。PostgreSQL 历史跨进程保留；内存模式服务重启会丢失。旧 TTL helper 仅为既有兼容测试保留，不是线上历史权威。

## 接口

- `POST /api/v2/knowledge/chat`：V2 生成式知识问答。
- `POST /api/v2/knowledge/chat/stream`：公开 SSE；增量仅表示“生成中”，最终引用核验结果覆盖所有临时文本。
- `POST/GET /api/v2/knowledge/sessions`、`GET/PATCH/DELETE /sessions/{id}`：有归属的会话管理；消息分页使用 `after`，标题搜索使用 `q`。
- `GET /api/v2/knowledge/turns/{id}`：断线后回查正式终态；同一 `client_message_id` 不重新调用模型。
- `POST /api/v1/knowledge/qa`：返回 HTTP 410 `QA_ENDPOINT_DEPRECATED`，指向新接口。
- `/api/v1/knowledge/clauses/*`、`/api/v1/knowledge/formulas/*`、`/api/v1/knowledge/search/classic`、`/api/v1/knowledge/evidence/*`：继续保留精确来源和条文速查能力。

部署应使用 `KNOWLEDGE_QA_MODE=llm`；未启用时新聊天返回 `QA_CHAT_NOT_CONFIGURED`，不会静默回退旧摘录链路。V2.1 交付核对时本地已为 `llm`，本次没有修改真实 `.env`。

## 页面

只修改 `[data-view="knowledge"]`：新增会话搜索、分页、重命名、归档/恢复、删除；运行支持停止、回查和显式重试。来源绑定每轮回答并保留精确原文展开。浏览器仅保存 QA 会话/幂等标识，问答正文来自服务器。模型及来源使用安全 DOM `textContent` 渲染；条文速查能力保留。临床工作台、执行轨迹和会诊历史 view 未改动。

## V2.1 部署与恢复

第一次启用 PostgreSQL QA 历史前，在项目根目录使用既有配置执行：

```powershell
conda run -n langgraph python -m alembic -c backend/alembic.ini current
conda run -n langgraph python -m alembic -c backend/alembic.ini upgrade 0003_qa_history
```

执行前确认数据库当前版本为 `0002_reliable_runtime`，不要顺带运行未审查的临床迁移。新迁移 `0003_qa_history` 仅新增 QA 表；应用不隐式执行 DDL。本地已于本次交付执行此迁移。未迁移/数据库失败返回 `QA_HISTORY_UNAVAILABLE`，不会改用临床持久化。生产主体必须由可信服务端认证中间件注入，未注入返回 401；任意客户端身份 header 无效。研发合成主体不能作为上线认证。

每轮预留 user/assistant 后释放事务锁，再运行模型；完整回答与来源一起提交。断线取消写 interrupted，半截正文不保存。已保存但发布失败时回查正式答案，不再收费调用。近期上下文最多六轮完整已核验问答，裁剪只作用于入模上下文，不删除历史；旧引用编号先去除，新医学断言必须本轮重新检索。

QA 保留四工具及 `create_agent()`，每轮模型上限 4 次、工具上限 5 次。重复工具参数使用快照绑定缓存；只对明确连接故障、429/5xx 重试一次，共享工具 30 秒和整轮总预算。无来源、跨版本、伪造 ID 不重试或补造；Provider usage 汇总本轮所有 AIMessage，缺失为 unavailable，部分为 partial。编号存在只能证明来源身份，不等同于医学语义忠实。

V2.1 的实际验收证据与线上评测状态见 `P7_ACCEPTANCE_REPORT.md`；以下原 V2 结果仅为历史记录。

## 原 QA V2 历史验证（不代表本次 V2.1 状态）

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
