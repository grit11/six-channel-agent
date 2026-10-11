---
title: "六经辨证临床辅助系统 Codex 开发规范 V1.1"
subtitle: "面向 Codex 编码实现的仓库级工程约束、代码风格、接口预留与质量门槛"
version: "1.1"
status: "Final"
---

# 0. 文档目的

本规范用于指导 Codex 对“六经辨证临床辅助系统”进行持续编码实现。它不是新的业务设计文档，而是对既定《六经辨证临床辅助系统_最终开发文档_V1.0》和《六经辨证_Reasoning_Protocol_V1.0》的**工程执行约束**。

核心目标只有三个：

1. 保证 Codex 严格按照既定目录、架构边界和六经 Reasoning Protocol 实现，不因局部任务擅自改变总体设计；
2. 保证所有代码风格统一、缩进美观、中文注释充分、类型清晰、测试可执行；
3. 为后续替换或接入 LLM、Embedding、Reranker、Milvus、PostgreSQL 等外部基础设施预留稳定接口，避免厂商 SDK 渗透业务代码。

> 本规范中的“必须”“禁止”“不得”均为强制要求。Codex 在无法同时满足需求和规范时，必须先报告冲突，不能静默绕过。

---

# 1. 规范优先级与开工前阅读顺序

每次开发任务开始前，Codex 必须按以下顺序读取：

1. 根目录 `AGENTS.md`；
2. 根目录 `CODEX_DEVELOPMENT_STANDARD_V1.1.md`；
3. 《六经辨证临床辅助系统_最终开发文档_V1.0》对应章节；
4. 《六经辨证_Reasoning_Protocol_V1.0》对应阶段；
5. `knowledge/v1.2/` 中本任务涉及的知识数据；
6. 当前代码、已有测试、已有迁移脚本。

当信息冲突时，优先级为：

```text
用户最新明确要求
    > Codex 开发规范
    > 最终开发文档
    > Reasoning Protocol
    > 当前代码实现
```

不得因为“现有代码已经这样写”就继续放大与文档不一致的实现。

---

# 2. 当前项目正式定位

系统是**医生侧六经辨证临床辅助系统**，不是自动诊疗机器人，也不是自动处方系统。

核心链路固定为：

```text
医生病例输入
    -> Clinical Evidence 结构化
    -> 八纲特征分析
    -> 六经 Top-N 候选
    -> 病证关系判断
    -> 方证候选
    -> 鉴别 / 反证 / Evidence Gap
    -> RAG 证据约束
    -> Top-K 辨证路径
    -> 医生审核
    -> 最终方案
```

工程职责固定为：

```text
LLM        负责语义理解与结构化推理
RAG        负责查证与证据召回
规则        负责硬约束和安全边界
LangGraph  负责状态、路由、循环与恢复
FastAPI    负责服务化边界
React      负责医生工作台交互
医生        负责最终决策
```

---

# 3. 主工程目录是实现契约

Codex 必须严格按以下目录编码，不得为了局部方便随意新增顶层目录、重命名目录或移动职责。

```text
six-channel-clinical-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   ├── exceptions.py
│   │   │   └── dependencies.py
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── router.py
│   │   │       ├── consultation.py
│   │   │       ├── knowledge.py
│   │   │       ├── review.py
│   │   │       ├── stream.py
│   │   │       └── health.py
│   │   ├── services/
│   │   │   ├── consultation_service.py
│   │   │   ├── knowledge_qa_service.py
│   │   │   ├── review_service.py
│   │   │   └── stream_service.py
│   │   ├── graph/
│   │   │   ├── main_graph.py
│   │   │   ├── six_channel_graph.py
│   │   │   └── graph_factory.py
│   │   ├── agents/
│   │   │   ├── six_channel_agent.py
│   │   │   └── knowledge_qa_agent.py
│   │   ├── nodes/
│   │   │   ├── clinical_parser.py
│   │   │   ├── eight_principles.py
│   │   │   ├── channel_candidate.py
│   │   │   ├── syndrome_relation.py
│   │   │   ├── formula_candidate.py
│   │   │   ├── differential_reasoner.py
│   │   │   ├── evidence_gap.py
│   │   │   ├── topk_builder.py
│   │   │   └── safety_triage.py
│   │   ├── knowledge/
│   │   │   ├── service.py
│   │   │   ├── classic.py
│   │   │   ├── formula.py
│   │   │   ├── differential.py
│   │   │   ├── case.py
│   │   │   └── framework.py
│   │   ├── rag/
│   │   │   ├── query_builder.py
│   │   │   ├── retriever.py
│   │   │   ├── fusion.py
│   │   │   ├── reranker.py
│   │   │   ├── grader.py
│   │   │   └── vector_store.py
│   │   ├── schemas/
│   │   │   ├── clinical.py
│   │   │   ├── reasoning.py
│   │   │   ├── evidence.py
│   │   │   ├── consultation.py
│   │   │   ├── review.py
│   │   │   ├── knowledge.py
│   │   │   └── events.py
│   │   ├── persistence/
│   │   │   ├── database.py
│   │   │   ├── repositories.py
│   │   │   ├── checkpoint.py
│   │   │   └── models.py
│   │   ├── prompts/
│   │   │   ├── clinical_parser.md
│   │   │   ├── eight_principles.md
│   │   │   ├── channel_reasoner.md
│   │   │   ├── relationship_reasoner.md
│   │   │   ├── differential_reasoner.md
│   │   │   └── final_builder.md
│   │   ├── safety/
│   │   │   ├── rules.py
│   │   │   └── risk_registry.py
│   │   └── observability/
│   │       ├── tracing.py
│   │       └── metrics.py
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── api/
│   ├── alembic/
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── router.tsx
│   │   │   └── providers.tsx
│   │   ├── pages/
│   │   │   ├── ConsultationWorkbench.tsx
│   │   │   ├── KnowledgeCenter.tsx
│   │   │   └── ConsultationHistory.tsx
│   │   ├── components/
│   │   │   ├── ClinicalInputPanel/
│   │   │   ├── EvidenceSummary/
│   │   │   ├── TopKReasoningCard/
│   │   │   ├── DifferentialPanel/
│   │   │   ├── EvidenceDrawer/
│   │   │   ├── SupplementBar/
│   │   │   ├── DoctorReviewPanel/
│   │   │   └── ReasoningProgress/
│   │   ├── api/
│   │   │   ├── client.ts
│   │   │   ├── consultation.ts
│   │   │   ├── knowledge.ts
│   │   │   └── stream.ts
│   │   ├── stores/
│   │   │   └── consultationStore.ts
│   │   ├── hooks/
│   │   ├── types/
│   │   └── utils/
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
│
├── knowledge/
│   └── v1.2/
├── eval/
├── scripts/
├── docker-compose.yml
├── .env.example
├── Makefile
└── README.md
```

治理文件 `AGENTS.md` 和本规范可位于根目录，不视为破坏运行时目录契约。

如确实需要新增运行时代码文件：

1. 优先放入上述已有目录；
2. 新文件职责必须与所在目录一致；
3. 禁止新增新的顶层领域层；
4. 如果需要新目录，先更新开发文档后再实施。


## 3.1 Codex 本地调试环境（强制）

用户本地 Anaconda 已存在 Conda 环境：

```text
langgraph
```

Codex 在本机调试 Python 后端、LangGraph、RAG、Milvus、PostgreSQL Adapter 和测试时，**必须优先使用该环境**，不得擅自创建新的 `.venv`、`venv` 或新的 Conda 环境，也不得把依赖安装到 `base` 环境。

自动化命令优先使用不依赖 shell 激活状态的写法：

```powershell
conda run -n langgraph python --version
conda run -n langgraph python -m pytest
conda run -n langgraph python -m ruff check backend
conda run -n langgraph python -m black --check backend
```

若需要启动 FastAPI：

```powershell
conda run -n langgraph python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

若当前终端已经正确初始化 Conda，也允许：

```powershell
conda activate langgraph
python -m pytest
```

但 Codex 的可复现调试命令和最终测试报告应优先记录 `conda run -n langgraph ...` 形式，避免不同 Shell 下 `conda activate` 状态不一致。

开始编码前至少执行一次：

```powershell
conda run -n langgraph python --version
conda run -n langgraph python -c "import sys; print(sys.executable)"
```

并确认解释器确实来自 `langgraph` 环境。

依赖安装规则：

1. 优先根据 `backend/pyproject.toml` 安装项目依赖；
2. 如需新增依赖，必须同步更新 `pyproject.toml` / lock 文件，而不是只在本机临时安装；
3. 临时验证也只能安装到 `langgraph` 环境；
4. 不得修改用户其他 Conda 环境；
5. 不得因为某依赖暂时缺失就跳过测试并声称功能已完成。

---

# 4. 代码美观与缩进规范

## 4.1 Python

强制：

- 4 个空格缩进；禁止 Tab；
- UTF-8；
- Black 风格；推荐 line length = 100；
- Ruff 负责 lint、import 排序和低级错误检查；
- 所有函数参数和返回值必须有类型标注；
- 公共 API、Pydantic Model、Repository、Service、LangGraph Node 返回结构必须可静态理解；
- 同一层级空行、换行和参数对齐应保持一致；
- 单函数建议不超过 60 行，超过时优先拆分纯函数；
- 单文件建议不超过 500 行，超过时先判断是否职责过载；
- 禁止深度超过 4 层的 if/for 嵌套，优先 guard clause；
- 禁止巨型“万能函数”。

推荐工具：

```text
black
ruff
mypy 或 pyright（二选一，项目统一）
pytest
```

## 4.2 TypeScript / React

强制：

- 2 个空格缩进；
- Prettier；
- ESLint；
- `strict: true`；
- 禁止大量 `any`；
- API 类型必须集中定义；
- React 组件单一职责；
- 页面组件负责组合，不承载网络细节和医学规则；
- Server State 使用 TanStack Query；
- 仅 UI 局部状态进入 Zustand；
- 不在 JSX 中塞入复杂推理/转换表达式。

---

# 5. 中文注释规范（强制）

这是本项目的核心代码风格要求。

## 5.1 每个脚本开头必须有中文说明

Python 新文件必须使用模块 docstring，例如：

```python
"""
六经辨证候选生成节点。

职责：
1. 根据已结构化的 ClinicalEvidence 生成六经宽召回候选；
2. 调用 KnowledgeService 获取经典证据进行校正；
3. 输出可供后续病证关系判断使用的 ChannelCandidate 列表。

边界：
- 本模块不直接确定最终诊断；
- 本模块不直接访问 Milvus 或 PostgreSQL；
- 所有知识检索必须通过 KnowledgeService。
"""
```

TypeScript/React 新文件必须在文件顶部写中文块注释：

```ts
/**
 * 六经辨证 Top-K 候选卡片。
 *
 * 职责：展示候选证型、排序分、支持证据、反证和缺失证据。
 * 边界：仅负责展示和交互，不在前端重新计算医学评分。
 */
```

## 5.2 每个函数必须有中文 docstring/JSDoc

Python：

```python
async def search_classic(query: ClassicSearchQuery) -> list[Evidence]:
    """检索《伤寒论》经典条文，并返回带来源信息的标准 Evidence。"""
```

复杂函数建议使用统一格式：

```python
def build_channel_score(...) -> float:
    """
    计算单个六经候选的排序分。

    参数：
        ...

    返回：
        0~1 的排序分，仅用于候选排序，不代表临床概率。

    约束：
        Unknown 不得计入 Negative；反证惩罚权重高于普通正证。
    """
```

TypeScript：

```ts
/**
 * 将后端 consultation snapshot 转换为工作台展示模型。
 * 不得在这里重新实现任何六经医学判断。
 */
function mapConsultationSnapshot(...) { ... }
```

## 5.3 重要逻辑必须写中文行内注释

必须注释的场景：

- LangGraph 条件路由；
- interrupt / resume；
- Candidate Score；
- Differential Matrix；
- Evidence Gap；
- Dense/Sparse 融合；
- Reranker 截断；
- Milvus metadata filter；
- 数据库事务；
- 幂等逻辑；
- 高风险安全规则；
- 三值状态转换；
- 异常 fallback；
- 版本兼容逻辑。

注释重点解释“为什么”，不要写这种无效注释：

```python
# 给 i 加 1
index += 1
```

应写：

```python
# 仅在本轮成功获得新的区分性证据时增加补充轮次，避免空回复耗尽轮次。
supplement_round += 1
```

---

# 6. 命名规范

## Python

- 文件：`snake_case.py`
- 函数/变量：`snake_case`
- 类/Pydantic Model：`PascalCase`
- 常量：`UPPER_SNAKE_CASE`
- 私有实现：前缀 `_`

避免：

```text
data1
result2
tmp
obj
info
handle_data
process
```

优先：

```text
clinical_evidence
classic_evidence
formula_candidates
differential_matrix
consultation_snapshot
```

## TypeScript

- 组件：`PascalCase`
- 函数/变量：`camelCase`
- 类型/interface：`PascalCase`
- 常量：`UPPER_SNAKE_CASE` 或项目统一常量规范

医学领域名称保持开发文档中的标准术语，不随意发明缩写。

---

# 7. 架构边界

## 7.1 FastAPI

FastAPI 是 Application Boundary，只负责：

- HTTP 路由；
- 鉴权依赖；
- 输入校验；
- HTTP 状态码；
- SSE；
- Application Service 调用；
- 异常到 API Error 的映射。

禁止：

```python
@router.post("/consultations")
async def create(...):
    # 禁止在 Router 中调用 LLM、写辨证 Prompt、算六经分数、直接查 Milvus。
```

正确调用链：

```text
FastAPI Router
    -> ConsultationService
    -> LangGraph / Domain
    -> KnowledgeService / Repository
```

## 7.2 Agent / Domain

- 不 import FastAPI；
- 不依赖 HTTP Request/Response；
- 不依赖 React；
- 不直接拼接数据库连接；
- 不直接创建 Milvus Client；
- 通过依赖注入获得模型和知识服务。

## 7.3 Frontend

- 不复制医学规则；
- 不在前端重新计算六经评分；
- consultation snapshot 是权威状态；
- SSE 只是增量提示，最终状态以 GET snapshot 为准。

---

# 8. 外部依赖统一采用 Adapter / Factory 注入

业务代码不得绑定具体厂商。

当前必须预留：

1. LLM / Chat Model；
2. Embedding Model；
3. Reranker Model；
4. Milvus；
5. PostgreSQL；
6. LangGraph PostgresSaver。

## 8.1 统一配置入口

所有外部配置集中在：

```text
backend/app/core/config.py
```

使用 Pydantic Settings。禁止在模块顶部散落 `os.getenv()`。

依赖创建集中在：

```text
backend/app/core/dependencies.py
```

Provider 特有代码只能出现在“配置/适配层”中，不得散落到 node、agent、router。

---

# 9. LLM 接口预留规范

业务层不要直接假设 OpenAI、DeepSeek、Qwen 或其他厂商。

推荐统一能力接口：

```python
from typing import Protocol, TypeVar

T = TypeVar("T")


class StructuredLLM(Protocol):
    """结构化大模型调用接口，屏蔽具体厂商 SDK。"""

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """执行普通文本模型调用。"""

    async def ainvoke_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
    ) -> T:
        """按照指定 Pydantic Schema 返回结构化结果。"""
```

若直接采用 LangChain ChatModel，也必须通过工厂构建：

```python
def build_chat_model(settings: Settings):
    """根据配置构建 ChatModel；业务模块不得关心具体 Provider。"""
```

必须配置：

```text
LLM_PROVIDER
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
LLM_TEMPERATURE
LLM_TIMEOUT_SECONDS
LLM_MAX_RETRIES
```

原则：

- 医学结构化节点 temperature 默认低；
- 模型不可用时明确失败，不允许偷偷改用一个硬编码模型；
- 所有模型调用应支持超时和有限重试；
- 重试不得导致重复写数据库或重复提交医生审核。

---

# 10. Embedding 接口预留规范

Embedding 必须与 KnowledgeService 解耦。

建议接口：

```python
class EmbeddingProvider(Protocol):
    """向量化接口，供知识索引和在线查询统一使用。"""

    @property
    def dimension(self) -> int:
        """返回向量维度。"""

    async def embed_query(self, text: str) -> list[float]:
        """生成单条查询向量。"""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成知识文档向量。"""
```

必须配置：

```text
EMBEDDING_PROVIDER
EMBEDDING_BASE_URL
EMBEDDING_API_KEY
EMBEDDING_MODEL
EMBEDDING_DIMENSION
EMBEDDING_BATCH_SIZE
```

硬性约束：

- 索引时记录 embedding model/version/dimension；
- 更换模型维度不得直接写进旧 collection；
- 模型升级必须重建或新建版本化 collection；
- 在线查询模型必须与索引模型兼容。

---

# 11. Reranker 接口预留规范

实现位置：

```text
backend/app/rag/reranker.py
```

统一接口：

```python
class Reranker(Protocol):
    """重排接口，将召回候选按医学查询相关度重新排序。"""

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """返回重排后的候选，必须保留原 Evidence 来源信息。"""
```

必须配置：

```text
RERANKER_PROVIDER
RERANKER_BASE_URL
RERANKER_API_KEY
RERANKER_MODEL
RERANKER_TIMEOUT_SECONDS
```

禁止 Reranker 返回“诊断”；它只能改变证据排序。

---

# 12. Milvus 向量库规范

## 12.1 生产环境与本地环境策略

正式部署的生产目标是 **Docker 中运行的真实 Milvus 服务**。应用代码通过 `pymilvus` 连接该服务，不得把 Milvus 运行进程嵌入 FastAPI。

同时，为保证 Codex 在本地没有启动 Docker Milvus 时仍能完成开发、单测和端到端调试，向量层必须实现 **Remote Milvus -> Milvus Lite 自动降级**：

```text
优先连接 Docker / Remote Milvus
        |
        | 可用
        v
    使用 Remote Milvus

        | 不可用
        v
使用 Milvus Lite 本地文件
```

Milvus Lite 必须使用用户指定的官方客户端入口：

```python
from pymilvus import MilvusClient
```

远程和 Lite 必须共用同一个 `VectorStorePort`，上层 `KnowledgeService`、Retriever、LangGraph Node 不得感知当前使用的是 Remote Milvus 还是 Milvus Lite。

## 12.2 不使用业务层散落的 if/else

“Remote 不可用则 Lite”的 if/else 只能存在于基础设施工厂或 VectorStore Adapter 初始化处，例如：

```text
backend/app/rag/vector_store.py
backend/app/core/dependencies.py
```

严禁在以下位置重复写 Milvus 可用性判断：

- LangGraph Node；
- Agent；
- KnowledgeService 业务方法；
- FastAPI Router；
- React 前端。

## 12.3 推荐配置模式

新增以下配置：

```text
MILVUS_MODE=auto                # auto | remote | lite
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_DATABASE=default
MILVUS_LITE_PATH=./data/milvus_lite.db
MILVUS_FALLBACK_TO_LITE=true
MILVUS_CONNECT_TIMEOUT_SECONDS=3
```

行为定义：

```text
MILVUS_MODE=remote
    -> 只允许 Remote Milvus；连接失败直接报错。

MILVUS_MODE=lite
    -> 直接使用 Milvus Lite，不尝试 Remote。

MILVUS_MODE=auto
    -> 先探测 Remote Milvus；
    -> 可用：使用 Remote；
    -> 不可用且 MILVUS_FALLBACK_TO_LITE=true：使用 Lite；
    -> Remote 与 Lite 均不可用：启动失败并输出明确错误。
```

本项目本地开发默认推荐 `auto`。Docker 生产环境可保持 `auto`，但发生 Lite 降级时必须在健康检查和日志中标记为 `degraded`，不能让运维误以为仍在使用生产 Milvus。

## 12.4 推荐实现骨架

```python
"""
Milvus 向量库适配器。

职责：优先连接 Docker/Remote Milvus；当配置允许且远程不可用时，
自动降级为 Milvus Lite。本模块只提供统一 VectorStorePort，
不包含六经辨证、方证判断或 KnowledgeService 业务逻辑。
"""

from pathlib import Path
from pymilvus import MilvusClient


def create_milvus_client(settings: Settings) -> tuple[MilvusClient, str]:
    """根据配置创建 Milvus Client，并返回实际使用的后端类型。"""
    if settings.milvus_mode == "lite":
        return _create_lite_client(settings), "lite"

    if settings.milvus_mode in {"auto", "remote"}:
        remote_client = _try_create_remote_client(settings)
        if remote_client is not None:
            return remote_client, "remote"

    if settings.milvus_mode == "remote":
        raise InfrastructureUnavailableError("Remote Milvus 不可用，且当前禁止降级。")

    if settings.milvus_fallback_to_lite:
        return _create_lite_client(settings), "lite"

    raise InfrastructureUnavailableError("Remote Milvus 不可用，Milvus Lite 降级也未启用。")


def _create_lite_client(settings: Settings) -> MilvusClient:
    """使用本地数据库文件创建 Milvus Lite Client。"""
    lite_path = Path(settings.milvus_lite_path)
    lite_path.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(uri=str(lite_path))
```

`_try_create_remote_client()` 必须执行一次真实轻量探测（例如列举 Collection 或等价健康操作），不能仅因为对象构造成功就认为 Remote Milvus 可用。

探测失败时：

- 记录 warning；
- 保留异常链/错误原因；
- 不打印 Token；
- 然后按照配置进入 Lite 分支。

## 12.5 Async 约束

`MilvusClient` 属于同步客户端。FastAPI / LangGraph 的 async 调用链中，不得直接执行可能长时间阻塞的同步网络或磁盘操作。

Adapter 应统一通过线程池桥接，例如 `asyncio.to_thread()` 或项目统一的 async wrapper，把同步 `search/upsert/list_collections` 等操作隔离在基础设施层。

上层依然暴露 async 接口：

```python
class VectorStorePort(Protocol):
    """医学知识向量检索接口。"""

    async def search(
        self,
        *,
        query_vector: list[float],
        knowledge_type: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedDocument]:
        """根据向量和 metadata filter 检索知识。"""

    async def upsert(self, documents: list[VectorDocument]) -> None:
        """幂等写入或更新知识向量。"""

    async def delete_by_version(self, knowledge_version: str) -> None:
        """按知识版本删除索引，用于可控重建。"""
```

## 12.6 Remote / Lite 数据契约必须一致

Remote Milvus 和 Milvus Lite 必须使用一致的：

- `knowledge_id`；
- `knowledge_type`；
- `knowledge_version`；
- `source_level`；
- `six_channel`；
- `formula_name`；
- `clause_id`；
- `canonical`；
- `metadata_json`；
- embedding 维度与距离度量语义。

若 Milvus Lite 对某个生产索引参数不支持，不得为了本地运行修改上层 Schema。应在 Adapter 内进行能力适配，并记录当前实际后端与实际索引策略。

## 12.7 推荐 Collection 设计

```text
six_channel_classic_v12
six_channel_formula_v12
six_channel_differential_v12
six_channel_case_v12
six_channel_framework_v12
```

生产 Milvus 的索引参数通过环境变量配置，不硬编码：

```text
MILVUS_METRIC_TYPE
MILVUS_INDEX_TYPE
MILVUS_HNSW_M
MILVUS_HNSW_EF_CONSTRUCTION
MILVUS_SEARCH_EF
```

V1 生产环境推荐以 `COSINE + HNSW` 为目标配置；Lite 的实际索引能力由 Adapter 处理，不允许医学业务代码分支。

## 12.8 索引脚本

`scripts/index_knowledge.py` 必须满足：

- 可重复执行；
- 幂等；
- 支持 `--knowledge-version`；
- 支持 `--rebuild`；
- 支持 dry-run；
- 索引前验证 JSONL；
- 记录 embedding model、dimension、文档数；
- 输出实际 backend：`remote` 或 `lite`；
- 出错时不留下“半完成但被标记成功”的版本。

## 12.9 Milvus 必测场景

至少覆盖：

1. Remote Milvus 可用 -> 选择 `remote`；
2. Remote Milvus 不可用 -> 自动选择 `lite`；
3. `MILVUS_MODE=remote` 且远程不可用 -> 明确失败，不偷偷 Lite；
4. Lite 文件目录不存在 -> 可安全创建；
5. Remote/Lite 返回统一 `RetrievedDocument`；
6. 降级时 `/health` 明确报告 `degraded` 和 `vector_backend=lite`。

---

# 13. PostgreSQL 与 LangGraph Checkpoint 降级规范

## 13.1 生产目标

正式部署仍以 PostgreSQL 为生产数据库目标：

- SQLAlchemy 2.x async；
- asyncpg；
- Alembic；
- Repository Pattern；
- Pydantic Schema 与 ORM Model 分离。

同时，LangGraph checkpoint 必须支持 **PostgreSQL -> SQLite 自动降级**，保证 Codex 在本地尚未接入 PostgreSQL 时也能完成多轮状态、interrupt/resume 和端到端调试。

SQLite checkpoint 必须使用用户指定入口：

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
```

## 13.2 必须区分“业务数据库”和“LangGraph Checkpoint”

`AsyncSqliteSaver` 只用于 **LangGraph checkpoint fallback**，它不是 SQLAlchemy 业务数据库的替代品。

必须保持两个概念分离：

```text
业务数据：consultation / review / audit / history
    -> Repository -> PostgreSQL（生产）

LangGraph 状态：thread checkpoint / interrupt / resume
    -> PostgreSQL Checkpointer（优先）
    -> AsyncSqliteSaver（PostgreSQL 未接入/不可用时降级）
```

Codex 不得把 `AsyncSqliteSaver` 当成业务 ORM Repository 使用。

本地未接入业务 PostgreSQL 时：

- 六经推理 Graph 仍必须可以依靠 SQLite checkpoint 正常运行；
- 需要业务数据库的历史记录、审计等能力必须通过 feature flag / Fake Repository / 明确不可用状态处理；
- 不得为了“能跑”把业务 Repository 悄悄改写成 checkpoint SQLite 表。

## 13.3 Checkpoint 配置

新增：

```text
CHECKPOINT_BACKEND=auto            # auto | postgres | sqlite
LANGGRAPH_DATABASE_URL=
LANGGRAPH_SQLITE_PATH=./data/langgraph_checkpoints.sqlite3
CHECKPOINT_FALLBACK_TO_SQLITE=true
POSTGRES_CONNECT_TIMEOUT_SECONDS=3
```

行为：

```text
CHECKPOINT_BACKEND=postgres
    -> 只允许 PostgreSQL checkpointer；不可用则启动失败。

CHECKPOINT_BACKEND=sqlite
    -> 直接 AsyncSqliteSaver。

CHECKPOINT_BACKEND=auto
    -> LANGGRAPH_DATABASE_URL 已配置且 PostgreSQL 可访问：PostgreSQL；
    -> 未配置或不可访问，并允许 fallback：AsyncSqliteSaver；
    -> 两者都不可用：明确失败。
```

## 13.4 Checkpointer 工厂

实现位置：

```text
backend/app/persistence/checkpoint.py
```

推荐职责：

```python
"""
LangGraph Checkpointer 工厂。

职责：生产优先使用 PostgreSQL；本地未配置或不可访问 PostgreSQL 时，
按配置降级为 AsyncSqliteSaver。Checkpointer 生命周期由应用 lifespan 管理，
禁止每个请求创建一个新的 Saver。
"""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def open_checkpointer(settings: Settings):
    """根据配置选择 PostgreSQL 或 SQLite Checkpointer。"""
    if settings.checkpoint_backend == "sqlite":
        return await _open_sqlite_checkpointer(settings)

    if settings.checkpoint_backend in {"auto", "postgres"}:
        postgres_saver = await _try_open_postgres_checkpointer(settings)
        if postgres_saver is not None:
            return postgres_saver

    if settings.checkpoint_backend == "postgres":
        raise InfrastructureUnavailableError("PostgreSQL Checkpointer 不可用。")

    if settings.checkpoint_fallback_to_sqlite:
        return await _open_sqlite_checkpointer(settings)

    raise InfrastructureUnavailableError("没有可用的 LangGraph Checkpointer。")
```

实际代码应根据当前安装的 LangGraph checkpoint package API 正确管理 async context manager。**不得照抄伪代码而忽略资源关闭。**

## 13.5 生命周期要求

Checkpointer 必须由 FastAPI `lifespan` 或统一 Application Container 创建并在应用退出时释放：

```text
startup/lifespan
    -> 创建 checkpointer
    -> compile graph
    -> 保存到 app/container

request
    -> 复用已创建 graph/checkpointer

shutdown
    -> 正确关闭连接
```

禁止：

```text
每个 HTTP 请求 -> 新建 AsyncSqliteSaver / PostgresSaver -> compile graph
```

否则会导致锁、连接泄露、thread state 不一致和性能问题。

## 13.6 PostgreSQL 业务数据库文件职责

```text
persistence/database.py
    PostgreSQL AsyncEngine / AsyncSession 工厂与连接健康探测

persistence/models.py
    ORM Model

persistence/repositories.py
    Repository 接口及 PostgreSQL 实现

persistence/checkpoint.py
    LangGraph Checkpointer 选择与生命周期
```

## 13.7 Service 控制事务

禁止 Router 直接：

```python
await session.execute(...)
```

推荐：

```text
Router
  -> Service
       -> Repository
```

跨多个 Repository 的业务操作由 Service 明确事务边界。

## 13.8 Schema 变更

业务 PostgreSQL 任何表结构变化：

```text
修改 ORM
-> 新增 Alembic migration
-> migration test
-> upgrade/downgrade 验证
```

禁止应用启动时隐式 CREATE/ALTER 业务表。

LangGraph 官方 checkpointer 自身需要的初始化按对应 Saver API 处理，与业务 Alembic migration 分开。

## 13.9 健康状态

`/api/v1/health` 至少返回内部状态字段：

```json
{
  "status": "ok_or_degraded",
  "vector_backend": "remote|lite",
  "checkpoint_backend": "postgres|sqlite",
  "business_database": "postgres|disabled|unavailable"
}
```

前端无需显示数据库密码、URL 或异常堆栈。

## 13.10 Checkpoint 必测场景

至少覆盖：

1. PostgreSQL URL 配置且可访问 -> PostgreSQL saver；
2. PostgreSQL 未配置 -> AsyncSqliteSaver；
3. PostgreSQL 配置但不可访问 -> AsyncSqliteSaver（auto + fallback）；
4. `CHECKPOINT_BACKEND=postgres` 且不可用 -> 明确失败；
5. SQLite checkpoint 在重启后可以恢复同一 `thread_id`；
6. interrupt/resume 在 SQLite fallback 模式下仍可用；
7. fallback 日志不泄露数据库密码。

---

# 14. LangGraph 编码规范

## 14.1 Graph 只负责编排

Graph 文件负责：

- StateGraph 定义；
- 节点连接；
- conditional edge；
- interrupt/resume；
- checkpoint；
- 子图装配。

禁止把大段 Prompt 和医学计算直接写在 graph 文件中。

## 14.2 Node 单一职责

每个 Node：

```text
一个明确输入
-> 一个明确处理目标
-> 返回局部 state update
```

例如 `channel_candidate.py` 不应同时实现：

```text
六经候选 + 方证匹配 + 鉴别 + Top-K + 数据库存储
```

## 14.3 State 更新原则

- 不原地神秘修改共享 dict；
- 返回明确的局部更新；
- 关键状态使用 Pydantic/TypedDict 清晰定义；
- 不把大型数据库对象、Milvus Client、LLM Client 塞入 State；
- State 保存数据，不保存基础设施连接。

## 14.4 interrupt/resume

进入 `needs_evidence` 时必须保存：

- consultation_id；
- thread_id；
- 当前 Top-K；
- missing_evidence；
- 已完成阶段；
- supplement_round。

恢复时使用同一 thread，不创建新的隐式会话。

---

# 15. 六经 Reasoning Protocol 强制规则

以下规则属于医学推理内核，不允许 Codex 为“简化实现”而改变：

1. 先结构化事实，后医学推理；
2. `unknown != negative`；
3. 六经候选允许多标签；
4. 不使用 softmax 强制六选一；
5. 候选必须记录 supporting / contradicting / missing evidence；
6. 关系层必须支持本证、兼证、合病、并病、直中、两感、传经、变证/坏病；
7. 医案是 case-derived evidence，不得覆盖 canonical；
8. RAG 只提供证据；
9. 最终默认 Top-3；
10. 医生最终审核。

任何变更此规则的代码提交必须明确标为“Reasoning Protocol 变更”，不能作为普通 refactor 混入。

---

# 16. KnowledgeService 编码规范

`backend/app/knowledge/service.py` 是唯一医学知识访问边界。

公共能力固定围绕：

```text
search_core_framework()
search_classic()
search_formula()
search_differential()
search_case()
get_clause_by_id()
get_formula_by_name()
```

Node、Agent、Service 不得：

- 自己打开 JSONL 搜索；
- 自己调用 Milvus；
- 自己访问“另一个临时知识库”；
- 直接在 Prompt 中硬编码大段条文替代知识检索。

所有 Evidence 返回必须可追溯：

```text
evidence_id
knowledge_id
knowledge_type
knowledge_version
source
source_level
content
retrieval_score
rerank_score
metadata
```

---

# 17. RAG 实现规范

推荐链路：

```text
QueryBuilder
    -> Embedding
    -> Milvus Dense Retrieval
    -> Sparse/BM25（可插拔）
    -> Fusion
    -> Reranker
    -> Grader
    -> Evidence
```

要求：

- RAG 不得直接返回 diagnosis；
- metadata filter 优先用于 six_channel、knowledge_type、canonical、version 等确定条件；
- Dense 召回和 rerank 分数不要混成同一个不可解释字段；
- 召回结果必须去重；
- 经典条文精确 ID 查询优先走 `get_clause_by_id()`，不必浪费向量检索；
- 低质量召回应允许返回“证据不足”，禁止模型补齐不存在的来源。

---

# 18. FastAPI API 编码规范

## 18.1 Router

Router 保持薄：

```python
@router.post(...)
async def create_consultation(
    request: CreateConsultationRequest,
    service: ConsultationService = Depends(...),
) -> ConsultationResponse:
    """创建辨证会话，并返回当前会话快照。"""
    return await service.start(...)
```

## 18.2 Pydantic Schema

要求：

- API Request / Response 单独定义；
- `extra="forbid"` 用于关键写接口；
- 枚举字段使用 Enum/Literal；
- 所有字段有清晰中文 Field description；
- API Response 不直接暴露 ORM object。

## 18.3 HTTP 与业务错误

在 `core/exceptions.py` 定义业务异常，不允许每个 Router 自创错误格式。

建议统一：

```json
{
  "error": {
    "code": "CONSULTATION_NOT_FOUND",
    "message": "未找到指定辨证会话",
    "request_id": "..."
  }
}
```

## 18.4 SSE

SSE 事件仅表达：

```text
stage_started
stage_completed
retrieval_completed
needs_evidence
topk_updated
review_required
completed
failed
heartbeat
```

SSE 不作为最终事实源；断线重连后必须重新 GET consultation snapshot。

---

# 19. 前端编码规范

## 19.1 页面职责

`ConsultationWorkbench.tsx`：组合工作台，不直接写 fetch 和医学逻辑。

`KnowledgeCenter.tsx`：独立知识查询。

`ConsultationHistory.tsx`：历史会话。

## 19.2 API 层

所有请求统一在：

```text
frontend/src/api/
```

- `client.ts`：base URL、拦截器、错误转换；
- `consultation.ts`：会话相关；
- `knowledge.ts`：知识查询；
- `stream.ts`：SSE。

组件内禁止散落 `fetch("/api/...`)`。

## 19.3 类型

后端关键 Response 类型必须在前端对应建模。

如未来加入 OpenAPI codegen，应保留清晰分层，不把生成代码和业务 UI 混在一个文件。

## 19.4 医学状态展示

必须明确区分：

```text
Positive
Negative
Unknown
```

视觉上可使用不同 icon/Tag，但不能把 Unknown 显示成“无”。

Top-K 卡片必须能展示：

- 排名；
- 排序分；
- 六经；
- 证型/方证；
- 支持证据；
- 反证；
- 缺失证据；
- 经典来源；
- 方证来源；
- 鉴别依据。

---

# 20. 配置文件与密钥规范

所有密钥：

- 只放环境变量或密钥管理系统；
- `.env` 永不提交；
- `.env.example` 只放占位符；
- 单测使用 fake/mock provider；
- 禁止把真实 API Key 写进测试快照、日志或 README。

配置必须在应用启动阶段完成校验。

缺少必要生产配置时，应 fail-fast，而不是运行到某节点才报 obscure error。

---

# 21. Async / I/O 规范

本项目后端默认异步。

必须 async 的典型路径：

- LLM 网络调用；
- Embedding API；
- Reranker API；
- PostgreSQL；
- Milvus（若 SDK 为同步接口，应通过明确适配策略处理，不能直接长时间阻塞 event loop）；
- SSE。

禁止：

```python
async def handler():
    requests.get(...)
```

同步 CPU 重任务需明确线程池/进程池策略。

---

# 22. 日志、Trace 与审计

## 22.1 结构化日志

每条关键日志尽量包含：

```text
request_id
consultation_id
thread_id
stage
knowledge_version
protocol_version
model_name
latency_ms
```

## 22.2 禁止记录

- API Key；
- 数据库密码；
- Milvus Token；
- 完整患者敏感信息；
- 模型隐藏 Chain-of-Thought；
- 无必要的完整 prompt 原文。

## 22.3 审计记录

审计对象是：

- 输入快照版本；
- 结构化临床事实；
- Evidence ID；
- Top-K；
- 医生审核动作；
- 模型/知识/协议版本；
- 错误事件。

---

# 23. 异常处理规范

禁止：

```python
try:
    ...
except Exception:
    pass
```

每类异常至少分为：

```text
ConfigurationError
ProviderUnavailableError
KnowledgeRetrievalError
PersistenceError
ConsultationNotFoundError
InvalidStateTransitionError
SafetyBlockedError
```

跨层异常转换：

```text
Adapter Exception
    -> Domain/Application Exception
    -> API Error
```

保留原始异常链：

```python
raise KnowledgeRetrievalError(...) from exc
```

---

# 24. 数据版本与兼容规范

每次 Reasoning 输出至少绑定：

```text
knowledge_base_version
protocol_version
scoring_config_version
llm_model
embedding_model
reranker_model
```

知识库升级不可直接覆盖历史结果引用的版本。

历史 consultation 必须能说明当时使用的是哪个知识版本。

---

# 25. 代码测试规范

## 25.1 Unit Tests

覆盖：

- 三值模型；
- parser；
- score；
- relationship；
- differential；
- evidence gap；
- KnowledgeService；
- metadata filter；
- config validation；
- repository。

外部系统使用 fake/mock，不要求 unit test 连真实模型。

## 25.2 Integration Tests

覆盖：

```text
FastAPI -> Service -> LangGraph
KnowledgeService -> Embedding -> Milvus test collection
Repository -> PostgreSQL test database
interrupt -> resume
review -> persistence
```

## 25.3 E2E

Playwright 至少覆盖：

```text
医生录入病例
-> 启动辨证
-> 查看 Top-K
-> 查看 Evidence
-> Agent 请求补充证据
-> 医生补充
-> 同 thread 恢复
-> Top-K 更新
-> 医生审核
```

## 25.4 医学评测

代码通过测试不等于医学推理合格。

必须保留：

```text
398 条检索评测
53 条鉴别规则评测
范中林医案回放
专家病例 Top-K Recall
```

---

# 26. 质量门槛

Codex 每次完成任务前，按实际涉及模块执行：

## Backend

```bash
ruff check backend
black --check backend
pytest backend/tests
```

如果项目使用 mypy/pyright：

```bash
mypy backend/app
# 或
pyright backend/app
```

## Frontend

```bash
npm run lint
npm run typecheck
npm run test
```

关键流程：

```bash
npm run e2e
```

## Knowledge / RAG

```bash
python scripts/validate_knowledge.py
python eval/retrieval_eval.py
```

Codex 不得声称“完成”但不说明测试是否执行、是否通过。

---

# 27. Git 与变更范围规范

- 一个任务只修改与当前目标相关的文件；
- 禁止顺手格式化整个仓库；
- 禁止无关重命名；
- 数据库 migration 单独清晰；
- 配置变更同步 `.env.example`；
- API 变更同步 Schema 和前端类型；
- Reasoning Protocol 变更同步评测数据。

如果任务需要大规模重构，先提供计划和风险，再执行。

---

# 28. Codex 任务执行流程

复杂任务（跨 3 个以上模块、数据库迁移、Graph 结构改变、RAG 基础设施改变）必须按此流程：

## Step 1：阅读与定位

Codex 先确认：

- 相关开发文档章节；
- 相关 Reasoning Protocol；
- 现有代码；
- 现有测试；
- 可能受影响的 API/Schema。

## Step 2：先给实现计划

简要列出：

```text
目标
预计修改文件
接口变化
数据变化
测试计划
风险
```

## Step 3：最小纵向切片实现

避免一次重写全部系统。

优先：

```text
Schema
-> Adapter/Service
-> Node
-> Graph
-> API
-> Frontend
-> Test
```

## Step 4：验证

先运行局部测试，再运行受影响的回归测试。

## Step 5：交付说明

必须明确：

```text
1. 修改了什么
2. 为什么这样实现
3. 哪些文件变更
4. 执行了哪些测试
5. 测试结果
6. 未解决问题/风险
```

---

# 29. 禁止事项清单

Codex 严禁：

1. 伪造经典条文、方证、医案、Evidence ID；
2. 将缺失信息默认解释为阴性；
3. 把六经写成固定单标签六分类；
4. 把历史医案覆盖 canonical 知识；
5. 让 RAG 检索结果直接成为最终诊断；
6. 在 Router 中写医学逻辑；
7. 在 React 中复制后端医学规则；
8. 在 Node 中直接创建 pymilvus/PostgreSQL Client；
9. 硬编码具体模型厂商和密钥；
10. 自动向患者输出高风险药物剂量；
11. 保存或展示模型隐藏 CoT；
12. 用 `except Exception: pass` 吞错；
13. 为“能跑”跳过类型、异常和测试；
14. 大面积无关重构；
15. 未经说明修改既定目录和 Reasoning Protocol。

---

# 30. 推荐 pyproject 工具约束

可按项目实际版本调整，但风格建议如下：

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ASYNC"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

不要为了“lint 0 warning”配置过度严格且影响研发效率的规则。新增规则应有实际价值。

---

# 31. Provider / Infrastructure 依赖注入示例

所有外部能力统一由 `core/dependencies.py` / Application Container 构建。

```python
"""
应用依赖工厂。

职责：集中创建 LLM、Embedding、Reranker、Milvus VectorStore、
业务 Repository、LangGraph Checkpointer 与 KnowledgeService。
上层业务模块只依赖抽象能力，不感知具体厂商 SDK 或本地降级实现。
"""


def get_chat_model(settings: Settings):
    """根据 LLM_PROVIDER 配置创建 LangChain ChatModel。"""
    ...


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """创建 Embedding Provider。"""
    ...


def get_reranker(settings: Settings) -> Reranker:
    """创建重排器。"""
    ...


def get_vector_store(settings: Settings) -> VectorStorePort:
    """优先使用 Remote Milvus，不可用时按配置降级 Milvus Lite。"""
    ...


async def get_checkpointer(settings: Settings):
    """优先创建 PostgreSQL Checkpointer，未接入时降级 AsyncSqliteSaver。"""
    ...
```

`graph_factory.py` 接收已经创建好的依赖：

```python
def build_six_channel_graph(
    *,
    chat_model,
    knowledge_service: KnowledgeService,
    checkpointer,
):
    """装配六经辨证子图，所有外部依赖均由外部注入。"""
    ...
```

强制原则：

```text
Node / Agent
    -> Port / Service
        -> Adapter / Factory
            -> Remote 或 Local Fallback
```

而不是：

```text
Node
    -> if Milvus 可用...
    -> if PostgreSQL 可用...
```

测试中应能注入：

- FakeChatModel；
- FakeEmbeddingProvider；
- FakeReranker；
- InMemory/FakeVectorStore；
- AsyncSqliteSaver；
- Fake Repository。

这样单元测试不依赖 Docker Milvus 或 PostgreSQL，集成测试再分别覆盖真实基础设施分支。

---

# 32. 前端注释与组件示例

```tsx
/**
 * 医生补充证据输入条。
 *
 * 当后端进入 needs_evidence 状态时，展示高信息增益的补充项，
 * 并将医生补充结果提交给原 consultation/thread 继续推理。
 */
export function SupplementBar(props: SupplementBarProps) {
  // 前端只负责采集医生输入，不根据答案自行修改六经候选。
  ...
}
```

重要交互必须能看出：

- 当前系统状态；
- 是否等待补充；
- 是否已完成；
- 医生是否已审核；
- Evidence 是否来自经典、方证、鉴别或医案。

---

# 33. `.env.example` 最低要求

项目 `.env.example` 至少保留以下接口：

```text
APP_ENV
APP_HOST
APP_PORT
LOG_LEVEL

LLM_PROVIDER
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
LLM_TEMPERATURE
LLM_TIMEOUT_SECONDS
LLM_MAX_RETRIES

EMBEDDING_PROVIDER
EMBEDDING_BASE_URL
EMBEDDING_API_KEY
EMBEDDING_MODEL
EMBEDDING_DIMENSION
EMBEDDING_BATCH_SIZE

RERANKER_PROVIDER
RERANKER_BASE_URL
RERANKER_API_KEY
RERANKER_MODEL
RERANKER_TIMEOUT_SECONDS

MILVUS_MODE
MILVUS_URI
MILVUS_TOKEN
MILVUS_DATABASE
MILVUS_LITE_PATH
MILVUS_FALLBACK_TO_LITE
MILVUS_CONNECT_TIMEOUT_SECONDS
MILVUS_COLLECTION_CLASSIC
MILVUS_COLLECTION_FORMULA
MILVUS_COLLECTION_DIFFERENTIAL
MILVUS_COLLECTION_CASE
MILVUS_COLLECTION_FRAMEWORK
MILVUS_METRIC_TYPE
MILVUS_INDEX_TYPE
MILVUS_HNSW_M
MILVUS_HNSW_EF_CONSTRUCTION
MILVUS_SEARCH_EF

DATABASE_ENABLED
DATABASE_URL
DATABASE_POOL_SIZE
DATABASE_MAX_OVERFLOW
DATABASE_POOL_RECYCLE_SECONDS

CHECKPOINT_BACKEND
LANGGRAPH_DATABASE_URL
LANGGRAPH_SQLITE_PATH
CHECKPOINT_FALLBACK_TO_SQLITE
POSTGRES_CONNECT_TIMEOUT_SECONDS

KNOWLEDGE_BASE_VERSION
PROTOCOL_VERSION
SCORING_CONFIG_VERSION
TOP_K_DEFAULT
TOP_K_MAX
RETRIEVAL_TOP_K
RERANK_TOP_K
MAX_SUPPLEMENT_ROUNDS
SSE_HEARTBEAT_SECONDS
```

其中本地开发推荐：

```text
MILVUS_MODE=auto
MILVUS_FALLBACK_TO_LITE=true
CHECKPOINT_BACKEND=auto
CHECKPOINT_FALLBACK_TO_SQLITE=true
```

这样在 Docker Milvus/PostgreSQL 尚未启动时，Codex 仍可基于 `MilvusClient` Lite + `AsyncSqliteSaver` 完成主链路调试。

---

# 34. Definition of Done（Codex 完成定义）

一个编码任务只有同时满足以下条件才算完成：

- [ ] 严格遵守既定目录；
- [ ] 新脚本顶部有中文职责说明；
- [ ] 函数/重要逻辑有中文注释；
- [ ] 缩进和格式工具通过；
- [ ] 类型完整；
- [ ] 未硬编码模型/密钥/数据库/Collection；
- [ ] 未破坏 KnowledgeService 边界；
- [ ] 未破坏 Reasoning Protocol；
- [ ] 新增/修改功能有测试；
- [ ] 相关测试已实际运行；
- [ ] API 变更同步 Schema；
- [ ] 配置变更同步 `.env.example`；
- [ ] Milvus remote/lite 两条分支均有测试；
- [ ] Checkpoint postgres/sqlite 两条分支均有测试；
- [ ] 本地 Python 调试命令已在 Conda `langgraph` 环境实际执行；
- [ ] 数据库变更有 Alembic；
- [ ] 最终说明测试结果和遗留风险。

---

# 35. 首轮编码推荐顺序

在现有最终开发文档基础上，Codex 首轮实现建议仍严格遵循：

```text
Phase 0  工程骨架
    -> FastAPI / React / 基础设施接口 / health / OpenAPI
    -> Conda langgraph 环境验证
    -> Remote Milvus -> Milvus Lite fallback
    -> PostgreSQL Checkpointer -> AsyncSqliteSaver fallback

Phase 1  KnowledgeService
    -> 五类知识加载
    -> Embedding Adapter
    -> Milvus VectorStore Adapter（Remote/Lite 同契约）
    -> Reranker Adapter
    -> search_* / get_* 单测

Phase 2  Clinical Parser + Schema
    -> positive / negative / unknown
    -> 冲突检测

Phase 3  最小六经纵向切片
    -> 输入 -> ClinicalEvidence -> 六经候选 -> RAG -> Top-3
    -> FastAPI -> React -> DoctorReview

Phase 4  Differential + Evidence Gap
    -> 53 条鉴别
    -> interrupt/resume

Phase 5  Relation / Safety / KnowledgeQA

Phase 6  医学评测与专家校准
```

本版本基础设施策略正式固定为：

> **生产使用 Docker/Remote Milvus，远程不可用时按配置自动降级 Milvus Lite；生产优先 PostgreSQL Checkpointer，本地未接入或不可用时按配置降级 AsyncSqliteSaver。**

两类降级都只能发生在 Infrastructure Adapter / Factory 层，不允许改变 `KnowledgeService`、RAG Evidence Schema、Reasoning Protocol 和上层节点契约。

---

# 36. 最终编码原则

后续 Codex 每次工作都应重复检查以下 12 条：

1. 目录是契约；
2. 文件开头必须有中文职责说明；
3. 每个函数和关键逻辑必须有中文注释；
4. Python 4 空格、TS 2 空格，格式统一；
5. 外部 Provider 必须配置化、接口化；
6. Milvus/PostgreSQL/Lite/SQLite 的选择不得渗透医学业务节点；
7. KnowledgeService 是唯一知识访问边界；
8. Unknown 不等于 Negative；
9. 六经允许 Top-K、多标签、多关系；
10. 不保存隐藏 CoT，只保存结构化证据链；
11. 医生拥有最终决策权；
12. 没有测试和验证结果，不算完成。



---

# V1.1 更新摘要

相较 V1.0，本版本新增并固定：

1. Codex 本地 Python 调试统一使用 Anaconda `langgraph` 环境；
2. 生产向量库使用 Docker/Remote Milvus；
3. Remote Milvus 不可用时通过 `from pymilvus import MilvusClient` 自动降级 Milvus Lite；
4. LangGraph Checkpoint 生产优先 PostgreSQL；
5. PostgreSQL 未接入或不可访问时通过 `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver` 自动降级 SQLite checkpoint；
6. `/health` 必须暴露实际 vector/checkpoint backend 与 degraded 状态；
7. Remote/Lite、Postgres/SQLite 两组分支均纳入自动化测试；
8. 明确 `AsyncSqliteSaver` 不替代业务 PostgreSQL Repository。
