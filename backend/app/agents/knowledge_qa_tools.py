"""QA V2 四个只读知识工具：严格校验、当前轮缓存和精确来源注册。

依赖唯一 KnowledgeService；不访问向量 SDK、临床状态或业务数据库。
来源文本是外部资料，工具只读取并返回短摘要，不执行其指令或生成诊断。
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.exceptions import ApplicationError, KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.knowledge import KnowledgeType

TOOL_TIMEOUT_SECONDS = 30.0


class QAError(ApplicationError):
    """QA 专用受控错误；沿用应用异常契约，HTTP 映射只发生在 QA Router。"""

    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        """保存安全错误码和消息，外部异常细节仅由异常链保留。"""
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ToolArgs(BaseModel):
    """拒绝多余字段、空白值和类型强制转换的只读工具参数基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class ExactSourceArgs(ToolArgs):
    """按现有查询接口可识别的条文编号或完整方名精确查询。"""

    kind: Literal["clause", "formula"]
    target: str = Field(min_length=1, max_length=100)


class SearchKnowledgeArgs(ToolArgs):
    """最多两类来源的定向检索，不接受任意数据库列名。"""

    query: str = Field(min_length=1, max_length=500)
    kinds: list[KnowledgeType] = Field(min_length=1, max_length=2)
    top_k: int = Field(default=5, ge=1, le=6)

    @field_validator("kinds")
    @classmethod
    def unique_kinds(cls, value: list[KnowledgeType]) -> list[KnowledgeType]:
        """拒绝重复来源类型，避免重复昂贵检索。"""
        if len(set(value)) != len(value):
            raise ValueError("知识类型不得重复")
        return value


class DifferentialArgs(ToolArgs):
    """只检索原始鉴别来源，不对患者进行规则评分。"""

    query: str = Field(min_length=1, max_length=500)
    candidate_names: list[str] = Field(default_factory=list, max_length=4)
    top_k: int = Field(default=5, ge=1, le=6)

    @field_validator("candidate_names")
    @classmethod
    def valid_candidates(cls, value: list[str]) -> list[str]:
        """拒绝空候选名和重复候选，避免无意义鉴别查询。"""
        if any(not item.strip() for item in value):
            raise ValueError("候选名称不能为空")
        return list(dict.fromkeys(item.strip() for item in value))


class EvidenceDetailsArgs(ToolArgs):
    """只允许展开当前轮工具已经返回的 ID。"""

    evidence_ids: list[str] = Field(min_length=1, max_length=5)


def evidence_title(item: Evidence) -> str:
    """从精确复核的真实元数据读取标题，不接受模型自造来源名。"""
    for key in ("formula_name", "title", "case_title", "clause_no"):
        if item.metadata.get(key):
            return str(item.metadata[key])
    return item.knowledge_id


@dataclass
class EvidenceRegistry:
    """每轮独立的引用 allowlist、缓存与调用审计；不保存跨请求可变全局状态。"""

    knowledge_version: str
    snapshot_digest: str = ""
    evidence: dict[str, Evidence] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    cache: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    tool_calls_used: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    markers: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    io_attempts: int = 0
    on_event: Callable[[str, dict[str, object]], Awaitable[None]] | None = None

    def register(self, items: list[Evidence], tool_name: str) -> list[dict[str, object]]:
        """给已精确复核的证据分配稳定 [n] 标记，重复来源不新增编号。"""
        cards: list[dict[str, object]] = []
        for item in items:
            if item.knowledge_version != self.knowledge_version:
                raise QAError("QA_INVALID_EVIDENCE_ID", "来源版本与当前知识快照不一致")
            self.evidence.setdefault(item.evidence_id, item.model_copy(deep=True))
            self.markers.setdefault(item.evidence_id, len(self.markers) + 1)
            cards.append(
                {
                    "ref": f"[{self.markers[item.evidence_id]}]",
                    "evidence_id": item.evidence_id,
                    "knowledge_type": item.knowledge_type,
                    "title": evidence_title(item),
                    "source": item.source,
                    "source_level": item.source_level,
                    "association_scope": item.metadata.get("association_scope"),
                    "excerpt": item.content[:600],
                    "truncated": len(item.content) > 600,
                }
            )
        return cards

    def resolve(self, evidence_ids: list[str]) -> list[Evidence]:
        """模型/用户猜测、旧轮或旧版本 ID 均不能用来遍历知识库。"""
        if any(item not in self.evidence for item in evidence_ids):
            raise QAError("QA_INVALID_EVIDENCE_ID", "Evidence ID 不属于当前 QA 轮次", 400)
        return [self.evidence[item] for item in dict.fromkeys(evidence_ids)]


class KnowledgeToolFactory:
    """为单轮 create_agent 绑定严格参数、超时、缓存和真实证据复核。"""

    def __init__(self, knowledge: KnowledgeService, registry: EvidenceRegistry) -> None:
        """注入唯一医学知识边界，不创建检索器或 SDK 客户端。"""
        self.knowledge = knowledge
        self.registry = registry
        self.registry.snapshot_digest = knowledge.snapshot_digest

    async def _read_with_retry(
        self, operation: Callable[[], Awaitable[list[Evidence]]]
    ) -> list[Evidence]:
        """只对明确的连接失败或 429/5xx 重试一次；共享外层工具和整轮预算。"""
        for attempt in range(2):
            self.registry.io_attempts += 1
            try:
                return await operation()
            except (ApplicationError, httpx.HTTPError) as exc:
                cause: BaseException | None = exc
                retryable = False
                visited: set[int] = set()
                while cause is not None and id(cause) not in visited:
                    visited.add(id(cause))
                    if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
                        retryable = True
                    if isinstance(cause, httpx.HTTPStatusError):
                        retryable = cause.response.status_code in {429, 500, 502, 503, 504}
                    cause = cause.__cause__
                if attempt or not retryable:
                    raise
                if self.registry.on_event:
                    await self.registry.on_event("qa.tool_retry", {"attempt": 1})
                await asyncio.sleep(0.2)
        raise RuntimeError("不可达的工具重试状态")

    async def _execute(
        self, name: str, args: ToolArgs, operation: Callable[[], Awaitable[list[Evidence]]]
    ) -> str:
        """单轮并行工具串行注册，缓存命中不重复 I/O；失败不静默改检索模式。"""
        key = json.dumps(
            [self.registry.snapshot_digest, name, args.model_dump()],
            ensure_ascii=False,
            sort_keys=True,
        )
        async with self.registry.lock:
            self.registry.tool_calls_used += 1
            if name not in self.registry.tools_used:
                self.registry.tools_used.append(name)
            if self.registry.tool_calls_used > 5:
                raise QAError("QA_TOOL_LIMIT_EXCEEDED", "本轮工具调用预算已耗尽")
            if key in self.registry.cache:
                self.registry.cache_hits += 1
                return self.registry.cache[key]
            started = perf_counter()
            if self.registry.on_event:
                await self.registry.on_event("qa.tool_started", {"name": name, "attempt": 0})
            try:
                async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
                    if self.knowledge.snapshot_digest != self.registry.snapshot_digest:
                        raise KnowledgeRetrievalError("知识快照在当前轮发生变化")
                    hits = await self._read_with_retry(operation)
                    # 每个 ID 精确复核后才注册，检索片段不能冒充原文。
                    ids = list(dict.fromkeys(item.evidence_id for item in hits))[:8]
                    originals = await self.knowledge.get_evidence_by_ids(ids) if ids else []
                    if {item.evidence_id for item in originals} != set(
                        ids
                    ) or self.knowledge.snapshot_digest != self.registry.snapshot_digest:
                        raise KnowledgeRetrievalError("检索来源与当前快照不一致")
                    cards = self.registry.register(originals, name)
                result: dict[str, object] = {
                    "status": "ok" if cards else "not_found",
                    "tool": name,
                    "items": cards,
                    "total_count": len(cards),
                }
            except QAError as exc:
                self.registry.errors.append(exc.code)
                result = {"status": "error", "tool": name, "error_code": exc.code, "items": []}
            except (TimeoutError, httpx.TimeoutException):
                self.registry.errors.append("QA_TOOL_TIMEOUT")
                result = {
                    "status": "error",
                    "tool": name,
                    "error_code": "QA_TOOL_TIMEOUT",
                    "items": [],
                }
            except (ApplicationError, httpx.HTTPError):
                self.registry.errors.append("QA_TOOL_UNAVAILABLE")
                result = {
                    "status": "error",
                    "tool": name,
                    "error_code": "QA_TOOL_UNAVAILABLE",
                    "items": [],
                }
            output = json.dumps(result, ensure_ascii=False)
            if self.registry.on_event:
                await self.registry.on_event(
                    "qa.tool_finished",
                    {
                        "name": name,
                        "duration_ms": int((perf_counter() - started) * 1000),
                        "source_count": result.get("total_count", 0),
                        "error_code": result.get("error_code"),
                    },
                )
            # 正常/空结果才缓存，暂时故障不阻止模型在预算内修正调用。
            if result["status"] != "error":
                self.registry.cache[key] = output
            return output

    async def lookup_exact_source(self, kind: str, target: str) -> str:
        """明确询问条文号或完整方名时调用；不存在不能用相似来源顶替。"""
        args = ExactSourceArgs(kind=cast(Literal["clause", "formula"], kind), target=target)

        async def read() -> list[Evidence]:
            item = (
                await self.knowledge.get_clause_by_id(args.target)
                if args.kind == "clause"
                else await self.knowledge.get_formula_by_name(args.target)
            )
            return [item] if item else []

        return await self._execute("lookup_exact_source", args, read)

    async def search_knowledge(self, query: str, kinds: list[KnowledgeType], top_k: int = 5) -> str:
        """解释、考据和框架问题使用最多两类定向检索；医案只用于用户明确问医案。"""
        args = SearchKnowledgeArgs.model_validate({"query": query, "kinds": kinds, "top_k": top_k})

        async def search() -> list[Evidence]:
            items: list[Evidence] = []
            for kind in args.kinds:
                if kind == "classic_clause":
                    items.extend(await self.knowledge.search_classic(args.query, top_k=args.top_k))
                elif kind == "formula_pattern":
                    items.extend(await self.knowledge.search_formula(args.query, top_k=args.top_k))
                elif kind == "six_channel_framework":
                    items.extend(await self.knowledge.search_core_framework(args.query, args.top_k))
                elif kind == "medical_case":
                    items.extend(await self.knowledge.search_case(args.query, top_k=args.top_k))
                else:
                    items.extend(
                        await self.knowledge.search_differential([], args.query, args.top_k)
                    )
            return items

        return await self._execute("search_knowledge", args, search)

    async def search_differential(
        self, query: str, candidate_names: list[str], top_k: int = 5
    ) -> str:
        """方证鉴别比较优先调用；原始鉴别资料不是当前患者诊断结论。"""
        args = DifferentialArgs(query=query, candidate_names=candidate_names, top_k=top_k)
        if any(not item.strip() or len(item) > 100 for item in args.candidate_names):
            raise QAError("QA_INVALID_ARGUMENTS", "候选名称必须为 1～100 字", 400)
        return await self._execute(
            "search_differential",
            args,
            lambda: self.knowledge.search_differential(
                args.candidate_names, args.query, args.top_k
            ),
        )

    async def get_evidence_details(self, evidence_ids: list[str]) -> str:
        """只展开当前轮实际工具已返回的 ID，不能猜 ID 扫描知识库。"""
        args = EvidenceDetailsArgs(evidence_ids=evidence_ids)

        async def read() -> list[Evidence]:
            return self.registry.resolve(args.evidence_ids)

        return await self._execute("get_evidence_details", args, read)

    def build(self) -> list[StructuredTool]:
        """构造仅四个只读工具，工具描述告诉模型何时调用、何时不要调用。"""
        specs = [
            ("lookup_exact_source", self.lookup_exact_source, ExactSourceArgs),
            ("search_knowledge", self.search_knowledge, SearchKnowledgeArgs),
            ("search_differential", self.search_differential, DifferentialArgs),
            ("get_evidence_details", self.get_evidence_details, EvidenceDetailsArgs),
        ]
        return [
            StructuredTool.from_function(
                coroutine=cast("Callable[..., Awaitable[Any]]", method),
                name=name,
                description=method.__doc__ or name,
                args_schema=cast("type[BaseModel]", schema),
                handle_validation_error="QA_INVALID_ARGUMENTS：请检查工具参数，不允许多余字段或超限值。",
            )
            for name, method, schema in specs
        ]
