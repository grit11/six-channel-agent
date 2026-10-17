"""QA V2 应用服务。

职责：管理独立 QA 历史、幂等运行、受控流式输出和本轮引用核验。
边界：不创建 LangGraph、不使用临床 checkpoint，精确读取仍委托 KnowledgeService。
"""

import asyncio
import hashlib
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from backend.app.agents.knowledge_qa_agent import PROMPT_PATH, AgentTurnResult, KnowledgeQAAgent
from backend.app.agents.knowledge_qa_tools import QAError
from backend.app.core.exceptions import KnowledgeNotFoundError, KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.qa_repository import QARepository
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.knowledge import (
    KnowledgeChatRequest,
    KnowledgeChatResponse,
    KnowledgeType,
    QACitation,
    QASourceCard,
    QAUsage,
)
from backend.app.services.qa_context import build_qa_context

_CITATION_RE = re.compile(r"\[(\d+)\]")
_MAX_SESSIONS = 100
_TTL_SECONDS = 30 * 60
_MAX_HISTORY = 4


@dataclass
class _Turn:
    """QA 会话中保存的最小 user/assistant 历史。"""

    role: str
    content: str
    citation_ids: list[str] = field(default_factory=list)


@dataclass
class _Session:
    """仅进程内短期 QA 会话，不与临床 thread 或患者主键关联。"""

    session_id: str
    updated_at: float
    history: list[_Turn] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class KnowledgeQAService:
    """QA-only 会话、Agent 运行及证据引用核验边界。"""

    def __init__(
        self,
        knowledge: KnowledgeService,
        agent: KnowledgeQAAgent,
        repository: QARepository | None = None,
        *,
        total_seconds: float = 180,
        context_tokens: int = 3500,
        heartbeat_seconds: float = 10,
    ) -> None:
        """注入单一 QA 历史权威和模型预算；旧 TTL helper 只供兼容测试读取。"""
        self.knowledge = knowledge
        self.agent = agent
        self.repository = repository or QARepository()
        self.total_seconds = total_seconds
        self.context_tokens = context_tokens
        self.heartbeat_seconds = heartbeat_seconds
        self._sessions: OrderedDict[str, _Session] = OrderedDict()

    @property
    def configured(self) -> bool:
        """返回 QA ChatModel 是否已装配。"""
        return self.agent.configured

    def _get_session(self, session_id: str | None) -> _Session:
        """生成或取得未过期 QA 会话，并执行 LRU 淘汰。"""
        now = perf_counter()
        if session_id is not None:
            session = self._sessions.get(session_id)
            if session is None or now - session.updated_at > _TTL_SECONDS:
                self._sessions.pop(session_id, None)
                raise KnowledgeNotFoundError("QA 临时会话已过期，请新建会话")
            self._sessions.move_to_end(session_id)
            return session
        new_id = str(uuid4())
        session = _Session(new_id, now)
        self._sessions[new_id] = session
        while len(self._sessions) > _MAX_SESSIONS:
            self._sessions.popitem(last=False)
        return session

    async def chat(
        self,
        request: KnowledgeChatRequest,
        owner: str = "dev:synthetic",
        on_event: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
    ) -> KnowledgeChatResponse:
        """事务预留单轮后执行 Agent，引用验证后原子完成；断线不留半截回答。"""
        started = perf_counter()
        if not self.configured:
            raise KnowledgeRetrievalError("QA_CHAT_NOT_CONFIGURED：请启用 KNOWLEDGE_QA_MODE=llm")
        session = (
            await self.repository.get(request.session_id, owner)
            if request.session_id
            else await self.repository.create(owner, request.message[:40], self.knowledge.version)
        )
        history = build_qa_context(
            await self.repository.recent(session.session_id, owner), self.context_tokens
        )
        prompt = await asyncio.to_thread(PROMPT_PATH.read_bytes)
        turn, created = await self.repository.begin(
            session.session_id,
            owner,
            request.client_message_id or uuid4(),
            request.message,
            seconds=self.total_seconds,
            model=str(getattr(self.agent.model, "model_name", "unavailable")),
            prompt_digest=hashlib.sha256(prompt).hexdigest(),
            version=self.knowledge.version,
        )
        if not created:
            if on_event:
                await on_event(
                    "qa.started",
                    {
                        "session_id": str(session.session_id),
                        "turn_id": str(turn.message_id),
                        "status": turn.status,
                    },
                )
            if turn.response:
                if on_event:
                    await on_event(
                        "qa.completed" if turn.response.status != "partial" else "qa.partial",
                        turn.response.model_dump(mode="json"),
                    )
                return turn.response
            raise QAError(
                turn.error_code or "QA_TURN_RUNNING",
                "该问题已有运行记录，请查询状态或使用新键显式重试",
                409,
            )
        try:
            if on_event:
                await on_event(
                    "qa.started",
                    {
                        "session_id": str(session.session_id),
                        "turn_id": str(turn.message_id),
                        "status": turn.status,
                    },
                )
            async with asyncio.timeout(self.total_seconds):
                result = (
                    await self.agent.run(request.message, history, on_event=on_event)
                    if on_event
                    else await self.agent.run(request.message, history)
                )
                response = self._verify(result, session.session_id, turn.message_id, started)
                await self.repository.finish(session.session_id, owner, turn.message_id, response)
        except asyncio.CancelledError:
            await self.repository.finish(
                session.session_id,
                owner,
                turn.message_id,
                error="QA_RUN_INTERRUPTED",
                interrupted=True,
            )
            raise
        except Exception as exc:
            code = (
                "QA_STAGE_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else exc.code if isinstance(exc, QAError) else "QA_PROVIDER_UNAVAILABLE"
            )
            await self.repository.finish(session.session_id, owner, turn.message_id, error=code)
            raise QAError(code, "QA 运行未完成，请查询状态后重试") from exc
        # 事务终态是权威结果。发布失败只能由客户端回查，不能再次 finish 或重新调用模型。
        if on_event:
            await on_event(
                "qa.sources_verified",
                {"citations": [c.model_dump(mode="json") for c in response.citations]},
            )
            await on_event(
                "qa.partial" if response.status == "partial" else "qa.completed",
                response.model_dump(mode="json"),
            )
        return response

    def _verify(
        self, result: AgentTurnResult, session_id: UUID, turn_id: UUID, started: float
    ) -> KnowledgeChatResponse:
        """确定性验证来源身份，不将有效编号冒充语义忠实性专家结论。"""
        evidence = result.registry.evidence
        markers = sorted({int(item) for item in _CITATION_RE.findall(result.answer)})
        invalid = [marker for marker in markers if marker < 1 or marker > len(evidence)]
        if not evidence:
            if result.registry.errors:
                raise QAError("QA_TOOL_UNAVAILABLE", "本轮检索失败，不能生成无据医学回答")
            status = "no_evidence"
            # 工具调用成功但返回空集时保留模型正文；明确标记未接地，不能制造引用。
            answer = _remove_unverified_markers(result.answer).strip()
            if not answer:
                answer = "模型未生成可展示的回答。"
            missing = ["本轮没有真实工具来源"]
            grounding_status = "ungrounded"
            grounding_notice = (
                "本回答未检索到本地知识库中可核验的依据，内容由模型基于通用知识生成，仅供参考。"
            )
        elif invalid or not markers or result.registry.errors:
            status = "partial"
            grounding_status = "partial"
            grounding_notice = "本轮回答仅部分内容与本地知识库来源完成核验，请结合来源卡复核。"
            answer = result.answer + "\n\n本轮回答的引用未能全部核验。"
            missing = [f"未核验引用：[{item}]" for item in invalid]
            if not markers:
                missing.append("模型未提供本轮来源标记")
            missing.extend(result.registry.errors)
        else:
            status = "answered"
            answer = result.answer
            missing = []
            grounding_status = "grounded"
            grounding_notice = None
        citations: list[QACitation] = []
        sources: list[QASourceCard] = []
        for index, item in enumerate(evidence.values(), start=1):
            title = _title(item)
            citations.append(
                QACitation(
                    marker=f"[{index}]",
                    evidence_id=item.evidence_id,
                    knowledge_type=cast(KnowledgeType, item.knowledge_type),
                    source_level=item.source_level,
                    title=title,
                )
            )
            sources.append(
                QASourceCard(
                    evidence_id=item.evidence_id,
                    title=title,
                    knowledge_type=cast(KnowledgeType, item.knowledge_type),
                    source=item.source,
                    source_level=item.source_level,
                    excerpt=item.content[:600],
                    association_scope=cast(
                        str | None,
                        (
                            item.metadata.get("association_scope")
                            if isinstance(item.metadata.get("association_scope"), str)
                            else None
                        ),
                    ),
                )
            )
        usage_status = (
            "available"
            if result.llm_calls and result.covered_calls == result.llm_calls
            else "partial" if result.covered_calls else "unavailable"
        )
        return KnowledgeChatResponse(
            session_id=session_id,
            turn_id=turn_id,
            status=cast(Literal["answered", "partial", "no_evidence"], status),
            grounding_status=cast(Literal["grounded", "partial", "ungrounded"], grounding_status),
            grounding_notice=grounding_notice,
            answer=answer,
            citations=citations,
            sources=sources,
            missing_evidence=missing,
            tools_used=result.registry.tools_used,
            tool_audit={
                "calls": result.registry.tool_calls_used,
                "cache_hits": result.registry.cache_hits,
                "io_attempts": result.registry.io_attempts,
            },
            knowledge_version=self.knowledge.version,
            usage=QAUsage(
                **result.usage,
                status=cast(Any, usage_status),
                llm_calls=result.llm_calls,
                covered_calls=result.covered_calls,
                time_to_first_token_ms=result.time_to_first_token_ms,
            ),
            duration_ms=int((perf_counter() - started) * 1000),
        )

    async def stream(
        self, request: KnowledgeChatRequest, owner: str
    ) -> AsyncIterator[tuple[str, dict[str, object]]]:
        """受监督流：断线取消任务并提交 interrupted；心跳不延长模型总预算。"""
        queue: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue(maxsize=64)

        async def emit(name: str, payload: dict[str, object]) -> None:
            await queue.put((name, payload))

        task = asyncio.create_task(self.chat(request, owner, emit))
        try:
            while not task.done() or not queue.empty():
                try:
                    yield await asyncio.wait_for(queue.get(), self.heartbeat_seconds)
                except TimeoutError:
                    if not task.done():
                        yield "heartbeat", {}
            await task
        except QAError as exc:
            yield "qa.failed", {"error_code": exc.code, "message": exc.message}
        except KnowledgeRetrievalError:
            yield "qa.failed", {"error_code": "QA_CHAT_NOT_CONFIGURED"}
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def ask(self, request):
        """兼容旧内部调用者；旧 HTTP 入口由 Router 返回 410。"""
        return await self.chat(KnowledgeChatRequest(message=request.query))

    async def get_clause(self, clause_id: str) -> Evidence:
        """精确读取条文，未命中不转向模糊检索。"""
        evidence = await self.knowledge.get_clause_by_id(clause_id)
        if evidence is None:
            raise KnowledgeNotFoundError("未找到指定条文")
        return evidence

    async def get_formula(self, formula_name: str) -> Evidence:
        """精确读取完整方名，保持原来源和风险字段。"""
        evidence = await self.knowledge.get_formula_by_name(formula_name)
        if evidence is None:
            raise KnowledgeNotFoundError("未找到指定方名")
        return evidence


def _remove_unverified_markers(answer: str) -> str:
    """空证据回答不保留可能被误解为本轮来源的编号标记。"""
    return _CITATION_RE.sub("", answer)


def _title(item: Evidence) -> str:
    """由真实 Evidence 元数据生成来源标题。"""
    for key in ("clause_no", "formula_name", "title", "case_title"):
        if item.metadata.get(key):
            return str(item.metadata[key])
    return item.evidence_id
