"""QA V2 应用服务。

职责：管理 QA-only 临时会话、串行并发、Agent 调用和本轮引用核验。
边界：不创建 LangGraph、不使用临床 checkpoint、不写业务数据库，精确知识读取仍委托 KnowledgeService。
"""

import asyncio
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.core.exceptions import KnowledgeNotFoundError, KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.knowledge import (
    KnowledgeChatRequest,
    KnowledgeChatResponse,
    KnowledgeType,
    QACitation,
    QASourceCard,
    QAUsage,
)

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

    def __init__(self, knowledge: KnowledgeService, agent: KnowledgeQAAgent) -> None:
        """注入只读知识服务和 QA Agent；历史仅保留在当前进程内。"""
        self.knowledge = knowledge
        self.agent = agent
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

    async def chat(self, request: KnowledgeChatRequest) -> KnowledgeChatResponse:
        """执行单轮生成式 QA；同一会话串行，失败时不写入半截 assistant。"""
        started = perf_counter()
        if not self.configured:
            raise KnowledgeRetrievalError("QA_CHAT_NOT_CONFIGURED：请启用 KNOWLEDGE_QA_MODE=llm")
        session = self._get_session(str(request.session_id) if request.session_id else None)
        async with session.lock:
            history = [{"role": turn.role, "content": turn.content} for turn in session.history]
            turn_id = str(uuid4())
            result = await self.agent.run(request.message, history)
            evidence = result.registry.evidence
            markers = sorted({int(item) for item in _CITATION_RE.findall(result.answer)})
            invalid = [marker for marker in markers if marker < 1 or marker > len(evidence)]
            if not evidence:
                status = "no_evidence"
                answer = "当前知识库未检得充分可核验的依据，请补充检索词或查阅原典。"
                missing = ["本轮没有真实工具来源"]
            elif invalid or not markers:
                status = "partial"
                answer = result.answer + "\n\n本轮回答的引用未能全部核验。"
                missing = [f"未核验引用：[{item}]" for item in invalid]
                if not markers:
                    missing.append("模型未提供本轮来源标记")
            else:
                status = "answered"
                answer = result.answer
                missing = []
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
            session.history.extend(
                [
                    _Turn("user", request.message),
                    _Turn("assistant", answer, [item.evidence_id for item in evidence.values()]),
                ]
            )
            session.history = session.history[-(_MAX_HISTORY * 2) :]
            session.updated_at = perf_counter()
            self._sessions.move_to_end(session.session_id)
            usage_status = (
                "available"
                if any(value is not None for value in result.usage.values())
                else "unavailable"
            )
            return KnowledgeChatResponse(
                session_id=UUID(session.session_id),
                turn_id=UUID(turn_id),
                status=cast(Literal["answered", "partial", "no_evidence"], status),
                answer=answer,
                citations=citations,
                sources=sources,
                missing_evidence=missing,
                tools_used=result.registry.tools_used,
                knowledge_version=self.knowledge.version,
                usage=QAUsage(**result.usage, status=cast(Any, usage_status)),
                duration_ms=int((perf_counter() - started) * 1000),
            )

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


def _title(item: Evidence) -> str:
    """由真实 Evidence 元数据生成来源标题。"""
    for key in ("clause_no", "formula_name", "title", "case_title"):
        if item.metadata.get(key):
            return str(item.metadata[key])
    return item.evidence_id
