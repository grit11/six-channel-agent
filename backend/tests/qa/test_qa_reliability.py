"""QA 真实 Agent、流式终态、工具限次重试和取消测试；仅使用合成问答和本地知识。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from backend.app.agents.knowledge_qa_agent import AgentTurnResult, KnowledgeQAAgent, visible_text
from backend.app.agents.knowledge_qa_tools import EvidenceRegistry, KnowledgeToolFactory, QAError
from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.qa_repository import QARepository
from backend.app.schemas.knowledge import KnowledgeChatRequest
from backend.app.services.knowledge_qa_service import KnowledgeQAService


class ToolModel(FakeMessagesListChatModel):
    """提供实际 create_agent 工具绑定的离线模型，不替代 Agent 编排。"""

    def bind_tools(self, tools, **kwargs):
        """沿用预设响应，允许 LangChain 真实调用只读工具。"""
        return self


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """加载只读真实来源，用于检测引用 allowlist。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


def test_empty_evidence_keeps_model_answer_and_marks_ungrounded() -> None:
    """正常空检索不能丢弃模型正文，也不能留下伪引用编号。"""
    registry = EvidenceRegistry("1.2")
    service = KnowledgeQAService(
        SimpleNamespace(version="1.2"),
        SimpleNamespace(configured=True),
    )
    result = service._verify(
        AgentTurnResult(
            answer="依据一般辨证经验可先从表里寒热分析。[1]",
            registry=registry,
            messages=[],
            usage={},
        ),
        uuid4(),
        uuid4(),
        0.0,
    )
    assert result.status == "no_evidence"
    assert result.grounding_status == "ungrounded"
    assert result.grounding_notice and "未检索到本地知识库" in result.grounding_notice
    assert result.answer == "依据一般辨证经验可先从表里寒热分析。"
    assert result.citations == [] and result.sources == []


def test_empty_evidence_with_tool_error_remains_unavailable() -> None:
    """工具真正失败时不能降级成无依据的通用模型回答。"""
    registry = EvidenceRegistry("1.2", errors=["lookup_exact_source: timeout"])
    service = KnowledgeQAService(
        SimpleNamespace(version="1.2"),
        SimpleNamespace(configured=True),
    )
    with pytest.raises(QAError, match="本轮检索失败"):
        service._verify(
            AgentTurnResult(
                answer="模型回答",
                registry=registry,
                messages=[],
                usage={},
            ),
            uuid4(),
            uuid4(),
            0.0,
        )


def model() -> ToolModel:
    """预设一次检索和一次有据回答，两次都有 provider usage。"""
    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    return ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_exact_source",
                        "args": {"kind": "clause", "target": "35"},
                        "id": "test-call",
                        "type": "tool_call",
                    }
                ],
                usage_metadata=usage,
            ),
            AIMessage(content="请核对本轮原文。[1]", usage_metadata=usage),
        ]
    )


@pytest.mark.asyncio
async def test_real_stream_usage_and_idempotency(knowledge: KnowledgeService) -> None:
    """实际 astream_events 最终消息可用；usage 汇总所有当轮调用且同键不再生成。"""
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge, model()))
    session = await service.repository.create("doctor-a", "测试", knowledge.version)
    request = KnowledgeChatRequest(
        session_id=session.session_id, message="第35条", client_message_id=uuid4()
    )
    events = [event async for event in service.stream(request, "doctor-a")]
    names = [name for name, _ in events]
    assert "qa.started" in names and "qa.tool_started" in names
    assert "qa.completed" in names and "qa.failed" not in names
    response = next(data for name, data in events if name == "qa.completed")
    assert response["usage"]["input_tokens"] == 20
    assert response["usage"]["llm_calls"] == 2
    assert response["usage"]["status"] == "available"
    duplicate = await service.chat(request, "doctor-a")
    assert str(duplicate.turn_id) == response["turn_id"]
    assert len(await service.repository.history(session.session_id, "doctor-a")) == 2
    assert service.repository.locks == {}


@pytest.mark.asyncio
async def test_publish_failure_does_not_rewrite_committed_turn(knowledge: KnowledgeService) -> None:
    """已完成事务后发布失败，回查仍为正式完成且不会覆盖成 failed。"""
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge, model()))
    session = await service.repository.create("a", "测试", knowledge.version)
    request = KnowledgeChatRequest(
        session_id=session.session_id, message="第35条", client_message_id=uuid4()
    )

    async def broken_publish(name: str, payload: dict[str, object]) -> None:
        """模拟最终核验事件发布时断线。"""
        if name == "qa.sources_verified":
            raise ConnectionError("synthetic disconnect")

    with pytest.raises(ConnectionError):
        await service.chat(request, "a", broken_publish)
    history = await service.repository.history(session.session_id, "a")
    assert history[-1].status == "completed" and history[-1].response is not None
    assert (await service.chat(request, "a")).answer == history[-1].content


@pytest.mark.asyncio
async def test_cancel_after_reservation_has_no_partial_answer(knowledge: KnowledgeService) -> None:
    """started 后取消写入 interrupted，释放 active_turn 和事务锁。"""
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge, model()))
    session = await service.repository.create("a", "测试", knowledge.version)

    async def cancel(name: str, payload: dict[str, object]) -> None:
        """在预留后模拟客户端取消，不允许静默继续模型。"""
        if name == "qa.started":
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await service.chat(
            KnowledgeChatRequest(
                session_id=session.session_id, message="第35条", client_message_id=uuid4()
            ),
            "a",
            cancel,
        )
    history = await service.repository.history(session.session_id, "a")
    assert history[-1].status == "interrupted" and history[-1].content == ""
    assert (await service.repository.get(session.session_id, "a")).active_turn_id is None


@pytest.mark.asyncio
async def test_tool_retry_cache_and_fake_id(knowledge: KnowledgeService) -> None:
    """仅确定的 503 重试一次；同参数缓存不重复 I/O，伪造 ID 不重试。"""
    registry = EvidenceRegistry(knowledge.version)
    factory = KnowledgeToolFactory(knowledge, registry)
    count = 0
    original = knowledge.get_clause_by_id

    async def fail_once(target: str):
        """模拟短暂上游服务错误。"""
        nonlocal count
        count += 1
        if count == 1:
            response = httpx.Response(503, request=httpx.Request("GET", "https://test"))
            raise httpx.HTTPStatusError("synthetic", request=response.request, response=response)
        return await original(target)

    knowledge.get_clause_by_id = fail_once
    first = await factory.lookup_exact_source("clause", "35")
    assert first == await factory.lookup_exact_source("clause", "35")
    assert count == 2 and registry.io_attempts == 2 and registry.cache_hits == 1
    result = await factory.get_evidence_details(["fake-id"])
    assert "QA_INVALID_EVIDENCE_ID" in result and registry.io_attempts == 3


@pytest.mark.asyncio
async def test_expired_running_is_persisted_and_unblocks_owner() -> None:
    """进程重启遗留的过期运行终态在回查时落库，同时清除会话占用。"""
    repository = QARepository()
    session = await repository.create("a", "测试", "1.2")
    turn, _ = await repository.begin(
        session.session_id,
        "a",
        uuid4(),
        "问题",
        seconds=1,
        model="fake",
        prompt_digest="test",
        version="1.2",
    )
    repository.messages[turn.message_id].expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(QAError):
        await repository.turn(turn.message_id, "b")
    assert (await repository.turn(turn.message_id, "a")).status == "interrupted"
    assert (await repository.get(session.session_id, "a")).active_turn_id is None


def test_provider_blocks_only_publish_text() -> None:
    """不同 provider 内容块只接收正文，不返回隐藏推理或工具参数。"""
    assert (
        visible_text(
            [
                {"type": "reasoning", "text": "hidden"},
                {"type": "text", "text": "正文"},
                {"type": "tool_call", "text": "private"},
            ]
        )
        == "正文"
    )
