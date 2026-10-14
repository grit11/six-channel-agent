"""QA V2 工具、引用和会话边界测试。"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.agents.knowledge_qa_tools import (
    DifferentialArgs,
    EvidenceDetailsArgs,
    EvidenceRegistry,
    ExactSourceArgs,
    QAError,
    SearchKnowledgeArgs,
)
from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeNotFoundError, KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.knowledge import KnowledgeChatRequest
from backend.app.services.knowledge_qa_service import KnowledgeQAService


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """加载只读测试知识快照。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


def test_tool_argument_contracts() -> None:
    """工具参数超限、重复类型和空候选必须在下游 I/O 前被拦截。"""
    assert ExactSourceArgs(kind="clause", target="35")
    with pytest.raises(ValidationError):
        ExactSourceArgs(kind="clause", target="")
    with pytest.raises(ValidationError):
        SearchKnowledgeArgs(query="x", kinds=["classic_clause", "classic_clause"])
    with pytest.raises(ValidationError):
        DifferentialArgs(query="x", candidate_names=[""])
    with pytest.raises(ValidationError):
        EvidenceDetailsArgs(evidence_ids=[])


@pytest.mark.asyncio
async def test_registry_rejects_unknown_evidence(knowledge: KnowledgeService) -> None:
    """Evidence 展开只允许本轮真实 tool output 注册的 ID。"""
    registry = EvidenceRegistry(knowledge.version)
    with pytest.raises(QAError):
        registry.resolve(["1.2:forged"])


@pytest.mark.asyncio
async def test_unconfigured_chat_fails_closed(knowledge: KnowledgeService) -> None:
    """未装配支持 Tool Calling 的模型时不得回落旧摘录回答。"""
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
    with pytest.raises(KnowledgeRetrievalError, match="QA_CHAT_NOT_CONFIGURED"):
        await service.chat(KnowledgeChatRequest(message="第35条原文是什么？"))


@pytest.mark.asyncio
async def test_expired_session_is_not_recovered(knowledge: KnowledgeService) -> None:
    """过期 QA 会话不借用其他会话或临床状态恢复。"""
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
    session = service._get_session(None)
    session.updated_at = 0
    with pytest.raises(KnowledgeNotFoundError, match="已过期"):
        service._get_session(session.session_id)


@pytest.mark.asyncio
async def test_create_agent_real_tool_loop(knowledge: KnowledgeService) -> None:
    """Fake ChatModel 仍必须经过 create_agent 的真实工具调用路径。"""

    class ToolCallingFake(FakeMessagesListChatModel):
        """为测试提供真实 bind_tools 接口的离线模型。"""

        def bind_tools(self, tools, **kwargs):
            """LangChain Agent 绑定工具后继续使用预设响应。"""
            return self

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup_exact_source",
                "args": {"kind": "clause", "target": "35"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    final = AIMessage(content="第35条可直接核对原文。[1]")
    agent = KnowledgeQAAgent(knowledge, ToolCallingFake(responses=[tool_call, final]))
    result = await agent.run("第35条原文是什么？")
    assert result.registry.tools_used == ["lookup_exact_source"]
    assert "1.2:clause_35" in result.registry.evidence
    assert result.answer.endswith("[1]")
