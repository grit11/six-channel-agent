"""QA 工具专项测试：参数、引用和 create_agent 工具循环。"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.agents.knowledge_qa_tools import (
    EvidenceDetailsArgs,
    EvidenceRegistry,
    ExactSourceArgs,
    QAError,
    SearchKnowledgeArgs,
)
from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService


def test_tool_argument_contracts() -> None:
    """严格 Schema 阻断空值、多余类型和重复来源。"""
    assert ExactSourceArgs(kind="clause", target="35")
    with pytest.raises(ValidationError):
        ExactSourceArgs(kind="clause", target="")
    with pytest.raises(ValidationError):
        SearchKnowledgeArgs(query="x", kinds=["classic_clause", "classic_clause"])
    with pytest.raises(ValidationError):
        EvidenceDetailsArgs(evidence_ids=[])


@pytest.mark.asyncio
async def test_registry_rejects_unknown_evidence() -> None:
    """当前轮 allowlist 之外的 Evidence ID 必须失败关闭。"""
    registry = EvidenceRegistry("1.2")
    with pytest.raises(QAError):
        registry.resolve(["1.2:forged"])


@pytest.mark.asyncio
async def test_create_agent_tool_calling() -> None:
    """离线模型仍必须通过真实 bind_tools 和工具循环。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )

    class ToolCallingFake(FakeMessagesListChatModel):
        """为测试提供真实 bind_tools 接口。"""

        def bind_tools(self, tools, **kwargs):
            """返回自身以继续执行预设 tool call 响应。"""
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
    agent = KnowledgeQAAgent(
        knowledge, ToolCallingFake(responses=[tool_call, AIMessage(content="依据见[1]")])
    )
    result = await agent.run("第35条原文是什么？")
    assert result.registry.tools_used == ["lookup_exact_source"]
    assert "1.2:clause_35" in result.registry.evidence
