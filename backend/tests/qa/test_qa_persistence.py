"""QA 独立历史仓储测试：归属、幂等、并发占用和终态原子性。"""

from uuid import uuid4

import pytest

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.agents.knowledge_qa_tools import QAError
from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.qa_repository import QARepository
from backend.app.schemas.knowledge import KnowledgeChatRequest
from backend.app.services.knowledge_qa_service import KnowledgeQAService


@pytest.mark.asyncio
async def test_qa_repository_owner_and_idempotency() -> None:
    """不同主体不能读取会话；相同 client_message_id 不创建第二个运行。"""
    repository = QARepository()
    session = await repository.create("doctor-a", "测试", "1.2")
    with pytest.raises(QAError, match="QA 会话不存在"):
        await repository.get(session.session_id, "doctor-b")
    client_id = uuid4()
    first, created = await repository.begin(
        session.session_id,
        "doctor-a",
        client_id,
        "问题",
        seconds=10,
        model="test",
        prompt_digest="prompt",
        version="1.2",
    )
    assert created
    second, duplicate = await repository.begin(
        session.session_id,
        "doctor-a",
        client_id,
        "问题",
        seconds=10,
        model="test",
        prompt_digest="prompt",
        version="1.2",
    )
    assert not duplicate and second.message_id == first.message_id
    with pytest.raises(QAError, match="幂等键"):
        await repository.begin(
            session.session_id,
            "doctor-a",
            client_id,
            "另一问题",
            seconds=10,
            model="test",
            prompt_digest="prompt",
            version="1.2",
        )


@pytest.mark.asyncio
async def test_qa_unconfigured_chat_does_not_create_session() -> None:
    """模型未配置时 fail-closed，不能留下伪造历史。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
    with pytest.raises(KnowledgeRetrievalError):
        await service.chat(KnowledgeChatRequest(message="第35条"))
    assert service.repository.sessions == {}
