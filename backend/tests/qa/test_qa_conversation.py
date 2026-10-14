"""QA 会话过期与隔离专项测试。"""

import pytest

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeNotFoundError
from backend.app.knowledge.service import KnowledgeService
from backend.app.services.knowledge_qa_service import KnowledgeQAService


@pytest.mark.asyncio
async def test_expired_session_is_not_recovered() -> None:
    """过期 QA 会话不借用其他会话或临床状态恢复。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
    session = service._get_session(None)
    session.updated_at = 0
    with pytest.raises(KnowledgeNotFoundError, match="已过期"):
        service._get_session(session.session_id)
