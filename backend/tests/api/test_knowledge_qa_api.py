"""独立 QA API 回归：精确来源、未知来源和临床状态隔离。"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_qa_api_exact_sources_and_no_clinical_execution(tmp_path: Path) -> None:
    """知识中心不依赖临床模型，所有引用精确展开且缺失返回 404。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        reranker_enabled=False,
        consultation_storage="memory",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "m.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "g.db"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        service = app.state.container.consultation_service
        service.create = AsyncMock(side_effect=AssertionError("QA 不许启动临床图"))
        for query in ("第35条", "三物小白散的主症", "桂枝汤证与麻黄汤证怎么鉴别", "范中林无汗医案"):
            response = client.post("/api/v1/knowledge/qa", json={"query": query, "top_k": 1})
            assert response.status_code == 200 and response.json()["sections"]
        service.create.assert_not_called()
        assert app.state.container.knowledge_qa_service.graph.checkpointer is False
        exact = client.get("/api/v1/knowledge/clauses/35").json()
        assert exact["content"] == exact["metadata"]["classic_text"]
        assert (client.get("/api/v1/knowledge/evidence/" + exact["evidence_id"]).json())[
            "content"
        ] == exact["content"]
        formula = client.get("/api/v1/knowledge/formulas/三物小白散").json()
        assert formula["metadata"]["risk_flags"] == ["巴豆"]
        for path in ("clauses/9999", "formulas/不存在方剂", "evidence/1.2:missing"):
            missing = client.get("/api/v1/knowledge/" + path)
            assert missing.status_code == 404
        assert (
            client.post(
                "/api/v1/knowledge/qa", json={"query": "第35条", "thread_id": "patient"}
            ).status_code
            == 422
        )
        assert (
            client.post("/api/v1/knowledge/qa", json={"query": "第9999条"}).json()["status"]
            == "no_evidence"
        )


@pytest.mark.asyncio
async def test_independent_qa_graph_query_isolation() -> None:
    """QA 无患者状态、不使用临床 saver，前后两次查询不能共享来源。"""
    from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
    from backend.app.knowledge.service import KnowledgeService
    from backend.app.schemas.knowledge import KnowledgeQARequest
    from backend.app.services.knowledge_qa_service import KnowledgeQAService

    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
    first = await service.ask(KnowledgeQARequest(query="第35条", context_text="无汗脉浮紧"))
    second = await service.ask(KnowledgeQARequest(query="第9999条"))
    assert first.context_used and not second.context_used
    assert first.query_id != second.query_id and second.evidence_catalog == []
