"""独立知识 API/主图集成：精确读取、真实层级及临床状态隔离，不使用在线模型。"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_qa_api_preserves_consultation_and_exact_sources(tmp_path: Path) -> None:
    """知识问题不触发临床分析或修改暂停会诊，精确 GET 未命中返回统一 404。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        reranker_enabled=False,
        consultation_storage="memory",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post("/api/v1/consultations", json={"text": "无汗恶寒发热脉浮"}).json()
        url = "/api/v1/consultations/" + created["consultation_id"]
        before = client.get(url).json()
        service = app.state.container.knowledge_qa_service
        workflow = app.state.container.consultation_service.workflow
        assert workflow is not None
        original_reason = workflow.analyze
        workflow.analyze = AsyncMock(side_effect=AssertionError("不许进入患者推理"))
        try:
            for query in [
                "第35条",
                "三物小白散的主症",
                "桂枝汤证与麻黄汤证怎么鉴别",
                "范中林无汗医案",
            ]:
                response = client.post("/api/v1/knowledge/qa", json={"query": query, "top_k": 1})
                assert response.status_code == 200
                assert response.json()["sections"]
            assert client.get(url).json() == before
            assert service.graph.checkpointer is False
        finally:
            workflow.analyze = original_reason
        exact = client.get("/api/v1/knowledge/clauses/35")
        assert exact.status_code == 200
        assert exact.json()["content"] == exact.json()["metadata"]["classic_text"]
        formula = client.get("/api/v1/knowledge/formulas/三物小白散")
        assert formula.status_code == 200 and formula.json()["metadata"]["risk_flags"] == ["巴豆"]
        for path in ["clauses/9999", "formulas/不存在方剂"]:
            missing = client.get("/api/v1/knowledge/" + path)
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "KNOWLEDGE_NOT_FOUND"
        rejected = client.post(
            "/api/v1/knowledge/qa", json={"query": "第35条", "thread_id": created["thread_id"]}
        )
        assert rejected.status_code == 422
        assert (
            client.post("/api/v1/knowledge/search/classic", json={"clause_nos": ["35"]}).status_code
            == 200
        )
        missing_qa = client.post("/api/v1/knowledge/qa", json={"query": "第9999条"}).json()
        assert missing_qa["status"] == "no_evidence"


@pytest.mark.asyncio
async def test_main_intent_routes_and_query_state_isolation() -> None:
    """主图只接固定意图；QA 前后查询无状态历史、不持有患者字段或 saver。"""
    from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
    from backend.app.graph.main_graph import route_main_intent
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
    assert first.query_id != second.query_id
    assert second.evidence_catalog == []
    assert route_main_intent("clinical") == "clinical"
    assert route_main_intent("knowledge") == "knowledge"
    from backend.app.core.exceptions import InvalidBusinessRequestError

    with pytest.raises(InvalidBusinessRequestError):
        route_main_intent("自由临床指令")
