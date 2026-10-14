"""QA V2 API 契约测试；旧入口只提供迁移提示，经典读取保持可用。"""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_legacy_qa_is_gone_and_exact_knowledge_remains(tmp_path: Path) -> None:
    """旧 QA 返回 410，条文速查和精确来源仍由 V1 Knowledge API 提供。"""
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
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/knowledge/qa", json={"query": "第35条"})
        assert response.status_code == 410
        assert response.json()["error"]["code"] == "QA_ENDPOINT_DEPRECATED"
        exact = client.get("/api/v1/knowledge/clauses/35")
        assert exact.status_code == 200
        assert exact.json()["content"] == exact.json()["metadata"]["classic_text"]
        unavailable = client.post("/api/v2/knowledge/chat", json={"message": "第35条原文是什么？"})
        assert unavailable.status_code == 503
        assert unavailable.json()["error"]["code"] == "QA_CHAT_NOT_CONFIGURED"
