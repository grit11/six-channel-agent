"""知识 API 单测：使用隔离的 BM25 配置验证 Router 只转发服务结果。"""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_classic_knowledge_api(tmp_path: Path) -> None:
    """经典精确条文 API 返回真实 Evidence，不暴露客户端对象。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        reranker_enabled=False,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/knowledge/classic",
            json={"query": "", "clause_nos": ["35"], "top_k": 1},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload[0]["knowledge_id"] == "clause_35"
        assert payload[0]["evidence_id"] == "1.2:clause_35"


def test_knowledge_api_rejects_unknown_fields(tmp_path: Path) -> None:
    """关键查询请求的未知字段必须由 Pydantic 拒绝。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/knowledge/classic", json={"query": "脉浮", "unexpected": True}
        )
        assert response.status_code == 422


def test_knowledge_api_preserves_scope_and_source_levels(tmp_path: Path) -> None:
    """方证范围、鉴别精确候选和医案来源等级均由服务层返回。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    with TestClient(create_app(settings)) as client:
        formula = client.post(
            "/api/v1/knowledge/formula",
            json={"query": "恶寒", "association_scope": ["classic_clause_based"], "top_k": 2},
        )
        differential = client.post(
            "/api/v1/knowledge/differential",
            json={"candidates": ["麻黄汤证", "桂枝汤证"], "top_k": 1},
        )
        case = client.post("/api/v1/knowledge/case", json={"query": "恶寒", "top_k": 1})
    assert formula.status_code == differential.status_code == case.status_code == 200
    assert formula.json()[0]["metadata"]["association_scope"] == "classic_clause_based"
    assert differential.json()[0]["retrieval_method"] == "candidate_pair_exact"
    assert case.json()[0]["source_level"] == "L5"
