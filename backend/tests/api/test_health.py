"""健康 API 测试，注入向量测试替身并使用真实 SQLite 生命周期。"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.rag.vector_store import VectorBackendStatus


def test_health_endpoints(tmp_path: Path) -> None:
    """正确的健康和版本路径必须暴露在 OpenAPI，无虚假连接声明。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoints.sqlite3"),
    )
    store = MagicMock()
    store.status = VectorBackendStatus("lite", True, "单测替身")
    store.close = AsyncMock()
    with (
        patch("backend.app.core.dependencies.build_milvus_store", return_value=store),
        TestClient(create_app(settings)) as client,
    ):
        assert client.get("/api/v1/health/live").json() == {"status": "ok"}
        health = client.get("/api/v1/health").json()
        assert health == {
            "status": "degraded",
            "vector_backend": "lite",
            "checkpoint_backend": "sqlite",
            "business_database": "disabled",
        }
        assert client.get("/api/v1/health/ready").status_code == 200
        assert client.get("/api/v1/system/versions").json()["knowledge_base_version"] == "1.2"
        versions = client.get("/api/v1/system/versions").json()
        assert versions["clinical_parser_mode"] == "rules"
        assert versions["reasoning_mode"] == "rules"
        assert versions["clinical_context_version"] == "clinical-context/v2.0"
        assert versions["relationship_version"] == "relationship/v2.0"
        assert versions["knowledge_qa_mode"] == "extractive"
        assert versions["knowledge_qa_version"] == "knowledge-qa/v1.0"
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/system/versions" in paths
        assert not any(".." in path for path in paths)
    store.close.assert_awaited_once()
