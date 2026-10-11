"""会话 API 契约回归：验证状态资源路径与 SSE 事件格式。"""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_consultation_openapi_and_disabled_storage(tmp_path: Path) -> None:
    """业务存储关闭时不伪造创建成功，但路由契约仍出现在 OpenAPI。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/consultations", json={"text": "发热"}).status_code == 503
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/consultations" in paths
        assert "/api/v1/consultations/{consultation_id}/stream" in paths


def test_sse_missing_consultation_returns_domain_error(tmp_path: Path) -> None:
    """SSE 不应在资源不存在时发送伪造完成事件。"""
    settings = Settings(
        consultation_storage="memory",
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/consultations/00000000-0000-0000-0000-000000000000/stream")
    assert response.status_code == 404
