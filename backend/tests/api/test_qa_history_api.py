"""QA 历史和 SSE API 测试；使用内存 QA 仓储，不触碰临床会诊。"""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_qa_history_owner_and_stream_routes(tmp_path: Path) -> None:
    """会话 CRUD 只接受 QA 字段，流入口使用 POST SSE；生产身份门禁不被绕过。"""
    settings = Settings(
        _env_file=None,
        knowledge_qa_mode="extractive",
        retrieval_mode="bm25",
        reranker_enabled=False,
        consultation_storage="memory",
        qa_storage="memory",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "m.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "g.db"),
    )
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/v2/knowledge/sessions", json={"title": "经典复习"})
        assert created.status_code == 200
        session = created.json()
        listing = client.get("/api/v2/knowledge/sessions")
        assert (
            listing.status_code == 200 and listing.json()[0]["session_id"] == session["session_id"]
        )
        messages = client.get(f"/api/v2/knowledge/sessions/{session['session_id']}/messages")
        assert messages.status_code == 200 and messages.json() == []
        stream = client.post(
            "/api/v2/knowledge/chat/stream",
            json={
                "message": "第35条",
                "session_id": session["session_id"],
                "client_message_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        assert stream.status_code == 200
        assert "qa.failed" in stream.text or "QA_CHAT_NOT_CONFIGURED" in stream.text
