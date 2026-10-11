"""异步会诊 API 回归：立即接受、状态快照、同线程补证 202 与完成审核。"""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def wait_snapshot(client: TestClient, path: str) -> dict:
    """按有界期限获取任务结果，不用固定延时假定任务完成。"""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        snapshot = client.get(path).json()
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.02)
    raise AssertionError("异步会诊未在期限内离开 running")


def test_async_create_supplement_and_review(tmp_path: Path) -> None:
    """201 返回运行态，后续补证 202；结果与审核通过权威快照恢复。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="async",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/consultations", json={"clinical_text": "发热，无汗，恶寒，身痛，喘，脉浮"}
        )
        assert response.status_code == 201
        accepted = response.json()
        assert accepted["status"] == "running" and accepted["reasoning"] is None
        path = "/api/v1/consultations/" + accepted["consultation_id"]
        result = wait_snapshot(client, path)
        assert result["status"] == "needs_evidence"
        supplement = client.post(
            path + "/evidence",
            json={
                "clinical_text": "无汗，脉浮紧",
                "expected_turn": result["turn"],
            },
        )
        assert supplement.status_code == 202
        assert supplement.json()["reasoning"] is None
        updated = wait_snapshot(client, path)
        assert updated["thread_id"] == accepted["thread_id"]
        assert updated["reasoning"]["reasoning_paths"]
        reviewed = client.post(
            path + "/review",
            json={
                "action": "accept",
                "expected_turn": updated["turn"],
                "selected_path_id": updated["reasoning"]["reasoning_paths"][0]["path_id"],
            },
        )
        assert reviewed.status_code == 200 and reviewed.json()["status"] == "completed"
