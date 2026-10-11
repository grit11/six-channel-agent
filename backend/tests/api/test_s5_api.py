"""S5 API 协议回归：真实固定图与事务收据，重复提交不重复会诊、补证和审核。"""

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_mutation_idempotency_events_and_invalid_cursors(tmp_path: Path) -> None:
    """重放已提交响应不重复人工历史，错用键/游标明确拒绝。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        payload = {"clinical_text": "发热，无汗，恶寒，身痛，喘，脉浮"}
        header = {"Idempotency-Key": "s5-create-0001"}
        created = client.post("/api/v1/consultations", json=payload, headers=header)
        repeated = client.post("/api/v1/consultations", json=payload, headers=header)
        assert created.status_code == repeated.status_code == 201
        assert created.json() == repeated.json()
        initial = created.json()
        path = "/api/v1/consultations/" + initial["consultation_id"]
        assert client.get("/api/v1/consultations").json()["total"] == 1
        assert (
            client.post(
                "/api/v1/consultations", json={"clinical_text": "其他"}, headers=header
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v1/consultations", json=payload, headers={"Idempotency-Key": "bad"}
            ).status_code
            == 400
        )
        evidence = {"clinical_text": "无汗，脉浮紧", "expected_turn": initial["turn"]}
        header = {"Idempotency-Key": "s5-evidence-0001"}
        added = client.post(path + "/evidence", json=evidence, headers=header)
        assert added.status_code == 200
        assert client.post(path + "/evidence", json=evidence, headers=header).json() == added.json()
        assert client.get(path).json()["supplementation_round"] == 1
        review = {
            "action": "accept",
            "expected_turn": added.json()["turn"],
            "selected_path_id": added.json()["reasoning"]["reasoning_paths"][0]["path_id"],
        }
        header = {"Idempotency-Key": "s5-review-0001"}
        final = client.post(path + "/review", json=review, headers=header)
        assert final.status_code == 200 and final.json()["status"] == "completed"
        assert client.post(path + "/review", json=review, headers=header).json() == final.json()
        assert len(client.get(path).json()["review_history"]) == 1
        events = client.get(path + "/events").json()["events"]
        while len(events) % 100 == 0:
            page = client.get(path + "/events", params={"cursor": events[-1]["event_id"]}).json()[
                "events"
            ]
            events.extend(page)
            if len(page) < 100:
                break
        assert events and events[0]["sequence"] == 1
        assert any(event["event"] == "stage_started" for event in events)
        assert any(event["event"] == "retrieval_completed" for event in events)
        assert all(payload["clinical_text"] not in str(event) for event in events)
        cursor = events[-1]["event_id"]
        assert client.get(path + "/events", params={"cursor": cursor}).json()["events"] == []
        stream = client.get(path + "/stream", headers={"Last-Event-ID": events[0]["event_id"]})
        assert stream.status_code == 200 and "event: stage_started" in stream.text
        assert f'id: {events[0]["event_id"]}\n' not in stream.text
        assert (
            client.get(path + "/stream", headers={"Last-Event-ID": f"{uuid4()}:1"}).status_code
            == 400
        )
