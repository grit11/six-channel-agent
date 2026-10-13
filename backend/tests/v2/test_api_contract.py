"""V2 API 契约测试：版本路径和旧评分字段隔离。"""

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.api.v2.clinical import ClinicalParseRequest
from backend.app.api.v2.router import router
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.services.clinical_v2_service import ClinicalV2Service
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.runtime_service import ReliableRuntime
from backend.app.services.stream_service import StreamService
from backend.tests.v2.test_end_to_end_workflow import FakeKnowledge, FullLLM
from backend.tests.v2.test_s2_hitl_resume import UncertainLLM


def test_v2_routes_and_input_contract() -> None:
    """V2 路由使用独立前缀，输入 Schema 拒绝旧评分字段。"""
    paths = {route.path for route in router.routes}
    assert "/api/v2/clinical/parse" in paths
    assert "/api/v2/consultations/{consultation_id}" in paths
    assert "score" not in ClinicalParseRequest.model_fields


def test_v2_api_parse_pause_review_and_sse(tmp_path: Path) -> None:
    """实际 API 执行唯一图，审核前保持暂停，过期/跨会诊游标明确拒绝。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        reranker_enabled=False,
        database_enabled=False,
        consultation_storage="memory",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "m.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "g.db"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        container = app.state.container
        llm = FullLLM()
        workflow = ClinicalV2Service(
            llm, container.checkpoint_runtime.checkpointer, FakeKnowledge()
        )
        service = container.consultation_service
        service.workflow = workflow
        service.run_mode = "sync"
        container.clinical_v2_service = workflow
        parsed = client.post("/api/v2/clinical/parse", json={"text": "发热"})
        assert parsed.status_code == 200 and len(llm.calls) == 1
        assert (
            client.post("/api/v2/clinical/parse", json={"text": "发热", "score": 1}).status_code
            == 422
        )
        created = client.post("/api/v2/consultations", json={"clinical_text": "发热"}).json()
        identifier = created["consultation_id"]
        assert created["status"] == "awaiting_doctor_review"
        assert created["paused_at"] == "s5_doctor_review"
        assert (
            client.post("/api/v1/consultations", json={"clinical_text": "发热"}).status_code == 404
        )
        malformed = {
            "kind": "review",
            "consultation_id": identifier,
            "revision": 0,
            "action_turn": created["turn"],
            "action_id": "bad",
            "review": {"action": "bad"},
        }
        assert (
            client.post(f"/api/v2/consultations/{identifier}/review", json=malformed).status_code
            == 400
        )
        events = client.get(f"/api/v2/consultations/{identifier}/events")
        assert events.status_code == 200
        event_rows = [
            json.loads(line[6:]) for line in events.text.splitlines() if line.startswith("data: ")
        ]
        assert "review_requested" in {e["event"] for e in event_rows}
        stages = {e["payload"].get("stage") for e in event_rows if e["event"] == "stage_completed"}
        assert {"s1_extract", "s2_localize", "s3_retrieve", "s4_compare"} <= stages
        assert all(
            e["payload"]["duration_ms"] >= 0 for e in event_rows if e["event"] == "stage_completed"
        )
        page_url = f"/api/v2/consultations/{identifier}/events/page"
        assert client.get(page_url).json() == event_rows
        assert (
            client.get(page_url, params={"after": event_rows[0]["event_id"]}).json()
            == event_rows[1:]
        )
        assert client.get(page_url, params={"after": f"{uuid4()}:1"}).status_code == 400
        assert client.get(f"/api/v2/consultations/{uuid4()}/events/page").status_code == 404
        last_id = event_rows[-1]["event_id"]
        assert (
            client.get(
                f"/api/v2/consultations/{identifier}/events", headers={"Last-Event-ID": last_id}
            ).text
            == ""
        )
        assert (
            client.get(
                f"/api/v2/consultations/{identifier}/events",
                headers={"Last-Event-ID": f"{uuid4()}:1"},
            ).status_code
            == 400
        )
        command = {**malformed, "action_id": "accept", "review": {"action": "accept"}}
        completed = client.post(f"/api/v2/consultations/{identifier}/review", json=command).json()
        assert completed["status"] == "completed" and len(completed["review_history"]) == 1
        saved_events = client.get(page_url).json()
        assert len({e["event_id"] for e in saved_events}) == len(saved_events)
        assert any(e["event"] == "consultation_completed" for e in saved_events)
        assert (
            client.post(
                f"/api/v2/consultations/{identifier}/review", json={**command, "action_id": "stale"}
            ).status_code
            == 409
        )

        async def append_public_events() -> None:
            """隔离存储追加无病例的事件，验证 API 分页边界与尾游标。"""
            from uuid import UUID

            for _ in range(105):
                await service.repository.runtime.emit(UUID(identifier), 1, "heartbeat", {})

        client.portal.call(append_public_events)
        first_page = client.get(page_url).json()
        assert len(first_page) == 100
        next_page = client.get(page_url, params={"after": first_page[-1]["event_id"]}).json()
        assert next_page and next_page[0]["sequence"] > first_page[-1]["sequence"]
        assert client.get(page_url, params={"after": next_page[-1]["event_id"]}).json() == []


async def test_v2_sse_cursor_at_pause_finishes(tmp_path: Path) -> None:
    """暂停快照已事务提交时，尾游标重连不等待永远不会发生的旧轮事件。"""
    repo = InMemoryConsultationRepository()
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "s.db")) as saver:
        service = ConsultationService(
            repo, ClinicalV2Service(UncertainLLM(), saver), run_mode="sync"
        )
        service.runtime = ReliableRuntime(service)
        snapshot = await service.create("难受")
        events = await repo.runtime.events(snapshot.consultation_id, 0)
        stream = StreamService(repo.runtime).stream(snapshot.consultation_id, events[-1].sequence)
        assert [item async for item in stream] == []
