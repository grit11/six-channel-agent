"""S2 纵向回归：真实 API/知识/SQLite、阶段回放、分批信息隔离和关系病史修订。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableConfig

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.main import create_app
from backend.app.persistence.checkpoint import build_checkpoint_runtime
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.services.reasoning_service import ReasoningService


def test_replay_api_steps_do_not_leak_future_or_case_labels(tmp_path: Path) -> None:
    """新增信息逐步累积且独立线程；原医案结论不进入前步患者事实。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        retrieval_mode="bm25",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        replay = client.post(
            "/api/v1/reasoning/replay",
            json={
                "case_evidence_id": "1.2:case_0001",
                "steps": [
                    {"label": "初始", "text": "发热"},
                    {"label": "补充脉证", "text": "恶寒，无汗，脉浮紧"},
                ],
            },
        )
        assert replay.status_code == 200
        result = replay.json()
        assert len(result["steps"]) == 2
        first, second = [step["result"] for step in result["steps"]]
        assert first["thread_id"] != second["thread_id"]
        assert not any(
            fact["key"] == "pulse_tight" and fact["polarity"] == "positive"
            for fact in first["clinical_evidence"]["findings"]
        )
        assert any(
            fact["key"] == "pulse_tight" and fact["polarity"] == "positive"
            for fact in second["clinical_evidence"]["findings"]
        )
        assert result["historical_channels"] == ["太阳"]
        assert "六经" not in first["clinical_evidence"]["chief_complaint"]
        assert (
            client.post(
                "/api/v1/reasoning/replay",
                json={
                    "case_evidence_id": "1.2:core_0010",
                    "steps": [{"label": "输入", "text": "无汗"}],
                },
            ).status_code
            == 400
        )
        assert client.post("/api/v1/reasoning/replay", json={"steps": []}).status_code == 422
        created = client.post("/api/v1/consultations", json={"text": "恶寒，口渴"}).json()
        path = "/api/v1/consultations/" + created["consultation_id"]
        public = client.get(path + "/replay").json()
        assert public["status"] == "available"
        assert [stage["stage"] for stage in public["stages"]] == created["reasoning"][
            "completed_stages"
        ]
        assert public["stages"][-1]["stage"] == "evidence_topk"
        assert client.get(path).json() == created
        revised = client.post(
            path + "/evidence",
            json={
                "expected_turn": created["turn"],
                "clinical_evidence": {
                    "relationship_history": {
                        "source_text": "医生核对病史",
                        "history_complete": True,
                    }
                },
                "replace_context_fields": ["relationship_history"],
            },
        ).json()
        assert revised["thread_id"] == created["thread_id"]
        assert revised["clinical_evidence"]["findings"] == created["clinical_evidence"]["findings"]
        assert revised["clinical_evidence"]["relationship_history"]["history_complete"] is True
        assert len(revised["reasoning"]["stage_snapshots"]) == 5
        saved = client.post(
            path + "/evidence", json={"clinical_text": "脉浮", "resume": False}
        ).json()
        assert saved["reasoning"] is None
        assert client.get(path + "/replay").json()["status"] == "unavailable"


@pytest.mark.asyncio
async def test_stage_snapshots_survive_sqlite_restart_and_old_dto_defaults(tmp_path: Path) -> None:
    """重开真实 saver 后阶段/医案记录仍在；旧 DTO 默认空不补造数据。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    runtime = await build_checkpoint_runtime(settings)
    try:
        result = await ReasoningService(knowledge, runtime.checkpointer).analyze_text(
            "发热，无汗，脉浮紧"
        )
    finally:
        await runtime.close()
    runtime = await build_checkpoint_runtime(settings)
    try:
        graph = ReasoningService(knowledge, runtime.checkpointer).graph
        config: RunnableConfig = {"configurable": {"thread_id": result.thread_id}}
        state = await graph.aget_state(config)
        assert [item.sequence for item in state.values["stage_snapshots"]] == list(range(1, 6))
        assert state.values["similar_cases"] == result.similar_cases
        old = result.model_dump(exclude={"stage_snapshots", "similar_cases"})
        assert ReasoningResponse.model_validate(old).stage_snapshots == []
    finally:
        await runtime.close()
