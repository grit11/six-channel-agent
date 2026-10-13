"""V2 业务集成：真实图、SQLite 与 Repository 校验版本、幂等和审核边界。"""

from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.clinical_v2.schemas import HumanGraphCommandV2
from backend.app.core.exceptions import InvalidStateTransitionError
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.services.clinical_v2_service import ClinicalV2Service
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.runtime_service import ReliableRuntime
from backend.app.services.stream_service import StreamService
from backend.tests.v2.test_end_to_end_workflow import FakeKnowledge, FullLLM


async def test_v2_business_review_idempotency_and_stale_revision(tmp_path: Path) -> None:
    """重复动作不二次调用模型，过时审核拒绝，业务与图状态一致。"""
    repo = InMemoryConsultationRepository()
    llm = FullLLM()
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "graph.db")) as saver:
        workflow = ClinicalV2Service(llm, saver, FakeKnowledge())
        service = ConsultationService(repo, workflow, run_mode="sync")
        service.runtime = ReliableRuntime(service)
        paused = await service.create("发热", idempotency_key="create-1")
        assert paused.status == "awaiting_doctor_review"
        assert paused.turn == 2 and paused.paused_at == "s5_doctor_review"
        replay = await service.create("发热", idempotency_key="create-1")
        assert replay.consultation_id == paused.consultation_id
        assert len(llm.calls) == 3
        with pytest.raises(InvalidStateTransitionError):
            await service.create("发热头痛", idempotency_key="create-1")
        command = HumanGraphCommandV2(
            kind="review",
            consultation_id=str(paused.consultation_id),
            revision=paused.revision,
            action_turn=paused.turn,
            action_id="review-1",
            review={"action": "accept"},
        )
        completed = await service.resume(paused.consultation_id, command)
        assert completed.status == "completed"
        assert len(completed.review_history) == 1
        duplicate = await service.resume(paused.consultation_id, command)
        assert duplicate.consultation_id == completed.consultation_id
        assert (await repo.get(paused.consultation_id)).turn == completed.turn
        with pytest.raises(InvalidStateTransitionError, match="STALE_REVISION"):
            await service.resume(
                paused.consultation_id,
                command.model_copy(update={"action_id": "new"}),
            )
        events = await StreamService(repo.runtime).page(paused.consultation_id)
        assert any(e.event == "review_requested" for e in events)
        assert any(e.event == "consultation_completed" for e in events)
        assert all("发热" not in str(e.payload) for e in events)


async def test_v2_resume_rejects_cross_consultation(tmp_path: Path) -> None:
    """动作身份不能跨会诊，拒绝时业务轮次保持不变。"""
    from backend.app.core.exceptions import InvalidBusinessRequestError

    repo = InMemoryConsultationRepository()
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "g.db")) as saver:
        service = ConsultationService(
            repo,
            ClinicalV2Service(FullLLM(), saver, FakeKnowledge()),
            run_mode="sync",
        )
        service.runtime = ReliableRuntime(service)
        paused = await service.create("发热")
        with pytest.raises(InvalidBusinessRequestError):
            await service.resume(
                paused.consultation_id,
                HumanGraphCommandV2(
                    kind="review",
                    consultation_id=str(uuid4()),
                    revision=0,
                    action_turn=paused.turn,
                    action_id="bad",
                    review={"action": "accept"},
                ),
            )
        assert (await repo.get(paused.consultation_id)).turn == paused.turn
