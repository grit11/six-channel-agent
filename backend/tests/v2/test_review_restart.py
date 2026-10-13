"""S5 重启恢复：只持久化公开状态，重开 saver 后医生审核不重新生成。"""

from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.clinical_v2.schemas import HumanGraphCommandV2
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.services.clinical_v2_service import ClinicalV2Service
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.runtime_service import ReliableRuntime
from backend.tests.v2.test_end_to_end_workflow import FakeKnowledge, FullLLM


async def test_review_after_sqlite_saver_restart(tmp_path: Path) -> None:
    """业务权威由同一 Repository 提供，重启仅验证真实图恢复。"""
    path = str(tmp_path / "state.db")
    repo = InMemoryConsultationRepository()
    llm = FullLLM()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        service = ConsultationService(
            repo, ClinicalV2Service(llm, saver, FakeKnowledge()), run_mode="sync"
        )
        service.runtime = ReliableRuntime(service)
        paused = await service.create("发热")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        service = ConsultationService(
            repo, ClinicalV2Service(llm, saver, FakeKnowledge()), run_mode="sync"
        )
        service.runtime = ReliableRuntime(service)
        completed = await service.resume(
            paused.consultation_id,
            HumanGraphCommandV2(
                kind="review",
                consultation_id=str(paused.consultation_id),
                revision=0,
                action_turn=paused.turn,
                action_id="after-restart",
                review={"action": "accept"},
            ),
        )
        assert completed.status == "completed"
        assert len(llm.calls) == 3
