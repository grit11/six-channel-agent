"""V2 后台任务集成：提交后重建运行协调器，恢复已提交任务。"""

from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.runtime import RunJob
from backend.app.services.clinical_v2_service import ClinicalV2Service
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.runtime_service import ReliableRuntime
from backend.tests.v2.test_end_to_end_workflow import FakeKnowledge, FullLLM


async def test_committed_job_survives_worker_restart(tmp_path: Path) -> None:
    """调度实例重建后执行同一个持久意图，不创建第二个会诊。"""
    repository = InMemoryConsultationRepository()
    llm = FullLLM()
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as saver:
        service = ConsultationService(repository, ClinicalV2Service(llm, saver, FakeKnowledge()))
        first = ReliableRuntime(service)
        service.runtime = first
        # 模拟提交成功但进程来不及启动任务；不修改业务状态为假成功。
        first.launch = lambda job: None
        running = await service.create("发热")
        assert running.status == "running"
        raw = (await repository.runtime.scan("job"))[0]
        second = ReliableRuntime(service)
        service.runtime = second
        result = await second.execute(RunJob.model_validate(raw))
        assert result.status == "awaiting_doctor_review"
        assert result.consultation_id == running.consultation_id
        assert len(llm.calls) == 3
        await second.execute(RunJob.model_validate(raw))
        assert len(llm.calls) == 3


async def test_checkpoint_result_recovers_after_business_publish_failure(tmp_path: Path) -> None:
    """跨库发布故障后重试不重复模型调用，并最终发布权威快照。"""
    repository = InMemoryConsultationRepository()
    llm = FullLLM()
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as saver:
        service = ConsultationService(repository, ClinicalV2Service(llm, saver, FakeKnowledge()))
        runtime = ReliableRuntime(service)
        service.runtime = runtime
        runtime.launch = lambda job: None
        running = await service.create("发热")
        raw = (await repository.runtime.scan("job"))[0]
        original = repository.save

        async def fail(*args, **kwargs):
            raise RuntimeError("publish interrupted")

        repository.save = fail
        import pytest

        with pytest.raises(RuntimeError):
            await runtime.execute(RunJob.model_validate(raw))
        repository.save = original
        result = await runtime.execute(RunJob.model_validate(raw))
        assert result.status == "awaiting_doctor_review"
        assert result.consultation_id == running.consultation_id
        assert len(llm.calls) == 3
