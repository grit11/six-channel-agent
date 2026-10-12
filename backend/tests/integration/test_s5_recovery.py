"""S5 真实图恢复：SQLite 重开、结果发布与人工补偿，不使用在线模型。

业务数据显式复用 memory 替身，仅核验图和协调逻辑；PostgreSQL 实机验收后置。
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.runnables import RunnableConfig

from backend.app.core.config import Settings
from backend.app.core.exceptions import PersistenceError
from backend.app.graph.graph_factory import build_consultation_workflow
from backend.app.graph.main_graph import HumanGraphCommand
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.checkpoint import build_checkpoint_runtime
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.consultation import ConsultationCreateRequest
from backend.app.schemas.review import DoctorReviewRequest
from backend.app.schemas.runtime import GraphAction, RunJob
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.reasoning_service import ReasoningService
from backend.app.services.review_service import ReviewService


@pytest.mark.asyncio
async def test_checkpoint_result_and_review_outbox_survive_reopen(tmp_path: Path) -> None:
    """真实计算已落图但发布丢失，重开 saver 恢复相同轮次；审核不重复采用。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "recovery.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    repository = InMemoryConsultationRepository()
    runtime = await build_checkpoint_runtime(settings)

    def build(saver: object) -> ConsultationService:
        """重建应用对象，仍由固定知识边界构造主图。"""
        return ConsultationService(
            repository,
            ReasoningService(knowledge),
            protocol_version="1.0",
            knowledge_base_version="1.2",
            scoring_config_version="1.0",
            workflow=build_consultation_workflow(
                knowledge, saver, channel_weight=0.6, max_supplement_rounds=3
            ),
        )

    first = build(runtime.checkpointer)
    first._save_result = AsyncMock(side_effect=PersistenceError("发布故障"))
    try:
        with pytest.raises(PersistenceError):
            await first.create(ConsultationCreateRequest(text="发热，无汗，恶寒，身痛，喘，脉浮"))
        job = (
            RunJob.model_validate((await repository.runtime.scan("job"))[0])
            if await repository.runtime.scan("job")
            else RunJob.model_validate(
                next(
                    value
                    for (kind, _), value in repository.runtime.records.items()
                    if kind == "job"
                )
            )
        )
        config: RunnableConfig = {"configurable": {"thread_id": job.snapshot.thread_id}}
        state = await first.workflow.graph.aget_state(config)
        assert state.values["result_turn"] == job.turn and state.next == ("doctor_review",)
        # 模拟 reason 已提交、下一 doctor_wait 尚未首次执行的崩溃边界。
        await first.workflow.graph.aupdate_state(
            config,
            {
                "result": state.values["result"],
                "run_turn": job.turn,
                "result_turn": job.turn,
                "phase": state.values["phase"],
                "route": "wait",
            },
            as_node="reason",
        )
        state = await first.workflow.graph.aget_state(config)
        assert state.next == ("doctor_review",) and not any(task.interrupts for task in state.tasks)
        # 模拟崩溃发生在 checkpoint 计算完成后、任务结果记录提交之前。
        job.result = None
        job.lease_until = None
        job.owner = None
        await repository.runtime.write("job", str(job.run_id), job.model_dump(mode="json"))
    finally:
        await runtime.close()
    reopened = await build_checkpoint_runtime(settings)
    second = build(reopened.checkpointer)
    second.compute = AsyncMock(side_effect=AssertionError("不应重新执行医学节点"))
    try:
        saved = await second.runtime.execute(job)
        assert saved.reasoning and saved.reasoning.reasoning_paths
        second.compute.assert_not_awaited()
        events = await repository.runtime.events(saved.consultation_id, 0, limit=1000)
        assert any(item.event == "recovery_required" for item in events)
        # 已保存的医生采用成功，checkpoint 暂时失败，outbox 可由另一应用对象补偿。
        reconcile = second.workflow.reconcile
        second.workflow.reconcile = AsyncMock(side_effect=PersistenceError("checkpoint 不可用"))
        final = await ReviewService(second).review(
            saved.consultation_id,
            DoctorReviewRequest(
                action="accept",
                expected_turn=saved.turn,
                selected_path_id=saved.reasoning.reasoning_paths[0].path_id,
            ),
        )
        assert final.status == "completed" and len(final.review_history) == 1
        action = GraphAction.model_validate(
            (await repository.runtime.scan("action", final.consultation_id))[0]
        )
        second.workflow.reconcile = reconcile
        # 模拟命令完成后、completed 标记提交前故障，再补偿不会第二次 resume。
        await reconcile(
            action.snapshot, HumanGraphCommand.model_validate(action.command), str(action.action_id)
        )
        before = await second.workflow.graph.aget_state(config)
        await second.runtime.compensate(action)
        after = await second.workflow.graph.aget_state(config)
        assert after.values == before.values
        assert after.next == () and after.values["phase"] == "completed"
        assert len((await repository.get(final.consultation_id)).review_history) == 1
        assert await repository.runtime.scan("action", final.consultation_id) == []
    finally:
        await second.close()
        await reopened.close()


@pytest.mark.asyncio
async def test_business_failure_does_not_mutate_doctor_checkpoint(tmp_path: Path) -> None:
    """业务写入先失败时 checkpoint 不提前采用医生结论，下一次合法审核仍可执行。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "business-first.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    repository = InMemoryConsultationRepository()
    runtime = await build_checkpoint_runtime(settings)
    workflow = build_consultation_workflow(
        knowledge, runtime.checkpointer, channel_weight=0.6, max_supplement_rounds=3
    )
    service = ConsultationService(
        repository,
        ReasoningService(knowledge),
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
        workflow=workflow,
    )
    try:
        snapshot = await service.create(ConsultationCreateRequest(text="发热，恶寒，脉浮"))
        config: RunnableConfig = {"configurable": {"thread_id": snapshot.thread_id}}
        before = await workflow.graph.aget_state(config)
        save = repository.save
        repository.save = AsyncMock(side_effect=PersistenceError("业务事务回滚"))
        with pytest.raises(PersistenceError):
            await ReviewService(service).review(
                snapshot.consultation_id,
                DoctorReviewRequest(action="note", note="测试", expected_turn=snapshot.turn),
            )
        after = await workflow.graph.aget_state(config)
        assert after.values == before.values and after.next == before.next
        assert await repository.runtime.scan("action", snapshot.consultation_id) == []
        repository.save = save
    finally:
        await service.close()
        await runtime.close()
