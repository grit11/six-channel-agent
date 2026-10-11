"""可靠运行故障回归：共享存储多实例、崩溃接管、取消、发布补偿和幂等。

测试替身不访问真实模型/数据库；共享 memory 仅模拟已保留业务数据的重建实例，
不把 memory 的模拟接管宣传成进程重启持久化。
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.app.core.exceptions import InvalidStateTransitionError, PersistenceError
from backend.app.observability.progress import emit_progress
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.consultation import ConsultationCreateRequest, ConsultationSnapshot
from backend.app.schemas.reasoning import EightPrinciplesState, PrincipleAxis
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.schemas.runtime import RunJob
from backend.app.services.consultation_service import ConsultationService


def service(repository: InMemoryConsultationRepository, **options: object) -> ConsultationService:
    """注入无医学判断的应用替身，保持正式任务/租约实现。"""
    return ConsultationService(
        repository,
        AsyncMock(),
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
        **options,
    )


def result() -> ReasoningResponse:
    """构造合法空候选公开结果，不伪造任何来源或诊断。"""
    axis = PrincipleAxis(lean="uncertain", score_left=0, score_right=0)
    principles = EightPrinciplesState(
        yin_yang=axis,
        exterior_interior=axis,
        cold_heat=axis,
        deficiency_excess=axis,
        summary="无事实",
    )
    return ReasoningResponse(
        thread_id="test",
        clinical_evidence=ClinicalEvidence(),
        eight_principles=principles,
        candidates=[],
        reasoning_paths=[],
        evidence_gaps=[],
        completed_stages=[],
    )


async def pending(repository: InMemoryConsultationRepository) -> RunJob:
    """直接提交精确任务模拟进程在接收请求后、执行前崩溃。"""
    now = datetime.now(UTC)
    snapshot = ConsultationSnapshot(
        consultation_id=uuid4(),
        thread_id=str(uuid4()),
        status="running",
        turn=2,
        raw_input="非患者故障用例",
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
        created_at=now,
        updated_at=now,
    )
    job = RunJob(
        run_id=uuid4(),
        consultation_id=snapshot.consultation_id,
        turn=2,
        snapshot=snapshot,
        evidence=ClinicalEvidence(),
    )
    await repository.create(
        snapshot,
        runtime_records=[
            ("job", str(job.run_id), job.model_dump(mode="json")),
            ("active", str(job.consultation_id), {"run_id": str(job.run_id)}),
        ],
    )
    return job


@pytest.mark.asyncio
async def test_two_instances_execute_pending_only_once() -> None:
    """两个实例竞争同图只计算一次，业务完成与任务终态一起保存。"""
    repository = InMemoryConsultationRepository()
    job = await pending(repository)
    first, second = service(repository), service(repository)
    compute = AsyncMock(return_value=result())
    first.compute = second.compute = compute
    saved = await asyncio.gather(first.runtime.execute(job), second.runtime.execute(job))
    assert compute.await_count == 1
    assert all(item.status == "awaiting_doctor_review" for item in saved)
    raw = await repository.runtime.read("job", str(job.run_id))
    assert raw["status"] == "succeeded"


@pytest.mark.asyncio
async def test_live_lease_not_stolen_then_expired_lease_recovered() -> None:
    """活租约不能接管，过期后显式恢复并受最大次数限制。"""
    repository = InMemoryConsultationRepository()
    job = await pending(repository)
    job.owner = "crashed-owner"
    job.lease_until = datetime.now(UTC) + timedelta(seconds=30)
    job.status = "running"
    await repository.runtime.write("job", str(job.run_id), job.model_dump(mode="json"))
    current = service(repository)
    current.compute = AsyncMock(return_value=result())
    assert (await current.runtime.execute(job)).status == "running"
    current.compute.assert_not_awaited()
    job.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    await repository.runtime.write("job", str(job.run_id), job.model_dump(mode="json"))
    assert (await current.runtime.execute(job)).status == "awaiting_doctor_review"
    current.compute.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_failure_reuses_persisted_result() -> None:
    """计算结束后的业务提交故障可恢复发布，不重复模型/医学节点。"""
    repository = InMemoryConsultationRepository()
    job = await pending(repository)
    first = service(repository, lease_seconds=0.03)
    first.compute = AsyncMock(return_value=result())
    save = repository.save
    repository.save = AsyncMock(side_effect=PersistenceError("业务提交故障"))
    with pytest.raises(PersistenceError):
        await first.runtime.execute(job)
    persisted = RunJob.model_validate(await repository.runtime.read("job", str(job.run_id)))
    assert persisted.result is not None
    assert (await repository.get(job.consultation_id)).status == "running"
    persisted.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    await repository.runtime.write("job", str(job.run_id), persisted.model_dump(mode="json"))
    repository.save = save
    restarted = service(repository)
    restarted.compute = AsyncMock(side_effect=AssertionError("不能重复计算"))
    assert (await restarted.runtime.execute(persisted)).status == "awaiting_doctor_review"
    restarted.compute.assert_not_awaited()


@pytest.mark.asyncio
async def test_cross_instance_cancel_stops_computation_and_fences_result() -> None:
    """第二实例取消持久任务，第一实例续约检测并停止，旧结果不能复活会诊。"""
    repository = InMemoryConsultationRepository()
    job = await pending(repository)
    first, second = service(repository, lease_seconds=0.03), service(repository)
    entered = asyncio.Event()

    async def blocked(*args: object) -> ReasoningResponse:
        """模拟尚未完成的模型调用，并报告真实执行中事件。"""
        await emit_progress("stage_started", {"stage": "eight_principles"})
        entered.set()
        await asyncio.Event().wait()
        return result()

    first.compute = blocked
    task = first.runtime.launch(job)
    await entered.wait()
    saved = await asyncio.wait_for(second.cancel(job.consultation_id), 1)
    assert saved.status == "cancelled"
    await asyncio.gather(task, return_exceptions=True)
    assert (await repository.get(job.consultation_id)).status == "cancelled"
    raw = await repository.runtime.read("job", str(job.run_id))
    assert raw["status"] == "cancelled"
    events = await repository.runtime.events(job.consultation_id, 0)
    assert any(item.event == "stage_started" for item in events)


@pytest.mark.asyncio
async def test_timeout_and_exhausted_crash_attempts_leave_failed() -> None:
    """超时或接管次数耗尽都离开 running，不形成无限执行循环。"""
    repository = InMemoryConsultationRepository()
    job = await pending(repository)
    current = service(repository, timeout_seconds=0.01)
    current.compute = AsyncMock(side_effect=lambda *args: None)

    async def blocked(*args: object) -> ReasoningResponse:
        """保持未完成模拟硬超时。"""
        await asyncio.Event().wait()
        return result()

    current.compute = blocked
    with pytest.raises(TimeoutError):
        await current.runtime.execute(job)
    assert (await repository.get(job.consultation_id)).status == "failed"
    another = await pending(repository)
    another.attempts = 3
    await repository.runtime.write("job", str(another.run_id), another.model_dump(mode="json"))
    assert (await service(repository).runtime.execute(another)).status == "failed"


@pytest.mark.asyncio
async def test_concurrent_idempotency_collision_and_lost_response() -> None:
    """重复创建和响应丢失只创建一份业务；键与不同输入冲突明确拒绝。"""
    repository = InMemoryConsultationRepository()
    current = service(repository, asynchronous=True)
    request = ConsultationCreateRequest(text="测试")

    async def create() -> ConsultationSnapshot:
        """提交后主动抛错模拟响应发送之前进程退出。"""
        await current.create(request)
        raise PersistenceError("响应丢失")

    with pytest.raises(PersistenceError):
        await current.runtime.idempotent(
            "test-key-0001", "create", request.model_dump(mode="json"), create
        )
    current.runtime.tasks[next(iter(current.runtime.tasks))].cancel()
    await asyncio.gather(*current.runtime.tasks.values(), return_exceptions=True)
    results = await asyncio.gather(
        *[
            current.runtime.idempotent(
                "test-key-0001",
                "create",
                request.model_dump(mode="json"),
                AsyncMock(side_effect=AssertionError("不得再创建")),
            )
            for _ in range(5)
        ]
    )
    assert len({item.consultation_id for item in results}) == 1
    assert len(repository._items) == 1
    with pytest.raises(InvalidStateTransitionError):
        await current.runtime.idempotent(
            "test-key-0001", "create", {"different": True}, AsyncMock()
        )


@pytest.mark.asyncio
async def test_shutdown_releases_lease_for_recovery() -> None:
    """优雅退出保存待接管状态，后续实例继续任务而不是丢失请求。"""
    repository = InMemoryConsultationRepository()
    job = await pending(repository)
    first = service(repository)
    entered = asyncio.Event()

    async def blocked(*args: object) -> ReasoningResponse:
        """停在计算中模拟进程退出。"""
        entered.set()
        await asyncio.Event().wait()
        return result()

    first.compute = blocked
    first.runtime.launch(job)
    await entered.wait()
    await first.close()
    saved = RunJob.model_validate(await repository.runtime.read("job", str(job.run_id)))
    assert saved.lease_until is None and saved.owner is None
    restarted = service(repository)
    restarted.compute = AsyncMock(return_value=result())
    assert (await restarted.runtime.execute(saved)).status == "awaiting_doctor_review"
