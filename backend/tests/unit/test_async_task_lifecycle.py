"""异步任务生命周期回归：运行中并发拒绝、取消与 shutdown 先落状态再关资源。"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.app.core.exceptions import InvalidStateTransitionError
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.consultation import ConsultationCreateRequest, ConsultationRerunRequest
from backend.app.services.consultation_service import ConsultationService


def service(reasoning: AsyncMock) -> ConsultationService:
    """注入悬挂模型替身，核验任务控制而非医学推理。"""
    return ConsultationService(
        InMemoryConsultationRepository(),
        reasoning,
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
        asynchronous=True,
    )


@pytest.mark.asyncio
async def test_cancel_running_task_and_reject_second_run() -> None:
    """取消先停止任务，不允许晚到结果覆盖 cancelled；运行中重跑及时拒绝。"""
    started = asyncio.Event()

    async def hanging(*args: object, **kwargs: object) -> None:
        """模拟模型长请求，直到取消才结束。"""
        started.set()
        await asyncio.Event().wait()

    reasoning = AsyncMock()
    reasoning.analyze_text.side_effect = hanging
    consultations = service(reasoning)
    initial = await consultations.create(ConsultationCreateRequest(text="测试"))
    await started.wait()
    with pytest.raises(InvalidStateTransitionError):
        await consultations.rerun(initial.consultation_id, ConsultationRerunRequest())
    cancelled = await consultations.cancel(initial.consultation_id)
    assert cancelled.status == "cancelled"
    assert not consultations._tasks
    assert (await consultations.get_snapshot(initial.consultation_id)).status == "cancelled"


@pytest.mark.asyncio
async def test_shutdown_releases_interrupted_task_lease() -> None:
    """退出先停计算与释放租约；已接收任务待下一实例恢复，不伪装为计算失败。"""
    started = asyncio.Event()

    async def hanging(*args: object, **kwargs: object) -> None:
        """暂停请求，确认 close 等待任务退出。"""
        started.set()
        await asyncio.Event().wait()

    reasoning = AsyncMock()
    reasoning.analyze_text.side_effect = hanging
    consultations = service(reasoning)
    initial = await consultations.create(ConsultationCreateRequest(text="测试"))
    await started.wait()
    await consultations.close()
    assert (await consultations.get_snapshot(initial.consultation_id)).status == "running"
    jobs = await consultations.repository.runtime.scan("job", initial.consultation_id)
    assert len(jobs) == 1 and jobs[0]["lease_until"] is None and jobs[0]["owner"] is None
    assert not consultations._tasks


@pytest.mark.asyncio
async def test_shutdown_before_task_first_schedule_leaves_recoverable_job() -> None:
    """首次调度前退出仍保留 pending 任务，下一实例可以认领。"""
    consultations = service(AsyncMock())
    initial = await consultations.create(ConsultationCreateRequest(text="测试"))
    await consultations.close()
    assert (await consultations.get_snapshot(initial.consultation_id)).status == "running"
    jobs = await consultations.repository.runtime.scan("job", initial.consultation_id)
    assert len(jobs) == 1 and jobs[0]["status"] == "pending"
