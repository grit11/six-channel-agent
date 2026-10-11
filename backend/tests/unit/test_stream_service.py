"""持久 SSE 回归：真实序列续传、跨会诊拒绝、分页、心跳与安全故障。"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from backend.app.core.exceptions import InvalidBusinessRequestError, PersistenceError
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.stream_service import StreamService


def snapshot() -> ConsultationSnapshot:
    """无医学结论的快照，原文不得进入事件。"""
    now = datetime.now(UTC)
    return ConsultationSnapshot(
        consultation_id=uuid4(),
        thread_id=str(uuid4()),
        status="running",
        turn=2,
        raw_input="不应进入事件的原文",
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
        created_at=now,
        updated_at=now,
    )


async def setup_stream() -> tuple[StreamService, ConsultationSnapshot]:
    """使用共享真实 memory 存储，避免快照派生假事件。"""
    repository = InMemoryConsultationRepository()
    initial = snapshot()
    await repository.create(initial)
    service = ConsultationService(
        repository,
        AsyncMock(),
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
    )
    return StreamService(service, poll_seconds=0.001, heartbeat_seconds=0), initial


@pytest.mark.asyncio
async def test_live_event_before_completion_and_heartbeat() -> None:
    """运行中的节点事件能立即订阅，未完成轮次无需等待结果回放。"""
    service, initial = await setup_stream()
    await service.store.emit(
        initial.consultation_id, initial.turn, "stage_started", {"stage": "eight_principles"}
    )
    generator = service.events(initial, AsyncMock(return_value=False))
    assert "snapshot_updated" in await anext(generator)
    stage = await anext(generator)
    assert "stage_started" in stage and "eight_principles" in stage
    heartbeat = await anext(generator)
    assert "heartbeat" in heartbeat and "id:" not in heartbeat
    assert all(initial.raw_input not in frame for frame in [stage, heartbeat])
    await generator.aclose()


@pytest.mark.asyncio
async def test_reconnect_replays_missing_terminal_and_rejects_other_cursor() -> None:
    """重连严格跳过已提交序列；终态事件排空，其他会诊/超前游标拒绝。"""
    service, initial = await setup_stream()
    page = await service.page(initial.consultation_id, None)
    cursor = page.events[-1].event_id
    final = await service.consultations.repository.save(
        initial.model_copy(update={"status": "completed", "turn": 3}), expected_turn=2
    )
    frames = [frame async for frame in service.events(final, AsyncMock(return_value=False), cursor)]
    assert len(frames) == 2 and "event: completed" in frames[-1]
    with pytest.raises(InvalidBusinessRequestError):
        await service.validate_cursor(initial.consultation_id, f"{uuid4()}:1")
    with pytest.raises(InvalidBusinessRequestError):
        await service.validate_cursor(initial.consultation_id, f"{initial.consultation_id}:999")
    assert [frame async for frame in service.events(final, AsyncMock(return_value=True))] == []


@pytest.mark.asyncio
async def test_sequence_pagination_and_failure_without_internal_message() -> None:
    """超过一页按游标精确续传，响应开始后的基础设施故障不输出内部信息。"""
    service, initial = await setup_stream()
    for _ in range(120):
        await service.store.emit(
            initial.consultation_id, initial.turn, "model_started", {"capability": "chat"}
        )
    first = await service.page(initial.consultation_id, None)
    second = await service.page(initial.consultation_id, first.events[-1].event_id)
    assert len(first.events) == 100 and len(second.events) == 21
    assert [item.sequence for item in first.events + second.events] == list(range(1, 122))
    service.store.events = AsyncMock(side_effect=PersistenceError("内部敏感连接"))
    frames = [frame async for frame in service.events(initial, AsyncMock(return_value=False))]
    assert "PERSISTENCE_ERROR" in frames[-1] and "内部敏感" not in frames[-1]


@pytest.mark.asyncio
async def test_terminal_transition_drains_more_than_one_page() -> None:
    """终态查询前并发产生超过一页事件时，不能只排空第一页就关闭 SSE。"""
    service, initial = await setup_stream()
    original_get = service.consultations.get_snapshot

    async def finish(identifier: UUID) -> ConsultationSnapshot:
        """在首次页查询后提交多页事件和终态，模拟运行提交与 SSE 读取竞争。"""
        if (await original_get(identifier)).status == "running":
            for _ in range(250):
                await service.store.emit(
                    identifier, initial.turn, "stage_completed", {"stage": "topk"}
                )
            await service.consultations.repository.save(
                initial.model_copy(update={"status": "completed", "turn": 3}), expected_turn=2
            )
        return await original_get(identifier)

    service.consultations.get_snapshot = finish
    frames = [frame async for frame in service.events(initial, AsyncMock(return_value=False))]
    assert len(frames) == 253 and "event: completed" in frames[-1]
