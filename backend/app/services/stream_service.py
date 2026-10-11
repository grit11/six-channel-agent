"""持久 SSE 订阅：读取运行事件序列，不把事后阶段回放伪装为实时进度。

Repository 是业务快照权威；事件仅包含公开阶段/计数/耗时与状态。
游标严格归属会诊，心跳不携带持久事件 ID，不会破坏浏览器自动重连位置。
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID

from backend.app.core.exceptions import ApplicationError, InvalidBusinessRequestError
from backend.app.persistence.runtime import cursor_sequence
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.schemas.events import SSEEvent
from backend.app.schemas.runtime import EventPage
from backend.app.services.consultation_service import ConsultationService


class StreamService:
    """按持久序列读取实际事件，业务终态后排空事件页再结束连接。"""

    def __init__(
        self,
        consultations: ConsultationService,
        *,
        poll_seconds: float = 0.25,
        heartbeat_seconds: float = 15,
    ) -> None:
        """复用运行存储，不创建第二个事件历史或模型连接。"""
        self.consultations = consultations
        self.store = consultations.repository.runtime
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds

    async def get_snapshot(self, consultation_id: UUID) -> ConsultationSnapshot:
        """响应头提交前核验资源存在。"""
        return await self.consultations.get_snapshot(consultation_id)

    async def validate_cursor(self, identifier: UUID, cursor: str | None) -> int:
        """校验归属与服务端已提交上限，超前游标明确失败。"""
        sequence = cursor_sequence(identifier, cursor)
        head = await self.store.read("head", str(identifier))
        if sequence > int(str((head or {}).get("sequence", 0))):
            raise InvalidBusinessRequestError("事件游标超出当前会诊序列")
        return sequence

    async def page(self, identifier: UUID, cursor: str | None) -> EventPage:
        """刷新后读取一页公开进度，不回传任务内部输入或结果。"""
        await self.get_snapshot(identifier)
        after = await self.validate_cursor(identifier, cursor)
        return EventPage(events=await self.store.events(identifier, after))

    @staticmethod
    def frame(event: SSEEvent) -> str:
        """JSON 转义防止事件注入，只有已持久化事件发送浏览器游标。"""
        return f"id: {event.event_id}\nevent: {event.event}\ndata: {event.model_dump_json()}\n\n"

    async def events(
        self,
        snapshot: ConsultationSnapshot,
        disconnected: Callable[[], Awaitable[bool]],
        last_event_id: str | None = None,
    ) -> AsyncIterator[str]:
        """有序续传、心跳和安全故障提示；断线立即停止读库。"""
        after = await self.validate_cursor(snapshot.consultation_id, last_event_id)
        heartbeat_at = time.monotonic()
        while not await disconnected():
            try:
                batch = await self.store.events(snapshot.consultation_id, after)
                for item in batch:
                    yield self.frame(item)
                    after = item.sequence
                if len(batch) == 100:
                    continue
                snapshot = await self.get_snapshot(snapshot.consultation_id)
                if snapshot.status in {"completed", "cancelled", "failed"}:
                    # 查询之后仍可能有并发终态事件提交，再排空一次而不遗漏。
                    terminal_batch = await self.store.events(snapshot.consultation_id, after)
                    for item in terminal_batch:
                        yield self.frame(item)
                        after = item.sequence
                    if len(terminal_batch) == 100:
                        continue
                    return
                if time.monotonic() - heartbeat_at >= self.heartbeat_seconds:
                    now = datetime.now(UTC).isoformat()
                    yield f'event: heartbeat\ndata: {{"server_time": "{now}"}}\n\n'
                    heartbeat_at = time.monotonic()
            except ApplicationError as exc:
                yield f'event: failed\ndata: {{"code": "{exc.code}"}}\n\n'
                return
            await asyncio.sleep(self.poll_seconds)
