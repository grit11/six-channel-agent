"""V2 SSE 读取服务：公开事件有严格会诊游标，临床状态始终另读业务快照。"""

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from backend.app.persistence.runtime import RuntimeStore
from backend.app.schemas.events import SSEEvent


class StreamService:
    """分页和持续事件读取共用同一个持久 RuntimeStore。"""

    def __init__(self, store: RuntimeStore) -> None:
        """绑定应用级存储。"""
        self.store = store

    async def page(self, consultation_id: UUID, after: int = 0) -> list[SSEEvent]:
        """按顺序读取公开事件页。"""
        return await self.store.events(consultation_id, after)

    async def stream(
        self,
        consultation_id: UUID,
        after: int = 0,
        *,
        heartbeat_seconds: float = 15,
    ) -> AsyncIterator[str]:
        """支持 Last-Event-ID 重连；不把进度事件当诊断结果。"""
        idle = 0.0
        while True:
            events = await self.page(consultation_id, after)
            for event in events:
                yield (
                    f"id: {event.event_id}\nevent: {event.event}\n"
                    f"data: {event.model_dump_json()}\n\n"
                )
                after = event.sequence
            if events:
                idle = 0
                # 人工暂停前的事件均已落库，浏览器收到后读取业务快照并重新订阅新轮。
                if events[-1].event in {
                    "clarification_requested",
                    "review_requested",
                    "consultation_completed",
                    "failed",
                }:
                    return
            else:
                # 重连游标可能已经位于本轮暂停/终态的最后一条，仍应结束流。
                # snapshot_updated 在业务事务内落库，故可覆盖后续通知尚未发布的崩溃窗口。
                head = await self.store.read("head", str(consultation_id))
                if head and int(str(head.get("sequence", 0))) <= after:
                    last = await self.store.events(consultation_id, max(0, after - 1), 1)
                    if last and (
                        last[-1].event
                        in {
                            "clarification_requested",
                            "review_requested",
                            "consultation_completed",
                            "failed",
                        }
                        or (
                            last[-1].event == "snapshot_updated"
                            and last[-1].payload.get("status") != "running"
                        )
                    ):
                        return
                idle += 0.25
                if idle >= heartbeat_seconds:
                    yield ": heartbeat\n\n"
                    idle = 0
                await asyncio.sleep(0.25)
