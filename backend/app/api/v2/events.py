"""V2 SSE API：先核验会诊和游标，再订阅公开事件；不输出患者原文。"""

from uuid import UUID

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from backend.app.persistence.runtime import cursor_sequence

router = APIRouter(prefix="/consultations", tags=["events-v2"])


@router.get("/{consultation_id}/events")
async def events(
    request: Request,
    consultation_id: UUID,
    after: str | None = Query(default=None),
    last_event_id: str | None = Header(default=None),
) -> StreamingResponse:
    """会诊不存在或游标跨会诊时在发送 SSE 响应前明确拒绝。"""
    container = request.app.state.container
    await container.consultation_service.get_snapshot(consultation_id)
    sequence = cursor_sequence(consultation_id, last_event_id or after)
    return StreamingResponse(
        container.stream_service.stream(
            consultation_id,
            sequence,
            heartbeat_seconds=container.settings.sse_heartbeat_seconds,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
