"""会话 SSE 协议边界：先验证资源再提交响应头，事件由应用服务产生。"""

from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from backend.app.schemas.runtime import EventPage

router = APIRouter(prefix="/consultations", tags=["stream"])


@router.get("/{consultation_id}/stream")
async def stream(request: Request, consultation_id: UUID) -> StreamingResponse:
    """订阅版本变化；最终完整数据始终通过快照查询获得。"""
    service = request.app.state.container.stream_service
    snapshot = await service.get_snapshot(consultation_id)
    cursor = request.headers.get("last-event-id") or request.query_params.get("cursor")
    await service.validate_cursor(consultation_id, cursor)

    return StreamingResponse(
        service.events(snapshot, request.is_disconnected, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{consultation_id}/events", response_model=EventPage)
async def events(
    request: Request,
    consultation_id: UUID,
    cursor: str | None = Query(default=None, max_length=128),
) -> EventPage:
    """读取有界事件页；客户端按最后序列继续读取，完整事实走快照接口。"""
    return await request.app.state.container.stream_service.page(consultation_id, cursor)
