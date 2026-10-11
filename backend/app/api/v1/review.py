"""医生审核 HTTP 边界：只校验请求并委托 ReviewService，保存人工采用与原建议差异。"""

from uuid import UUID

from fastapi import APIRouter, Request

from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.schemas.review import DoctorReviewRequest

router = APIRouter(prefix="/consultations", tags=["review"])


@router.post("/{consultation_id}/review", response_model=ConsultationSnapshot)
async def review(
    request: Request, consultation_id: UUID, payload: DoctorReviewRequest
) -> ConsultationSnapshot:
    """校验路径身份、轮次和状态，审核与业务持久化由服务处理。"""
    container = request.app.state.container
    return await container.consultation_service.runtime.idempotent(
        request.headers.get("Idempotency-Key"),
        f"{consultation_id}:review",
        payload.model_dump(mode="json"),
        lambda: container.review_service.review(consultation_id, payload),
    )
