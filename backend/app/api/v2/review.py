"""V2 人工审核 API：薄边界，所有状态和版本核验交给应用服务。"""

from uuid import UUID

from fastapi import APIRouter, Request

from backend.app.clinical_v2.schemas import HumanGraphCommandV2
from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.schemas.consultation import ConsultationSnapshot

router = APIRouter(prefix="/consultations", tags=["reviews-v2"])


@router.post("/{consultation_id}/review", response_model=ConsultationSnapshot)
async def review(
    request: Request,
    consultation_id: UUID,
    command: HumanGraphCommandV2,
) -> ConsultationSnapshot:
    """提交医生审核动作，不允许模型自动完成会诊。"""
    if command.kind != "review":
        raise InvalidBusinessRequestError("审核入口仅允许 review")
    return await request.app.state.container.consultation_service.resume(consultation_id, command)
