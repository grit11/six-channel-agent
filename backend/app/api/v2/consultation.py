"""V2 会诊 API：身份、业务快照与人工恢复都委托应用服务。"""

from uuid import UUID

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.app.clinical_v2.schemas import HumanGraphCommandV2
from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.schemas.consultation import (
    ConsultationHistoryPage,
    ConsultationHistoryQuery,
    ConsultationSnapshot,
)

router = APIRouter(prefix="/consultations", tags=["consultations-v2"])


class ConsultationCreateV2(BaseModel):
    """创建原文和可选 UUID；幂等键通过请求头传递。"""

    model_config = ConfigDict(extra="forbid")
    clinical_text: str = Field(min_length=1, max_length=20000)
    consultation_id: UUID | None = None


@router.post("", response_model=ConsultationSnapshot)
async def create(
    request: Request,
    payload: ConsultationCreateV2,
    idempotency_key: str | None = Header(default=None, max_length=128),
) -> ConsultationSnapshot:
    """事务提交会诊运行意图；异步模式立即返回 running。"""
    return await request.app.state.container.consultation_service.create(
        payload.clinical_text,
        payload.consultation_id,
        idempotency_key=idempotency_key,
    )


@router.get("", response_model=ConsultationHistoryPage)
async def history(request: Request, page: int = 1, page_size: int = 10) -> ConsultationHistoryPage:
    """从业务 Repository 获取分页历史。"""
    return await request.app.state.container.consultation_service.list_history(
        ConsultationHistoryQuery(page=page, page_size=page_size),
    )


@router.get("/{consultation_id}", response_model=ConsultationSnapshot)
async def snapshot(request: Request, consultation_id: UUID) -> ConsultationSnapshot:
    """权威快照来自 Repository，图恢复点不替代业务存储。"""
    return await request.app.state.container.consultation_service.get_snapshot(consultation_id)


@router.post("/{consultation_id}/resume", response_model=ConsultationSnapshot)
async def resume(
    request: Request,
    consultation_id: UUID,
    command: HumanGraphCommandV2,
) -> ConsultationSnapshot:
    """提交版本化人工动作，幂等键由 action_id 提供。"""
    return await request.app.state.container.consultation_service.resume(consultation_id, command)


@router.post("/{consultation_id}/clarify", response_model=ConsultationSnapshot)
async def clarify(
    request: Request,
    consultation_id: UUID,
    command: HumanGraphCommandV2,
) -> ConsultationSnapshot:
    """仅接受 S2 澄清动作，不允许从此入口跳过医生审核。"""
    if command.kind != "clarify":
        raise InvalidBusinessRequestError("澄清入口仅允许 clarify")
    return await request.app.state.container.consultation_service.resume(consultation_id, command)
