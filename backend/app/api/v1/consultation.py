"""会话 HTTP 边界：校验 DTO 并委托应用服务，不调用图或实现医学规则。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response

from backend.app.schemas.consultation import (
    ConsultationCreateRequest,
    ConsultationEvidenceRequest,
    ConsultationHistoryPage,
    ConsultationHistoryQuery,
    ConsultationReplay,
    ConsultationRerunRequest,
    ConsultationSnapshot,
)

router = APIRouter(prefix="/consultations", tags=["consultations"])


@router.get("", response_model=ConsultationHistoryPage)
async def history(
    request: Request, query: Annotated[ConsultationHistoryQuery, Query()]
) -> ConsultationHistoryPage:
    """查询业务历史摘要分页，不访问 checkpoint 或患者全文。"""
    return await request.app.state.container.consultation_service.list_history(query)


@router.get("/{consultation_id}/replay", response_model=ConsultationReplay)
async def replay(request: Request, consultation_id: UUID) -> ConsultationReplay:
    """返回当前业务快照公开阶段回放，不触发重算或写操作。"""
    return await request.app.state.container.consultation_service.get_replay(consultation_id)


@router.post("", response_model=ConsultationSnapshot, status_code=201)
async def create(request: Request, payload: ConsultationCreateRequest) -> ConsultationSnapshot:
    """创建业务会话，异步模式返回 running，SSE 订阅后续状态变化。"""
    service = request.app.state.container.consultation_service
    return await service.runtime.idempotent(
        request.headers.get("Idempotency-Key"),
        "create",
        payload.model_dump(mode="json"),
        lambda: service.create(payload),
    )


@router.get("/{consultation_id}", response_model=ConsultationSnapshot)
async def snapshot(request: Request, consultation_id: UUID) -> ConsultationSnapshot:
    """返回业务快照，不用 checkpoint 代替业务数据库。"""
    return await request.app.state.container.consultation_service.get_snapshot(consultation_id)


@router.post("/{consultation_id}/evidence", response_model=ConsultationSnapshot)
async def add_evidence(
    request: Request,
    response: Response,
    consultation_id: UUID,
    payload: ConsultationEvidenceRequest,
) -> ConsultationSnapshot:
    """提交并合并补充信息，复用原 thread。"""
    service = request.app.state.container.consultation_service
    result = await service.runtime.idempotent(
        request.headers.get("Idempotency-Key"),
        f"{consultation_id}:evidence",
        payload.model_dump(mode="json"),
        lambda: service.add_evidence(consultation_id, payload),
    )
    if result.status == "running":
        response.status_code = 202
    return result


@router.post("/{consultation_id}/rerun", response_model=ConsultationSnapshot)
async def rerun(
    request: Request, response: Response, consultation_id: UUID, payload: ConsultationRerunRequest
) -> ConsultationSnapshot:
    """按当前业务快照重算，同 thread 的轮次检查由服务负责。"""
    service = request.app.state.container.consultation_service
    result = await service.runtime.idempotent(
        request.headers.get("Idempotency-Key"),
        f"{consultation_id}:rerun",
        payload.model_dump(mode="json"),
        lambda: service.rerun(consultation_id, payload),
    )
    if result.status == "running":
        response.status_code = 202
    return result


@router.post("/{consultation_id}/cancel", response_model=ConsultationSnapshot)
async def cancel(request: Request, consultation_id: UUID) -> ConsultationSnapshot:
    """取消未结束的业务会话。"""
    service = request.app.state.container.consultation_service
    return await service.runtime.idempotent(
        request.headers.get("Idempotency-Key"),
        f"{consultation_id}:cancel",
        {},
        lambda: service.cancel(consultation_id),
    )


@router.post("/{consultation_id}/review-ready", response_model=ConsultationSnapshot)
async def request_review(
    request: Request, consultation_id: UUID, payload: ConsultationRerunRequest
) -> ConsultationSnapshot:
    """医生选择按当前信息进入审核，停止继续要求补证。"""
    service = request.app.state.container.consultation_service
    return await service.runtime.idempotent(
        request.headers.get("Idempotency-Key"),
        f"{consultation_id}:review-ready",
        payload.model_dump(mode="json"),
        lambda: service.request_review(consultation_id, payload),
    )
