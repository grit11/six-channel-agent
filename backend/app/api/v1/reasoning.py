"""六经候选 API：调用 ReasoningService 返回八纲与证据引用，不在 Router 编写医学逻辑。"""

from fastapi import APIRouter, Request

from backend.app.schemas.reasoning_api import (
    EvidenceReasoningRequest,
    ReasoningRequest,
    ReasoningResponse,
    RetrospectiveReplayRequest,
    RetrospectiveReplayResponse,
)

router = APIRouter(prefix="/reasoning", tags=["reasoning"])


@router.post("/replay", response_model=RetrospectiveReplayResponse)
async def replay(
    request: Request, payload: RetrospectiveReplayRequest
) -> RetrospectiveReplayResponse:
    """逐步回放医生提供的信息，不修改正式会诊或注入原病例结论。"""
    return await request.app.state.container.reasoning_service.replay(payload)


@router.post("/channel-candidates", response_model=ReasoningResponse)
async def channel_candidates(request: Request, payload: ReasoningRequest) -> ReasoningResponse:
    """生成六经宽候选及其八纲倾向，排序分不是临床概率。"""
    return await request.app.state.container.reasoning_service.analyze_text(
        payload.text, source=payload.source, top_k=payload.top_k
    )


@router.post("/evidence", response_model=ReasoningResponse)
async def analyze_evidence(
    request: Request, payload: EvidenceReasoningRequest
) -> ReasoningResponse:
    """使用医生编辑后的三值事实和时间线运行同一子图。"""
    return await request.app.state.container.reasoning_service.analyze_evidence(
        payload.clinical_evidence, top_k=payload.top_k
    )
