"""V2 知识聊天 API。

职责：暴露 QA-only 生成式问答入口，不访问 consultation_service、不接受临床 thread。
边界：模型装配和医学检索分别由依赖工厂与 KnowledgeQAService 负责。
"""

import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute

from backend.app.agents.knowledge_qa_tools import QAError
from backend.app.core.exceptions import KnowledgeNotFoundError, KnowledgeRetrievalError
from backend.app.schemas.knowledge import (
    KnowledgeChatRequest,
    KnowledgeChatResponse,
    QAMessage,
    QASession,
    QASessionCreate,
    QASessionPatch,
)


class QARoute(APIRoute):
    """仅 QA 路由转换其专属状态码，不修改全局临床错误映射。"""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        """把安全 QAError 转为统一 envelope，保留 401/404/409。"""
        original = super().get_route_handler()

        async def handle(request: Request) -> Response:
            """不记录完整问题或原始异常。"""
            try:
                return await original(request)
            except QAError as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"error": {"code": exc.code, "message": exc.message}},
                )

        return handle


router = APIRouter(prefix="/knowledge", tags=["knowledge-v2"], route_class=QARoute)


def qa_owner(request: Request) -> str:
    """只信任服务端认证中间件注入主体；研发使用显式合成测试身份。"""
    owner = getattr(request.state, "owner_id", None)
    if isinstance(owner, str) and owner.strip():
        return owner
    if request.app.state.container.settings.app_env == "prod":
        raise QAError("QA_AUTH_REQUIRED", "QA 历史需要可信身份", 401)
    return "dev:synthetic"


@router.post("/sessions", response_model=QASession)
async def create_session(request: Request, payload: QASessionCreate) -> QASession:
    """新建有归属的 QA 会话，无患者或临床线程关联。"""
    service = request.app.state.container.knowledge_qa_service
    return await service.repository.create(
        qa_owner(request), payload.title, service.knowledge.version
    )


@router.get("/sessions", response_model=list[QASession])
async def list_sessions(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str = Query("", max_length=160),
) -> list[QASession]:
    """带 owner 的分页历史搜索，不读取临床会诊列表。"""
    return await request.app.state.container.knowledge_qa_service.repository.list_sessions(
        qa_owner(request), page, page_size, q
    )


@router.get("/sessions/{session_id}", response_model=QASession)
async def get_session(request: Request, session_id: UUID) -> QASession:
    """读取本人会话元数据。"""
    return await request.app.state.container.knowledge_qa_service.repository.get(
        session_id, qa_owner(request)
    )


@router.get("/sessions/{session_id}/messages", response_model=list[QAMessage])
async def get_messages(
    request: Request,
    session_id: UUID,
    after: int = Query(0, ge=0),
    page_size: int = Query(100, ge=1, le=100),
) -> list[QAMessage]:
    """分页读取权威消息，刷新不依赖 localStorage 的答案。"""
    return await request.app.state.container.knowledge_qa_service.repository.history(
        session_id, qa_owner(request), after, page_size
    )


@router.patch("/sessions/{session_id}", response_model=QASession)
async def patch_session(request: Request, session_id: UUID, payload: QASessionPatch) -> QASession:
    """版本化重命名或归档，不更新临床数据。"""
    return await request.app.state.container.knowledge_qa_service.repository.patch(
        session_id, qa_owner(request), payload
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    request: Request, session_id: UUID, expected_revision: int = Query(..., ge=0)
) -> Response:
    """按版本删除本人 QA 消息。"""
    await request.app.state.container.knowledge_qa_service.repository.remove(
        session_id, qa_owner(request), expected_revision
    )
    return Response(status_code=204)


@router.get("/turns/{turn_id}", response_model=QAMessage)
async def get_turn(request: Request, turn_id: UUID) -> QAMessage:
    """中断后查询本人 QA 运行最终状态。"""
    return await request.app.state.container.knowledge_qa_service.repository.turn(
        turn_id, qa_owner(request)
    )


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: KnowledgeChatRequest) -> StreamingResponse:
    """POST fetch SSE；断线停止受监督任务，不在后台无人负责地继续生成。"""
    owner = qa_owner(request)
    service = request.app.state.container.knowledge_qa_service
    # 流建立前校验归属，防止以 SSE 200 掩盖跨用户访问失败。
    if payload.session_id:
        await service.repository.get(payload.session_id, owner)

    async def events() -> AsyncIterator[str]:
        """只输出服务归一的公开事件，不输出 Agent 内部事件或推理字段。"""
        async for name, data in service.stream(payload, owner):
            if name == "heartbeat":
                yield ": heartbeat\n\n"
            else:
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat", response_model=KnowledgeChatResponse)
async def knowledge_chat(
    request: Request, payload: KnowledgeChatRequest
) -> KnowledgeChatResponse | JSONResponse:
    """执行 QA-only 生成式回答，模型未装配时明确返回不可用。"""
    service = request.app.state.container.knowledge_qa_service
    try:
        return await service.chat(payload, qa_owner(request))
    except QAError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc)}},
        )
    except KnowledgeRetrievalError as exc:
        if str(exc).startswith("QA_CHAT_NOT_CONFIGURED"):
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "QA_CHAT_NOT_CONFIGURED", "message": str(exc)}},
            )
        raise
    except KnowledgeNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "QA_SESSION_EXPIRED",
                    "message": str(exc),
                }
            },
        )
