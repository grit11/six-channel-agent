"""V2 知识聊天 API。

职责：暴露 QA-only 生成式问答入口，不访问 consultation_service、不接受临床 thread。
边界：模型装配和医学检索分别由依赖工厂与 KnowledgeQAService 负责。
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.app.agents.knowledge_qa_tools import QAError
from backend.app.core.exceptions import KnowledgeNotFoundError, KnowledgeRetrievalError
from backend.app.schemas.knowledge import KnowledgeChatRequest, KnowledgeChatResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge-v2"])


@router.post("/chat", response_model=KnowledgeChatResponse)
async def knowledge_chat(
    request: Request, payload: KnowledgeChatRequest
) -> KnowledgeChatResponse | JSONResponse:
    """执行 QA-only 生成式回答，模型未装配时明确返回不可用。"""
    service = request.app.state.container.knowledge_qa_service
    try:
        return await service.chat(payload)
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
