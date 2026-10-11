"""健康与版本 API，仅输出安全的运行状态，不执行医学推理。"""

from fastapi import APIRouter, Request

from backend.app.nodes.clinical_parser import RULE_PARSER_VERSION
from backend.app.schemas.events import HealthResponse, VersionsResponse

router = APIRouter(tags=["system"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    """返回进程存活状态，不探测外部依赖。"""
    return {"status": "ok"}


@router.get("/health", response_model=HealthResponse)
@router.get("/health/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    """读取已初始化的后端状态；降级状态与未就绪必须区分。"""
    container = request.app.state.container
    degraded = (
        container.vector_store.status.degraded
        or container.checkpoint_runtime.status.degraded
        or container.database.backend != "postgres"
    )
    return HealthResponse(
        status="degraded" if degraded else "ok",
        vector_backend=container.vector_store.status.backend,
        checkpoint_backend=container.checkpoint_runtime.status.backend,
        business_database=container.database.backend,
    )


@router.get("/system/versions", response_model=VersionsResponse)
async def versions(request: Request) -> VersionsResponse:
    """返回协议、知识库、模型和工程版本，未知配置保持明确空值。"""
    settings = request.app.state.container.settings
    return VersionsResponse(
        protocol_version=settings.protocol_version,
        knowledge_base_version=settings.knowledge_base_version,
        scoring_config_version=settings.scoring_config_version,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        clinical_parser_mode=settings.clinical_parser_mode,
        reasoning_mode=settings.reasoning_mode,
        knowledge_qa_mode=settings.knowledge_qa_mode,
        backend_build_version="0.1.0",
        prompt_version=(
            "clinical-parser/v1.0+reasoning/v1.0"
            if settings.clinical_parser_mode == "llm" and settings.reasoning_mode == "llm"
            else (
                "clinical-parser/v1.0"
                if settings.clinical_parser_mode == "llm"
                else "reasoning/v1.0" if settings.reasoning_mode == "llm" else RULE_PARSER_VERSION
            )
        ),
    )
