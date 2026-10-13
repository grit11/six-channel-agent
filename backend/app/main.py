"""FastAPI 应用入口，仅装配生命周期、协议错误、业务路由和原生 Console 页面。

职责：在不复制医学规则的前提下，为单页 Console 暴露真实会诊 API、SSE 和知识查询接口。
边界：页面路由与配置状态不执行辨证；临床结果由既有会诊和知识服务产生。
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.v1.router import api_router
from backend.app.api.v2.router import router as v2_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.dependencies import open_container
from backend.app.core.exceptions import (
    ApplicationError,
    ConsultationNotFoundError,
    InvalidBusinessRequestError,
    InvalidStateTransitionError,
    KnowledgeNotFoundError,
)

CONSOLE_HTML = Path(__file__).resolve().parent / "static" / "dashboard.html"


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建可注入配置的应用，所有基础设施由容器在 lifespan 中管理。"""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动时一次性创建依赖，退出时按反序释放资源。"""
        actual_settings = settings or get_settings()
        async with open_container(actual_settings) as container:
            application.state.container = container
            yield

    application = FastAPI(
        title="六经辨证临床辅助系统",
        version="2.0.0",
        description="面向临床医生的证据辅助服务，医生完成最终审核。",
        lifespan=lifespan,
    )
    application.include_router(v2_router)

    @application.exception_handler(ApplicationError)
    async def application_error(request: Request, exc: ApplicationError) -> JSONResponse:
        """将业务异常映射为统一错误，不输出原始异常或敏感连接信息。"""
        return JSONResponse(
            status_code=(
                404
                if isinstance(exc, (ConsultationNotFoundError, KnowledgeNotFoundError))
                else (
                    409
                    if isinstance(exc, InvalidStateTransitionError)
                    else 400 if isinstance(exc, InvalidBusinessRequestError) else 503
                )
            ),
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": str(uuid4()),
                }
            },
        )

    application.include_router(api_router)
    application.mount(
        "/console-static",
        StaticFiles(directory=CONSOLE_HTML.parent),
        name="console-static",
    )

    @application.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        """返回原生单页 Console；页面只负责调用既有业务 API。"""
        return FileResponse(CONSOLE_HTML, media_type="text/html; charset=utf-8")

    @application.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """将根路径指向 Console，便于本地快速演示。"""
        return FileResponse(CONSOLE_HTML, media_type="text/html; charset=utf-8")

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """兼容浏览器默认图标请求，返回同源 SVG 品牌图标。"""
        return FileResponse(CONSOLE_HTML.parent / "favicon.svg", media_type="image/svg+xml")

    @application.get("/api/console/config")
    async def console_config(request: Request) -> dict[str, object]:
        """返回真实容器状态，禁止把未连接的模型或数据库标为已连接。"""
        container = request.app.state.container
        degraded = (
            container.vector_store.status.degraded
            or container.checkpoint_runtime.status.degraded
            or container.database.backend != "postgres"
        )
        return {
            "mode": "live",
            "demo": False,
            "agent_connected": container.clinical_v2_service is not None,
            "consultation_enabled": container.clinical_v2_service is not None,
            "degraded": degraded,
            "vector_backend": container.vector_store.status.backend,
            "checkpoint_backend": container.checkpoint_runtime.status.backend,
            "business_database": container.database.backend,
            "knowledge_version": container.settings.knowledge_base_version,
            "clinical_engine": "clinical-v2.0",
            "knowledge_qa_mode": container.settings.knowledge_qa_mode,
            "retrieval_mode": container.settings.retrieval_mode,
            "reranker_enabled": container.settings.reranker_enabled,
            "llm_model": container.settings.llm_model,
            "counts": container.knowledge_service.counts(),
            "console_version": "2.0",
            "text": (
                "V2 LangGraph 临床图已装配；S5 医生审核是完成入口。"
                if container.clinical_v2_service is not None
                else "知识中心已装配；V2 临床模型尚未配置。"
            ),
        }

    return application


app = create_app()
