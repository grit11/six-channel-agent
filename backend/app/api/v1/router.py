"""
V1 API 路由汇总。

职责：组合各领域路由，不承载业务逻辑或基础设施选择。
"""

from fastapi import APIRouter

from backend.app.api.v1.clinical import router as clinical_router
from backend.app.api.v1.consultation import router as consultation_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.knowledge import router as knowledge_router
from backend.app.api.v1.reasoning import router as reasoning_router
from backend.app.api.v1.review import router as review_router
from backend.app.api.v1.stream import router as stream_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(clinical_router)
api_router.include_router(knowledge_router)
api_router.include_router(reasoning_router)
api_router.include_router(consultation_router)
api_router.include_router(review_router)
api_router.include_router(stream_router)
