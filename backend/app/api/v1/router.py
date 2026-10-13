"""
V1 API 路由汇总。

职责：组合各领域路由，不承载业务逻辑或基础设施选择。
"""

from fastapi import APIRouter

from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.knowledge import router as knowledge_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(knowledge_router)
