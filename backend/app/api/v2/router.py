"""V2 路由汇总：只组合薄 Router，不写六经医学逻辑。"""

from fastapi import APIRouter

from backend.app.api.v2.clinical import router as clinical_router
from backend.app.api.v2.consultation import router as consultation_router
from backend.app.api.v2.events import router as events_router
from backend.app.api.v2.review import router as review_router

router = APIRouter(prefix="/api/v2")
router.include_router(clinical_router)
router.include_router(consultation_router)
router.include_router(review_router)
router.include_router(events_router)
