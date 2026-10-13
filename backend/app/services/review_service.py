"""V2 人工审核应用边界，依赖会诊协调服务，不作模型或规则判断。"""

from uuid import UUID

from backend.app.clinical_v2.schemas import HumanGraphCommandV2
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.services.consultation_service import ConsultationService


class ReviewService:
    """人工审核统一进入会诊幂等和版本校验。"""

    def __init__(self, consultations: ConsultationService) -> None:
        """注入应用级会诊服务。"""
        self.consultations = consultations

    async def review(self, identifier: UUID, command: HumanGraphCommandV2) -> ConsultationSnapshot:
        """恢复当前 S5 中断，不允许模型自动审核。"""
        return await self.consultations.resume(identifier, command)
