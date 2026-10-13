"""V2 解析和研究运行 API：只委托应用服务，不包含医学逻辑。"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.app.clinical_v2.schemas import ClinicalEvidenceV2
from backend.app.core.exceptions import ConfigurationError
from backend.app.schemas.consultation import ConsultationSnapshot

router = APIRouter(tags=["clinical-v2"])


class ClinicalParseRequest(BaseModel):
    """只接收病例原文，禁止混入旧评分字段。"""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20000)


@router.post("/clinical/parse", response_model=ClinicalEvidenceV2)
async def parse(request: Request, payload: ClinicalParseRequest) -> ClinicalEvidenceV2:
    """单独运行 S1 事实抽取，不进入定位、检索或审核。"""
    service = request.app.state.container.clinical_v2_service
    if service is None:
        raise ConfigurationError("V2 临床模型未配置")
    return await service.parse(payload.text)


@router.post("/reasoning/evaluate", response_model=ConsultationSnapshot)
async def evaluate(request: Request, payload: ClinicalParseRequest) -> ConsultationSnapshot:
    """研究运行仍保存会诊和人工暂停，不要求完整 Top-3。"""
    return await request.app.state.container.consultation_service.create(payload.text)
