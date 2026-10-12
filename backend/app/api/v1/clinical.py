"""临床输入 API：调用事实解析节点并返回可编辑 ClinicalEvidence，不进行辨证。"""

from fastapi import APIRouter, Request

from backend.app.nodes.clinical_parser import RULE_PARSER_VERSION
from backend.app.schemas.clinical_api import ClinicalParseRequest, ClinicalParseResponse

router = APIRouter(prefix="/clinical", tags=["clinical"])


@router.post("/parse", response_model=ClinicalParseResponse)
async def parse_clinical(payload: ClinicalParseRequest, request: Request) -> ClinicalParseResponse:
    """将医生原文转为三值临床事实，未提及的信息不生成阴性。"""
    service = request.app.state.container.reasoning_service
    return ClinicalParseResponse(
        clinical_evidence=await service.parse_text(payload.text, source=payload.source),
        parser_version="llm-clinical/v1.2" if service.llm is not None else RULE_PARSER_VERSION,
    )
