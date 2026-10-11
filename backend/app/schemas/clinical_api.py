"""临床事实解析 API 契约：只承载医生输入和可编辑事实，不承载六经判断。"""

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.clinical import ClinicalEvidence, FindingSource


class ClinicalParseRequest(BaseModel):
    """临床解析请求，输入来自医生工作台或结构化病历。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=20_000, description="医生输入的临床原文")
    source: FindingSource = Field(default="doctor_input", description="事实来源")


class ClinicalParseResponse(BaseModel):
    """解析结果快照，前端可在提交后逐条修订。"""

    model_config = ConfigDict(extra="forbid")
    clinical_evidence: ClinicalEvidence
    parser_version: str = Field(default="rules-v0.2", min_length=1)
