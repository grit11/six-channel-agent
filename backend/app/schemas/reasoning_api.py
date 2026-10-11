"""六经候选分析 API 的输入输出契约，不暴露模型隐藏推理。"""

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.clinical import ClinicalEvidence, FindingSource
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    DifferentialResult,
    EightPrinciplesState,
    EvidenceGap,
    EvidenceGapDecision,
    FormulaCandidate,
    ReasoningPath,
    ReasoningStageSnapshot,
    SimilarCaseAssessment,
    SyndromeRelationship,
)


class ReasoningRequest(BaseModel):
    """从医生原文开始执行 Phase 3 的最小纵向切片。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=20_000)
    source: FindingSource = Field(default="doctor_input")
    top_k: int = Field(default=3, ge=1, le=5)


class EvidenceReasoningRequest(BaseModel):
    """医生修订后的事实快照，复用相同推理图而不是重新解析原文覆盖修改。"""

    model_config = ConfigDict(extra="forbid")
    clinical_evidence: ClinicalEvidence
    top_k: int = Field(default=3, ge=1, le=5)


class ReasoningResponse(BaseModel):
    """返回可追溯 ClinicalEvidence、八纲倾向和六经 Top-K 候选。"""

    model_config = ConfigDict(extra="forbid")
    clinical_evidence: ClinicalEvidence
    eight_principles: EightPrinciplesState
    candidates: list[ChannelCandidate]
    formula_candidates: list[FormulaCandidate] = Field(default_factory=list)
    differential_results: list[DifferentialResult] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    evidence_gap_decision: EvidenceGapDecision | None = None
    syndrome_relationships: list[SyndromeRelationship] = Field(default_factory=list)
    thread_id: str = Field(description="本次 LangGraph checkpoint 线程标识")
    completed_stages: list[str] = Field(default_factory=list)
    reasoning_paths: list[ReasoningPath] = Field(
        default_factory=list, description="待医生审核的 Top-K 路径"
    )
    evidence_catalog: list[Evidence] = Field(
        default_factory=list, description="本次路径引用的精确来源目录"
    )
    excluded_paths: list[ReasoningPath] = Field(
        default_factory=list, description="来源明确排除的审计路径，不进入可采用 Top-K"
    )
    similar_cases: list[SimilarCaseAssessment] = Field(
        default_factory=list, description="后置历史医案核验，不影响经典排序"
    )
    stage_snapshots: list[ReasoningStageSnapshot] = Field(
        default_factory=list, description="本轮公开阶段回放，旧快照无数据不补造"
    )


class RetrospectiveReplayStep(BaseModel):
    """逐步放出信息的输入；每步只包含当时已知信息，按顺序累积文本。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    label: str = Field(min_length=1, max_length=128, description="步骤名称")
    text: str = Field(
        min_length=1, max_length=5000, description="此步新增病例原文，仅由医生选择信息"
    )


class RetrospectiveReplayRequest(BaseModel):
    """医案回放请求；原医案结论只在各步推理后比较，不注入患者事实。"""

    model_config = ConfigDict(extra="forbid")
    steps: list[RetrospectiveReplayStep] = Field(min_length=1, max_length=5)
    case_evidence_id: str | None = Field(
        default=None, max_length=128, description="可选真实 L5 医案引用"
    )
    top_k: int = Field(default=3, ge=1, le=5)


class RetrospectiveReplayResult(BaseModel):
    """每步独立线程重算，原医案标签匹配只作工程回放指标，非临床金标准。"""

    label: str
    result: ReasoningResponse
    historical_channel_covered: bool | None = None


class RetrospectiveReplayResponse(BaseModel):
    """可审计逐步回放结果，不写入正式会诊或复制历史剂量。"""

    steps: list[RetrospectiveReplayResult]
    case_evidence_id: str | None = None
    historical_channels: list[str] = Field(default_factory=list)
    knowledge_version: str
    version: str = "retrospective-replay/v1.0"
    notice: str = "原医案一致性仅为工程回放参考，不是现代临床金标准；各步不注入原辨证和随访结论"
