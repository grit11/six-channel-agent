"""八纲和六经候选的数据契约，只表达可审核排序结果，不替代医生诊断。"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.evidence import EvidenceAssessment, SourceConditionAssessment

AxisLean = Literal["left", "right", "mixed", "uncertain"]
CandidateStatus = Literal["active", "weak", "excluded"]


class ScoringPolicy(BaseModel):
    """RP-13 工程权重快照；未知不罚分，参数须由后续专家集校准。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    core: float = Field(default=0.25, ge=0, le=1, description="核心主证权重")
    differential: float = Field(default=0.25, ge=0, le=1, description="关键鉴别权重")
    classic: float = Field(default=0.20, ge=0, le=1, description="经典证据权重")
    formula: float = Field(default=0.10, ge=0, le=1, description="方证结构权重")
    temporal: float = Field(default=0.10, ge=0, le=1, description="时间关系权重")
    case: float = Field(default=0.05, ge=0, le=1, description="后置医案权重")
    context: float = Field(default=0.05, ge=0, le=1, description="其他已核验上下文权重")
    high_penalty: float = Field(default=0.35, gt=0, le=1, description="高权重明确反证惩罚")
    medium_penalty: float = Field(default=0.15, ge=0, le=1, description="普通明确反证惩罚")
    high_threshold: float = Field(default=0.8, gt=0, le=1, description="高区分度阈值")

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """权重总和固定为一，高反证惩罚须高于一般主证正证权重。"""
        total = sum(
            getattr(self, key)
            for key in ("core", "differential", "classic", "formula", "temporal", "case", "context")
        )
        if abs(total - 1.0) > 1e-8:
            raise ValueError("RP-13 正向评分权重总和必须为 1")
        if self.high_penalty <= max(self.medium_penalty, self.core):
            raise ValueError("高反证惩罚必须高于普通反证和核心主证权重")
        return self


class PrincipleAxis(BaseModel):
    """单个八纲轴的倾向、支持事实和未知事实。"""

    model_config = ConfigDict(extra="forbid")
    lean: AxisLean
    score_left: float = Field(ge=0, le=1)
    score_right: float = Field(ge=0, le=1)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)
    unknown_key_findings: list[str] = Field(default_factory=list)


class EightPrinciplesState(BaseModel):
    """八纲只是证据倾向，不把四轴压缩成一个诊断标签。"""

    model_config = ConfigDict(extra="forbid")
    yin_yang: PrincipleAxis
    exterior_interior: PrincipleAxis
    cold_heat: PrincipleAxis
    deficiency_excess: PrincipleAxis
    summary: str = Field(min_length=1)
    framework_evidence_ids: list[str] = Field(
        default_factory=list, description="约束八纲解释的真实框架来源 ID"
    )


class ChannelCandidate(BaseModel):
    """一个六经宽候选及其支持、反证、缺失证据和来源引用。"""

    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    six_channels: list[str] = Field(min_length=1)
    provisional_syndrome: str | None = None
    independent_score: float = Field(ge=0, le=1)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)
    missing_key_findings: list[str] = Field(default_factory=list)
    classic_evidence_ids: list[str] = Field(default_factory=list)
    formula_evidence_ids: list[str] = Field(default_factory=list)
    framework_evidence_ids: list[str] = Field(default_factory=list)
    evidence_completeness: float = Field(ge=0, le=1)
    pass_a_score: float | None = Field(
        default=None, ge=0, le=1, description="仅基于事实与八纲的宽召回分"
    )
    evidence_adjustment: float = Field(
        default=0.0, ge=-1, le=1, description="真实来源适用性校正量，不代表临床概率"
    )
    pass_b_score: float | None = Field(
        default=None, ge=0, le=1, description="经过来源适用性校正的排序分"
    )
    evidence_assessments: list[EvidenceAssessment] = Field(
        default_factory=list, description="Pass B 来源条件核验快照，用于复算校正量"
    )
    status: CandidateStatus


class FormulaCandidate(BaseModel):
    """方证候选的来源与匹配摘要，不包含处方剂量或治疗指令。"""

    model_config = ConfigDict(extra="forbid")
    formula_name: str = Field(min_length=1)
    association_scope: Literal["classic_clause_based", "case_derived_historical"]
    canonical_six_channels: list[str] = Field(default_factory=list)
    case_derived_six_channels: list[str] = Field(default_factory=list)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)
    missing_key_findings: list[str] = Field(default_factory=list)
    source_clause_nos: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    branch_ids: list[str] = Field(default_factory=list)
    branch_clause_evidence_ids: list[str] = Field(default_factory=list)
    branch_status: str = "unavailable"
    risk_level: str | None = None
    rank_score: float = Field(ge=0, le=1)


class DifferentialDimension(BaseModel):
    """鉴别规则中的一个维度及当前事实状态。"""

    model_config = ConfigDict(extra="forbid")
    dimension: str = Field(min_length=1)
    candidate_values: dict[str, str] = Field(default_factory=dict)
    observed_value: str | None = None
    observed_polarity: str = Field(min_length=1)
    discriminator_weight: float = Field(ge=0, le=1)
    source_rule_id: str = Field(min_length=1)
    status: Literal["supports", "contradicts", "unknown", "neutral"]
    condition_assessments: dict[str, SourceConditionAssessment] = Field(
        default_factory=dict, description="按候选保存完整条件及不可执行原因"
    )


class DifferentialResult(BaseModel):
    """候选组鉴别结果，保留规则来源和未解决维度。"""

    model_config = ConfigDict(extra="forbid")
    candidate_ids: list[str] = Field(min_length=2)
    shared_features: list[str] = Field(default_factory=list)
    dimensions: list[DifferentialDimension] = Field(default_factory=list)
    decisive_support: list[str] = Field(default_factory=list)
    decisive_contradictions: list[str] = Field(default_factory=list)
    unresolved_dimensions: list[str] = Field(default_factory=list)


class EvidenceGapPolicy(BaseModel):
    """RP-11 启发式参数；用于提问排序和流程停止，不表示诊断概率。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    separation_threshold: float = Field(default=0.15, gt=0, le=1)
    high_weight_threshold: float = Field(default=0.8, gt=0, le=1)
    minimum_priority: float = Field(default=0.1, gt=0, le=1)
    answerability: float = Field(default=1.0, gt=0, le=1)
    reviewed_source_reliability: float = Field(default=0.8, gt=0, le=1)
    unreviewed_source_reliability: float = Field(default=0.4, ge=0, le=1)
    repetition_penalty: float = Field(default=0.1, ge=0, le=1)
    question_cost: float = Field(default=0.05, ge=0, le=1)


class EvidenceGapContext(BaseModel):
    """医生补证控制快照；无法获得是流程状态，不能写成阴性事实。"""

    model_config = ConfigDict(extra="forbid")
    asked_counts: dict[str, int] = Field(default_factory=dict)
    unavailable_question_ids: list[str] = Field(default_factory=list)
    supplementation_round: int = Field(default=0, ge=0)


class EvidenceGapDecision(BaseModel):
    """可审计停止判定；停止只转交审核，不采用任何路径。"""

    model_config = ConfigDict(extra="forbid")
    stop: bool
    reason: Literal[
        "continue_supplement",
        "score_separation",
        "no_actionable_gaps",
        "all_unavailable",
        "max_rounds",
        "doctor_current_information",
    ]
    actionable_count: int = Field(default=0, ge=0)
    high_weight_counterevidence_checked: bool = False
    separation: float | None = Field(default=None, ge=0, le=1)
    policy: EvidenceGapPolicy = Field(default_factory=EvidenceGapPolicy)
    version: str = "rp11/v1.0"


class EvidenceGap(BaseModel):
    """最值得由医生补充的区分性信息，不与未知事实混同为阴性。"""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    priority: float = Field(ge=0, le=1)
    source_rule_ids: list[str] = Field(default_factory=list)
    candidate_coverage: int = Field(ge=1)
    question_id: str = ""
    finding_keys: list[str] = Field(default_factory=list)
    status: Literal["pending", "unavailable", "deferred"] = "pending"
    unresolved_reason: str | None = None
    priority_factors: dict[str, float] = Field(default_factory=dict)
    algorithm_version: str = "rp11/v1.0"


class TemporalEvidenceAssessment(BaseModel):
    """时间线/治疗记录的公开事实核对，不能冒充医学因果结论。"""

    model_config = ConfigDict(extra="forbid")
    record_type: Literal["timeline", "treatment"]
    record_id: str
    source_text: str
    status: Literal["usable", "incomplete", "conflicting", "unsupported"]
    supporting_findings: list[str] = Field(default_factory=list)
    missing_findings: list[str] = Field(default_factory=list)
    reason: str


RelationshipType = Literal[
    "single",
    "concurrent",
    "combined_disease",
    "sequential_concurrent_disease",
    "direct_invasion",
    "twofold_attack",
    "transmission",
    "changed_syndrome",
    "spoiled_disease",
    "uncertain",
]


class RelationshipAssessment(BaseModel):
    """逐类关系条件审计，缺失与明确反向条件分开；不保存模型隐藏推理。"""

    model_config = ConfigDict(extra="forbid")
    relationship_type: RelationshipType
    status: Literal["supported", "insufficient", "contradicted"]
    evidence_ids: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    contradicting_conditions: list[str] = Field(default_factory=list)
    rationale: str


class SyndromeRelationship(BaseModel):
    """病证关系及来源；多候选不自动等同于合病、并病或传经。"""

    model_config = ConfigDict(extra="forbid")
    relationship_type: Literal[
        "single",
        "concurrent",
        "combined_disease",
        "sequential_concurrent_disease",
        "direct_invasion",
        "twofold_attack",
        "transmission",
        "changed_syndrome",
        "spoiled_disease",
        "uncertain",
    ]
    primary_candidate_id: str = Field(min_length=1)
    secondary_candidate_ids: list[str] = Field(default_factory=list)
    temporal_evidence: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_timeline_evidence: list[str] = Field(default_factory=list)
    temporal_assessments: list[TemporalEvidenceAssessment] = Field(default_factory=list)
    treatment_evidence: list[str] = Field(default_factory=list)
    relationship_version: str = "relationship/v2.0"
    assessments: list[RelationshipAssessment] = Field(
        default_factory=list, description="九类关系条件审计"
    )
    previous_six_channels: list[str] = Field(
        default_factory=list, description="已核验历史经，不伪造当前候选 ID"
    )
    current_six_channels: list[str] = Field(default_factory=list, description="关系当前经")
    source_assessments: list[EvidenceAssessment] = Field(
        default_factory=list, description="精确来源类型及用途核验"
    )


class SimilarCaseAssessment(BaseModel):
    """相似医案公开核验；只作历史参考，不修改 canonical、当前方证或排名。"""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    path_ids: list[str] = Field(default_factory=list)
    title: str
    status: Literal["reference", "conflicting", "insufficient", "irrelevant"]
    matching_conditions: list[str] = Field(default_factory=list)
    conflicting_conditions: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    condition_assessments: list[SourceConditionAssessment] = Field(default_factory=list)
    similarity_score: float = Field(
        default=0, ge=0, le=1, description="唯一可核验症状覆盖率，非临床概率"
    )
    reason: str
    version: str = "similar-case/v1.0"


class ReasoningStageSnapshot(BaseModel):
    """本轮图阶段公开 DTO 快照；不可执行、无模型隐藏推理或连接对象。"""

    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=1)
    stage: str
    payload: dict[str, object] = Field(default_factory=dict)
    version: str = "stage-replay/v1.0"
    input_digest: str | None = None


class PathConditionCoverage(BaseModel):
    """完整度的唯一来源条件单元；三值分开计数，重复来源不重复计算。"""

    model_config = ConfigDict(extra="forbid")
    source_text: str
    kind: str
    status: Literal["supports", "contradicts", "unknown"]
    evidence_ids: list[str] = Field(default_factory=list)
    unavailable: bool = False


class PathCompleteness(BaseModel):
    """可复算来源条件覆盖率，既非匹配率也非临床充分性。"""

    model_config = ConfigDict(extra="forbid")
    supported_count: int = Field(default=0, ge=0)
    contradicted_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    total_count: int = Field(default=0, ge=0)
    conditions: list[PathConditionCoverage] = Field(default_factory=list)
    missing_timeline: list[str] = Field(default_factory=list)
    version: str = "path-completeness/v1.0"


class PathKeyDifference(BaseModel):
    """双方来源要求和当前事实核对；只保存公开结论，不保存隐藏推理。"""

    model_config = ConfigDict(extra="forbid")
    dimension: str
    evidence_id: str
    current_requirement: str
    next_requirement: str
    current_status: Literal["supports", "contradicts", "unknown"]
    next_status: Literal["supports", "contradicts", "unknown"]
    finding_keys: list[str] = Field(default_factory=list)
    conclusion: Literal[
        "favors_current", "favors_next", "shared", "unresolved", "both_contradicted"
    ]


class PathComparison(BaseModel):
    """与完整排序中次选的关键差异；无可执行规则时明确保留未定。"""

    model_config = ConfigDict(extra="forbid")
    next_path_id: str
    next_syndrome_name: str
    next_formula_pattern: str | None = None
    differences: list[PathKeyDifference] = Field(default_factory=list)
    status: Literal["compared", "no_source_rule", "same_formula", "no_formula"]
    summary: str


class TreatmentPrincipleSummary(BaseModel):
    """候选资料的来源治法摘要，不是对患者的治疗或处方指令。"""

    model_config = ConfigDict(extra="forbid")
    status: Literal["available", "ambiguous", "no_source", "historical", "insufficient_support"]
    text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_field: str = "treatment_method"
    reason: str


class ReasoningPath(BaseModel):
    """供医生审核的路径快照；排序分属于工程基线，不表达诊断概率或处方。"""

    model_config = ConfigDict(extra="forbid")
    rank: int = Field(ge=1, description="本次输出内的排名")
    path_id: str = Field(min_length=1, description="由候选和来源身份生成的稳定路径 ID")
    channel_candidate_ids: list[str] = Field(min_length=1, description="参与路径的六经候选 ID")
    six_channels: list[str] = Field(min_length=1, description="路径涉及的六经")
    syndrome_name: str = Field(min_length=1, description="待核验的候选名称")
    relationship: str | None = Field(default=None, description="已核验关系或 uncertain")
    eight_principles_summary: EightPrinciplesState = Field(description="上游八纲倾向快照")
    supporting_findings: list[str] = Field(default_factory=list, description="上游支持事实摘要")
    contradicting_findings: list[str] = Field(default_factory=list, description="上游反证摘要")
    missing_key_findings: list[str] = Field(
        default_factory=list, description="待补充事实与鉴别维度"
    )
    formula_pattern: str | None = Field(default=None, description="方证参考名称，不是处方")
    association_scope: Literal["classic_clause_based", "case_derived_historical"] | None = Field(
        default=None, description="方证关联来源；历史医案派生不得升级为经典归属"
    )
    treatment_principle: str | None = Field(
        default=None, description="来源唯一的候选治法摘要，由医生审核，非治疗指令"
    )
    treatment_summary: TreatmentPrincipleSummary | None = None
    completeness_detail: PathCompleteness | None = None
    comparison_with_next: PathComparison | None = None
    summary_version: str = "path-summary/v1.0"
    framework_evidence: list[str] = Field(default_factory=list, description="已展开的框架引用")
    classic_evidence: list[str] = Field(default_factory=list, description="已展开的经典引用")
    formula_evidence: list[str] = Field(default_factory=list, description="已展开的方证引用")
    differential_evidence: list[str] = Field(
        default_factory=list, description="精确候选对的鉴别引用"
    )
    case_evidence: list[str] = Field(default_factory=list, description="医案引用，本切片不新增召回")
    evidence_assessments: list[EvidenceAssessment] = Field(
        default_factory=list, description="按当前路径核对的引用用途；不适用来源只保留审计 ID"
    )
    ranking_score: float = Field(ge=0, le=1, description="上游排序分聚合，不代表临床概率")
    score_components: dict[str, float] = Field(description="可审计的原始分及聚合权重")
    evidence_completeness: float = Field(
        ge=0, le=1, description="唯一来源条件中已知支持或反证的比例，非临床充分性"
    )
    separation_from_next: float | None = Field(default=None, ge=0, le=1, description="与次选分差")
    risk_level: str | None = Field(default=None, description="来源风险标记")
    warnings: list[str] = Field(default_factory=list, description="来源适用性和工程基线限制")
    review_status: Literal["pending", "accepted", "modified", "rejected"] = Field(
        default="pending", description="医生审核状态；构建节点只输出 pending"
    )
    candidate_status: CandidateStatus = Field(default="active", description="排序或排除状态")
    exclusion_reasons: list[str] = Field(default_factory=list, description="带来源的排除依据")
    scoring_version: str = Field(default="rp13/v1.0", description="评分算法版本")
