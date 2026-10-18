"""V2 临床 DTO：表达原文事实和公开证据状态，不包含医学规则或 Provider。

模型草稿与程序签发事实分离，避免模型自行生成来源验证标志及事实身份。
"""

import hashlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Polarity = Literal["positive", "negative", "unknown"]
TimeScope = Literal["current", "past", "before_treatment", "after_treatment", "unspecified"]
SixChannel = Literal["太阳", "阳明", "少阳", "太阴", "少阴", "厥阴"]


class V2Model(BaseModel):
    """所有 V2 写入对象拒绝多余字段，防止旧评分字段混入。"""

    model_config = ConfigDict(extra="forbid")


class SourceSpanV2(V2Model):
    """连续原文范围；完整原文一致性在验证器中检查。"""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """拒绝空范围和与摘录长度矛盾的偏移。"""
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("原文范围与文本长度不一致")
        return self


class FindingDraft(V2Model):
    """模型仅报告自然概念、原文与语义，不签发标准键或校验状态。"""

    local_id: str = Field(min_length=1, max_length=80)
    concept: str = Field(min_length=1, max_length=150)
    polarity: Polarity
    time_scope: TimeScope
    source_text: str = Field(min_length=1, max_length=2000)
    occurrence: int = Field(default=0, ge=0, description="相同原文的零起始出现序号")
    needs_clarification: bool = False


class S1ExtractionDraft(V2Model):
    """S1 精简输出，不包含六经、方剂或诊断。"""

    findings: list[FindingDraft] = Field(max_length=200)
    chief_complaint_text: str | None = None
    unresolved_phrases: list[str] = Field(default_factory=list, max_length=30)
    conflicts: list[list[str]] = Field(default_factory=list)


class FindingV2(V2Model):
    """程序签发身份与原文来源；医学解释仍需医生审核。"""

    fact_id: str
    concept: str
    normalized_key: str | None = None
    polarity: Polarity
    time_scope: TimeScope
    source_span: SourceSpanV2
    review_status: Literal["verified_source", "needs_clarification", "doctor_verified"]


class ClinicalEvidenceV2(V2Model):
    """原文与事实版本；不把未提及补成阴性，不承载六经结论。"""

    raw_text: str = Field(min_length=1, max_length=20000)
    findings: list[FindingV2]
    chief_complaint_span: SourceSpanV2 | None = None
    conflicts: list[list[str]] = Field(default_factory=list)
    unresolved_phrases: list[SourceSpanV2] = Field(default_factory=list)
    doctor_revision: int = Field(default=0, ge=0)
    source: Literal["doctor_input", "doctor_edit", "record"] = "doctor_input"

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """拒绝越界、伪造引用和未知冲突 ID，不重新解释医学极性。"""
        ids = [finding.fact_id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("事实身份重复")
        spans = [finding.source_span for finding in self.findings] + self.unresolved_phrases
        if self.chief_complaint_span:
            spans.append(self.chief_complaint_span)
        if any(self.raw_text[s.start : s.end] != s.text for s in spans):
            raise ValueError("事实来源不属于连续原文")
        if any(len(group) < 2 or not set(group).issubset(ids) for group in self.conflicts):
            raise ValueError("冲突事实身份无效")
        return self

    def fingerprint(self) -> str:
        """完整事实与医生版本共同决定签名，避免复用旧患者结论。"""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class ChannelHypothesisV2(V2Model):
    """六经检索假设，解释来自模型，引用必须属于本例。"""

    candidate_id: str = Field(min_length=1)
    six_channels: list[SixChannel] = Field(min_length=1, max_length=6)
    support_fact_ids: list[str]
    counter_fact_ids: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    retrieval_focus: list[str] = Field(default_factory=list)
    short_basis: str = Field(max_length=300)


class ClarificationQuestionV2(V2Model):
    """有界、针对当前不确定项的医生问题，不生成患者事实。"""

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=300)
    targets: list[str]
    source_fact_ids: list[str] = Field(default_factory=list)
    expected_effect: str = Field(max_length=300)


class ChannelLocalizationV2(V2Model):
    """S2 可未决、多候选；流程守卫禁止未知时进入检索。"""

    status: Literal["resolved", "ambiguous", "unknown", "conflicting"]
    candidates: list[ChannelHypothesisV2] = Field(max_length=3)
    proceed_to_retrieval: bool
    clarification_questions: list[ClarificationQuestionV2] = Field(max_length=3)
    note: str = Field(max_length=500)

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        """只校验状态契约，不根据医学词项改变候选。"""
        if self.status in {"unknown", "conflicting"} and self.proceed_to_retrieval:
            raise ValueError("未知或冲突六经不得检索")
        if self.status == "unknown" and self.candidates:
            raise ValueError("无据定位不得凑候选")
        if self.proceed_to_retrieval and (
            not self.candidates or any(not item.support_fact_ids for item in self.candidates)
        ):
            raise ValueError("可检索候选必须有事实支持")
        if self.status == "resolved" and not self.proceed_to_retrieval:
            raise ValueError("resolved 必须具有可检索依据")
        return self


class HumanGraphCommandV2(V2Model):
    """业务层传入的已校验人工动作，与模型输出完全隔离。"""

    kind: Literal["clarify", "review", "cancel"]
    consultation_id: str
    revision: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    action_turn: int = Field(ge=1)
    clinical_text: str | None = None
    clinical_evidence: ClinicalEvidenceV2 | None = None
    no_additional_information: bool = False
    review: dict[str, object] | None = None


class RetrievalQueryPlanV2(V2Model):
    """S3 查询计划；字符串组装不承担六经或方证医学判断。"""

    channel_hypotheses: list[str] = Field(max_length=6)
    positive_facts: list[str] = Field(default_factory=list)
    explicit_negative_facts: list[str] = Field(default_factory=list)
    raw_synonym_phrases: list[str] = Field(default_factory=list)
    primary_query: str = Field(min_length=1)
    differential_query: str | None = None
    rescue_query: str | None = None
    filter_mode: Literal["soft_channel", "open_recovery"]
    query_variant: Literal["Q0_baseline"] = "Q0_baseline"
    positive_fact_ids: list[str] = Field(default_factory=list)
    negative_fact_ids: list[str] = Field(default_factory=list)
    treatment_after_fact_ids: list[str] = Field(default_factory=list)
    excluded_temporal_fact_ids: dict[str, str] = Field(default_factory=dict)
    primary_channel_indices: list[int] = Field(default_factory=list)
    secondary_channel_indices: list[int] = Field(default_factory=list)
    primary_focus_indices: list[int] = Field(default_factory=list)
    secondary_focus_indices: list[int] = Field(default_factory=list)


class EvidenceRefV2(V2Model):
    """可回查的真实来源；相关性分数不代表医学置信度。"""

    evidence_id: str = Field(min_length=1)
    knowledge_type: str = Field(min_length=1)
    knowledge_version: str = Field(min_length=1)
    clause_no: str | None = None
    formula_name: str | None = None
    excerpt: str = Field(min_length=1)
    review_status: str = Field(min_length=1)
    retrieval_score: float | None = None
    rerank_score: float | None = None


class FormulaEvidenceCandidateV2(V2Model):
    """一个方证对象及其来源集合；多条同方条文只占一个候选位。"""

    candidate_id: str = Field(min_length=1)
    formula_name: str = Field(min_length=1)
    six_channel_tags: list[str] = Field(default_factory=list)
    source_evidence: list[EvidenceRefV2] = Field(min_length=1, max_length=3)
    provenance: Literal["classic", "other_text", "historical_case"]
    coverage_notes: list[str] = Field(default_factory=list)


class RetrievalBundleV2(V2Model):
    """S3 输出真实证据与不同方证候选，不输出医学排序分。"""

    query_plan: RetrievalQueryPlanV2
    formula_candidates: list[FormulaEvidenceCandidateV2] = Field(max_length=5)
    retrieved_evidence_ids: list[str]
    retrieval_trace: dict[str, object] = Field(default_factory=dict)
    retrieval_status: Literal["ok", "partial", "no_relevant_evidence"]


class CandidateComparisonDraftV2(V2Model):
    """S4 模型草稿；只表达相对比较，不生成工程身份或来源签名。"""

    candidate_id: str = Field(min_length=1)
    six_channels: list[SixChannel] = Field(min_length=1, max_length=6)
    rank: int | None = Field(default=None, ge=1, le=3)
    support_fact_ids: list[str]
    counter_fact_ids: list[str]
    missing_dimensions: list[str]
    support_evidence_ids: list[str]
    counter_evidence_ids: list[str]
    brief_rationale: str = Field(max_length=120)
    differential_summary: str = Field(max_length=500)


class ComparisonDraftV2(V2Model):
    """S4 模型输出草稿；工程发布字段由程序绑定。"""

    status: Literal["ranked", "insufficient", "needs_clarification"]
    ordered_paths: list[CandidateComparisonDraftV2] = Field(max_length=3)
    next_questions: list[ClarificationQuestionV2] = Field(default_factory=list, max_length=3)
    unresolved_reason: str | None = None


class CandidateComparisonV2(V2Model):
    """S4 发布对象；工程身份、方名和核验状态由程序签发。"""

    path_id: str
    candidate_id: str
    six_channels: list[str]
    formula_name: str | None = None
    rank: int | None = Field(default=None, ge=1, le=3)
    support_fact_ids: list[str]
    counter_fact_ids: list[str]
    missing_dimensions: list[str]
    support_evidence_ids: list[str]
    counter_evidence_ids: list[str]
    brief_rationale: str = Field(max_length=120)
    differential_summary: str = Field(max_length=500)
    verification_status: Literal["pending", "grounded", "needs_clinician_review"]


class ComparisonV2(V2Model):
    """批量模型比较结果，允许无可靠排序或需要补证。"""

    status: Literal["ranked", "insufficient", "needs_clarification"]
    ordered_paths: list[CandidateComparisonV2] = Field(max_length=3)
    provisional_candidates: list[CandidateComparisonV2] = Field(
        default_factory=list,
        max_length=3,
        description="未决状态下的探索候选，不构成正式排名",
    )
    next_questions: list[ClarificationQuestionV2] = Field(default_factory=list, max_length=3)
    unresolved_reason: str | None = None
    source_evidence: list[EvidenceRefV2] = Field(default_factory=list)


class ClinicalV2RunOutcome(V2Model):
    """V2 临床运行联合结果；中途暂停不是失败，也不要求已有比较结果。"""

    kind: Literal[
        "needs_channel_clarification",
        "awaiting_doctor_review",
        "undetermined",
        "completed",
        "failed",
    ]
    consultation_id: str
    revision: int = Field(ge=0)
    evidence: ClinicalEvidenceV2 | None = None
    localization: ChannelLocalizationV2 | None = None
    retrieval: RetrievalBundleV2 | None = None
    comparison: ComparisonV2 | None = None
    paused_at: Literal["s2_clarification", "s5_doctor_review"] | None = None
    failure_code: str | None = None


class V2Event(V2Model):
    """V2 阶段事件摘要；完整事实和来源始终从权威快照读取。"""

    event: Literal[
        "stage_started",
        "stage_completed",
        "clarification_requested",
        "review_requested",
        "consultation_completed",
    ]
    consultation_id: str
    revision: int = Field(ge=0)
    stage: Literal["s1_extract", "s2_localize", "s3_retrieve", "s4_compare", "s5_doctor_review"]
