"""检索结果数据契约，来源信息与召回、重排分数分开保存。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceGrade = Literal["direct", "indirect", "contradict", "background", "irrelevant"]


class SourceConditionAssessment(BaseModel):
    """来源条件的公开规范化结果；替代项只表达原文逻辑，不生成医学规则。"""

    model_config = ConfigDict(extra="forbid")
    source_text: str = Field(min_length=1, description="未经改写的来源条件")
    kind: Literal["symptom", "temporal", "treatment", "mechanism", "question", "unparsed"] = Field(
        description="来源条件用途；说明和治疗不作为当前症状"
    )
    alternatives: list[dict[str, str]] = Field(
        default_factory=list, description="OR 分支；每个分支内的事实要求为 AND"
    )
    status: Literal["supports", "contradicts", "unknown"] = Field(
        default="unknown", description="整体条件与当前事实的三值匹配"
    )
    unresolved_reason: str | None = Field(default=None, description="无法完整执行的原因")
    normalization_version: str = Field(default="source-condition/v2", description="规范化版本")


class EvidenceAssessment(BaseModel):
    """单条来源对当前路径的适用性核对，不改变原始知识记录或触发硬排除。"""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(min_length=1, description="精确核验的原始来源 ID")
    grade: EvidenceGrade = Field(description="当前路径的证据用途，不代表诊断结论")
    rationale: str = Field(min_length=1, description="可审计的简短核对说明，不保存隐藏推理")
    supporting_requirements: list[str] = Field(
        default_factory=list, description="当前事实支持的来源要求"
    )
    contradicting_requirements: list[str] = Field(
        default_factory=list, description="明确相反的来源要求"
    )
    unresolved_requirements: list[str] = Field(
        default_factory=list, description="缺失或不能完整规范化的来源要求"
    )
    condition_assessments: list[SourceConditionAssessment] = Field(
        default_factory=list, description="条件规范化与当前事实核验审计"
    )
    grader_version: str = Field(default="grounded-v2", description="适用性核对规则版本")


class RetrievedDocument(BaseModel):
    """RAG 证据候选，属于知识记录而非患者诊断。"""

    knowledge_id: str = Field(description="原知识库记录 ID")
    knowledge_type: str = Field(description="知识类型")
    knowledge_version: str = Field(description="知识版本")
    source: str = Field(description="原始来源")
    source_level: str = Field(description="证据来源层级")
    content: str = Field(description="来源内容")
    retrieval_score: float = Field(description="召回相关度分数")
    rerank_score: float | None = Field(default=None, description="独立重排分数")
    metadata: dict[str, object] = Field(default_factory=dict, description="可追溯元数据")
    retrieval_method: str = Field(default="dense", description="召回或融合方式")
    channel_scores: dict[str, float] = Field(default_factory=dict, description="各召回通道原始分")


class Evidence(RetrievedDocument):
    """带版本来源 ID 的引用，不声明医学诊断概率。"""

    evidence_id: str = Field(description="知识版本与真实来源 ID 的组合引用")
    retrieval_method: str = Field(description="精确、BM25、Dense 或 Hybrid 方式")


class DifferentialRule(Evidence):
    """可追溯原始鉴别规则，尚未对患者事实应用。"""

    rule_id: str = Field(description="来源规则 ID")
    candidates: list[str] = Field(description="原始候选组")
    key_dimensions: dict[str, str] = Field(description="来源鉴别维度")
    decision_rules: list[str] = Field(description="来源鉴别说明")


class EvidenceBundle(BaseModel):
    """分层 Evidence 集合，医案不得覆盖经典。"""

    framework: list[Evidence] = Field(description="核心框架")
    classic: list[Evidence] = Field(description="经典原文")
    formulas: list[Evidence] = Field(description="结构化方证")
    differential: list[DifferentialRule] = Field(description="鉴别规则")
    cases: list[Evidence] = Field(description="历史医案")
