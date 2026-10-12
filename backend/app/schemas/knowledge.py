"""知识加载与查询契约，仅表示来源事实，不表达患者诊断。"""

import hashlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.evidence import Evidence

KnowledgeType = Literal[
    "classic_clause",
    "formula_pattern",
    "differential",
    "medical_case",
    "six_channel_framework",
]

KnowledgeQueryType = Literal[
    "exact_clause", "formula", "differential", "framework", "case", "mixed"
]


class KnowledgeQARequest(BaseModel):
    """独立知识问题；只接受医生主动提供的查询上下文，不接受会诊/thread 写入口。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000, description="医生知识问题")
    query_type: KnowledgeQueryType | None = Field(default=None, description="空值为自动分类")
    clause_id: str | None = Field(
        default=None, min_length=1, max_length=80, description="精确条文号/ID"
    )
    formula_name: str | None = Field(
        default=None, min_length=1, max_length=100, description="完整方名"
    )
    candidates: list[str] = Field(default_factory=list, max_length=8, description="鉴别候选名")
    context_text: str | None = Field(
        default=None, max_length=4000, description="医生主动复制的查询背景"
    )
    top_k: int = Field(default=5, ge=1, le=10, description="每种来源最多返回数量")

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        """明确目标与类别矛盾时拒绝，避免精确读取悄然变成模糊检索。"""
        if self.clause_id and self.query_type not in {None, "exact_clause", "mixed"}:
            raise ValueError("条文目标须使用 exact_clause 或 mixed")
        if self.formula_name and self.query_type not in {None, "formula", "mixed"}:
            raise ValueError("方名目标须使用 formula 或 mixed")
        if self.clause_id and self.formula_name and self.query_type != "mixed":
            raise ValueError("同时查询条文和方名须显式选择 mixed")
        if any(not candidate.strip() for candidate in self.candidates):
            raise ValueError("鉴别候选名不能为空")
        return self


class KnowledgeQueryPlan(BaseModel):
    """公开查询分类和精确目标，只用于工具路由，不是患者辨证状态。"""

    model_config = ConfigDict(extra="forbid")
    query_type: KnowledgeQueryType = Field(description="RP-21 六类查询")
    clause_ids: list[str] = Field(default_factory=list, description="从问题明确提及的条文号")
    formula_names: list[str] = Field(default_factory=list, description="来源中精确匹配的完整方名")
    candidates: list[str] = Field(default_factory=list, description="医生提供或来源匹配的候选")
    classification_method: Literal["explicit", "rules", "llm"] = Field(description="分类方式")


class KnowledgeAnswerItem(BaseModel):
    """可直接核对的来源摘录；正文和字段均由来源读取，不接受模型生成医学断言。"""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(description="真实来源引用")
    title: str = Field(description="来源标题")
    source_field: str = Field(description="摘录字段")
    text: str = Field(description="未经改写的来源字段文本")


class KnowledgeAnswerSection(BaseModel):
    """按来源类型/等级及经典或历史范围分组，医案不能混入经典正文。"""

    knowledge_type: KnowledgeType = Field(description="来源类型")
    source_level: str = Field(description="来源等级")
    association_scope: str | None = Field(default=None, description="经典/历史派生范围")
    label: str = Field(description="分层展示标题")
    items: list[KnowledgeAnswerItem] = Field(description="带逐项引用的来源摘录")


class KnowledgeQAResponse(BaseModel):
    """独立 QA 结果及公开工具审计；没有临床候选、处方或会诊写入字段。"""

    query_id: str = Field(description="本次独立查询标识")
    plan: KnowledgeQueryPlan = Field(description="分类及精确查询计划")
    status: Literal["answered", "partial", "no_evidence"] = Field(description="来源可回答状态")
    sections: list[KnowledgeAnswerSection] = Field(description="分层来源答案")
    evidence_catalog: list[Evidence] = Field(description="可展开的真实来源、风险和审核状态")
    missing_evidence: list[str] = Field(description="未找到的精确目标或来源层")
    notice: str = Field(description="知识用途及证据不足说明")
    tools_used: list[str] = Field(description="公开工具调用名称，不包含隐藏推理")
    context_used: bool = Field(description="是否使用医生主动提交的背景")
    answer_mode: Literal["extractive", "llm"] = Field(description="来源摘录或模型选择摘录")
    knowledge_version: str = Field(description="知识版本")
    qa_version: str = Field(default="knowledge-qa/v1.0", description="查询协议版本")


class KnowledgeRecord(BaseModel):
    """五类知识统一只读结构，grounded 元数据保持原有含义。"""

    knowledge_id: str = Field(min_length=1, description="来源记录 ID")
    knowledge_type: KnowledgeType = Field(description="知识类型")
    knowledge_version: str = Field(description="来源版本")
    source: str = Field(min_length=1, description="来源文档或著作")
    source_level: str = Field(description="知识来源等级")
    content: str = Field(min_length=1, description="证据内容")
    embedding_text: str = Field(min_length=1, description="优先使用 v2 审核文本")
    channels: list[str] = Field(default_factory=list, description="当前来源层级的六经标签")
    metadata: dict[str, object] = Field(description="原始来源字段，canonical 与 case-derived 分开")


class KnowledgeSearchRequest(BaseModel):
    """知识搜索参数，禁止未知写字段，空结果不以模型补齐。"""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000, description="知识检索文本")
    channels: list[str] | None = Field(default=None, description="六经元数据过滤")
    association_scope: list[str] | None = Field(default=None, description="方证来源范围")
    top_k: int = Field(default=8, ge=1, le=50, description="返回数量")


class DifferentialSearchRequest(BaseModel):
    """鉴别候选对查询，不允许模型生成规则替代原始来源。"""

    model_config = ConfigDict(extra="forbid")
    candidates: list[str] = Field(min_length=1, description="当前候选名称")
    query: str | None = Field(default=None, max_length=4000, description="附加检索文本")
    top_k: int = Field(default=5, ge=1, le=50, description="返回数量")


class CoreSearchRequest(KnowledgeSearchRequest):
    """框架查询请求，复用统一 query/top_k 校验。"""


class ClassicSearchRequest(KnowledgeSearchRequest):
    """经典条文查询请求，支持六经 metadata 过滤和精确条文号。"""

    query: str = Field(default="", max_length=4000, description="检索词；精确条文号查询可省略")
    clause_nos: list[str] | None = Field(default=None, min_length=1, description="精确条文号列表")

    @model_validator(mode="after")
    def validate_query_or_clause(self) -> Self:
        """精确查询允许无检索词，但不能接收没有目标或空条文号的请求。"""
        if self.clause_nos is not None:
            if any(not number.strip() for number in self.clause_nos):
                raise ValueError("条文号不能为空")
        elif not self.query.strip():
            raise ValueError("必须提供检索词或精确条文号")
        return self


class FormulaSearchRequest(KnowledgeSearchRequest):
    """方证查询请求，显式区分 canonical 与历史医案派生范围。"""


class CaseSearchRequest(KnowledgeSearchRequest):
    """历史医案查询请求，仅作后段参考。"""


class IndexProfile(BaseModel):
    """不可变索引身份，查询模型或来源发生变化时禁止混用旧向量。"""

    model_config = ConfigDict(frozen=True)
    knowledge_version: str = Field(min_length=1, description="知识版本")
    embedding_provider: str = Field(min_length=1, description="向量协议类型")
    embedding_model: str = Field(min_length=1, description="模型名称")
    embedding_model_version: str = Field(min_length=1, description="模型修订版本")
    embedding_endpoint: str = Field(default="", description="向量服务身份，区分不同部署")
    dimension: int = Field(gt=0, description="向量维度")
    corpus_digest: str = Field(min_length=1, description="完整知识来源快照摘要")
    metric_type: Literal["COSINE", "IP", "L2"] = Field(
        default="COSINE", description="向量距离度量，绑定索引身份"
    )

    def fingerprint(self) -> str:
        """生成稳定集合后缀，来源或模型变更不会覆盖历史集合。"""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:20]


class VectorDocument(BaseModel):
    """待索引的真实知识及模型向量，不能作为临床推理状态。"""

    record: KnowledgeRecord = Field(description="经 KnowledgeService 校验的来源")
    vector: list[float] = Field(description="模型生成的真实向量")


class IndexReport(BaseModel):
    """完整索引发布结果，失败时不会创建成功报告。"""

    backend: str = Field(description="实际 Remote/Lite 后端")
    index_type: str = Field(description="实际向量索引策略")
    profile: IndexProfile = Field(description="模型与来源身份")
    counts: dict[str, int] = Field(description="核验后的五类文档数量")
    collections: dict[str, str] = Field(description="当前已发布物理集合")
    reused: bool = Field(description="是否复用完全相同的已发布快照")


class FormulaBranchUnit(BaseModel):
    """从真实父方证和单条经典派生的分支，不是新增知识类型或专家核准规则。"""

    model_config = ConfigDict(extra="forbid")
    branch_id: str = Field(min_length=1)
    formula_evidence_id: str = Field(min_length=1)
    clause_evidence_ids: list[str] = Field(min_length=1)
    formula_name: str = Field(min_length=1)
    canonical_channels: list[str] = Field(default_factory=list)
    positive_conditions: list[str] = Field(default_factory=list)
    negative_conditions: list[str] = Field(default_factory=list)
    review_status: Literal["expert_approved", "source_verified_pending_expert"]
    provenance: str = Field(min_length=1)
    knowledge_version: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    logical_scope: Literal["single_clause", "unresolved_multiformula"] = "single_clause"
