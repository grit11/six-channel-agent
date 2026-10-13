"""检索结果数据契约，来源信息与召回、重排分数分开保存。"""

from typing import Literal

from pydantic import BaseModel, Field

EvidenceGrade = Literal["direct", "indirect", "contradict", "background", "irrelevant"]


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


class EvidenceBundle(BaseModel):
    """分层 Evidence 集合，医案不得覆盖经典。"""

    framework: list[Evidence] = Field(description="核心框架")
    classic: list[Evidence] = Field(description="经典原文")
    formulas: list[Evidence] = Field(description="结构化方证")
    differential: list[Evidence] = Field(description="鉴别来源")
    cases: list[Evidence] = Field(description="历史医案")
