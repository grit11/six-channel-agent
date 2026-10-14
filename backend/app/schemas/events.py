"""系统健康与版本 DTO，使用中文字段说明并与 ORM/基础设施对象分离。"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """实际基础设施状态，不包含 URL、凭据和异常堆栈。"""

    status: Literal["ok", "degraded"] = Field(description="服务正常或已降级")
    vector_backend: Literal["remote", "lite"] = Field(description="实际向量后端")
    checkpoint_backend: Literal["postgres", "sqlite"] = Field(description="实际图状态后端")
    business_database: Literal["postgres", "disabled"] = Field(description="业务数据库状态")


class VersionsResponse(BaseModel):
    """可复现性配置快照，未实现或未配置的能力不会伪造版本。"""

    protocol_version: str = Field(description="推理协议版本")
    knowledge_base_version: str = Field(description="知识库版本")
    scoring_config_version: str = Field(description="V2 中固定为 removed，保留字段仅用于运维兼容")
    llm_model: str = Field(description="已配置大模型名称")
    embedding_model: str = Field(description="已配置向量模型名称")
    reranker_model: str = Field(description="已配置重排模型名称")
    clinical_parser_mode: Literal["llm"] = Field(description="V2 S1 仅使用结构化 LLM")
    reasoning_mode: Literal["llm"] = Field(description="V2 S2-S4 仅使用 LLM + RAG")
    backend_build_version: str = Field(description="后端工程版本")
    prompt_version: str = Field(description="提示词版本或未实现状态")
    source_condition_version: str = Field(
        default="source-condition/v2", description="来源条件规范化版本"
    )
    evidence_grader_version: str = Field(default="grounded-v2", description="来源适用性分级版本")
    path_summary_version: str = Field(default="path-summary/v1.0", description="路径来源摘要版本")
    path_completeness_version: str = Field(
        default="path-completeness/v1.0", description="路径条件覆盖算法版本"
    )
    clinical_context_version: str = Field(
        default="clinical-context/v2.0", description="病程与治疗事实契约版本"
    )
    relationship_version: str = Field(
        default="relationship/v2.0", description="关系时间事实核验版本"
    )
    similar_case_version: str = "similar-case/v1.0"
    stage_replay_version: str = "stage-replay/v1.0"
    retrospective_replay_version: str = "retrospective-replay/v1.0"
    knowledge_qa_mode: Literal["extractive", "llm"] = Field(
        default="extractive", description="独立知识问答运行模式"
    )
    knowledge_qa_version: str = Field(default="knowledge-qa/v1.0", description="独立 QA 协议版本")


StreamEventName = Literal[
    "snapshot_updated",
    "clarification_requested",
    "review_requested",
    "consultation_completed",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "retrieval_started",
    "retrieval_completed",
    "model_started",
    "model_completed",
    "model_failed",
    "s4_stream_started",
    "s4_stream_progress",
    "s4_stream_failed",
    "s4_retrieval_fallback_ready",
    "s4_retry_started",
    "recovery_required",
    "needs_evidence",
    "topk_updated",
    "review_required",
    "completed",
    "failed",
    "heartbeat",
]


class SSEEvent(BaseModel):
    """可重连的增量事件；完整事实仍必须从 consultation snapshot 获取。"""

    event_id: str = Field(min_length=1)
    consultation_id: UUID
    event: StreamEventName
    turn: int = Field(ge=1)
    sequence: int = Field(default=0, ge=0)
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
