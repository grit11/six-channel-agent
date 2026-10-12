"""医生辨证会话契约：定义业务状态、快照版本与补充输入，不承载医学计算。"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.reasoning import EvidenceGapContext, ReasoningStageSnapshot
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.schemas.review import DoctorReviewRecord

ReasoningStage = Literal[
    "clinical_parser",
    "six_channel_screen",
    "formula_branch_retrieval",
    "differential_verification",
    "evidence_topk",
]
WorkflowStage = Literal[
    "clinical_parser",
    "six_channel_screen",
    "formula_branch_retrieval",
    "differential_verification",
    "evidence_topk",
    "doctor_review",
]


class PatientProfile(BaseModel):
    """可选患者基本资料；无姓名或证件字段，不作为模型推断的补全信息。"""

    model_config = ConfigDict(extra="forbid")
    age: int | None = Field(default=None, ge=0, le=130, description="医生填写年龄")
    sex: Literal["male", "female", "other", "unknown"] = Field(
        default="unknown", description="医生填写性别；未知保留"
    )


class FactGroup(BaseModel):
    """医生明确修订的同键同时间事实组；允许修订历史且保留其他时点。"""

    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1, description="事实键")
    time_scope: str | None = Field(default=None, description="原事实时间范围；空为当前")


ConsultationStatus = Literal[
    "created",
    "running",
    "needs_evidence",
    "awaiting_doctor_review",
    "completed",
    "safety_escalated",
    "failed",
    "cancelled",
]


class ConsultationCreateRequest(BaseModel):
    """创建会话请求；病例原文只保存于业务快照，不写入日志。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(
        min_length=1,
        max_length=20_000,
        description="医生输入的病例原文",
        validation_alias=AliasChoices("clinical_text", "text"),
    )
    clinical_evidence: ClinicalEvidence | None = Field(
        default=None, description="医生核对的事实快照"
    )
    doctor_id: str | None = Field(default=None, max_length=128, description="医生身份标识")
    patient_ref: str | None = Field(default=None, max_length=256, description="脱敏患者引用")
    patient_profile: PatientProfile = Field(
        default_factory=PatientProfile, description="可选基本资料"
    )
    top_k: int = Field(default=3, ge=1, le=5, description="希望查看的路径数量")


class ConsultationEvidenceRequest(BaseModel):
    """补充证据请求；可提交自然语言、结构化快照或两者之一。"""

    model_config = ConfigDict(extra="forbid")
    clinical_text: str | None = Field(default=None, max_length=20_000)
    clinical_evidence: ClinicalEvidence | None = None
    expected_turn: int | None = Field(default=None, ge=1)
    resume: bool = Field(default=True, description="是否继续同一 thread 的推理")
    replace_keys: list[str] = Field(default_factory=list, description="医生明确修订的当前事实键")
    replace_fact_groups: list[FactGroup] = Field(
        default_factory=list, max_length=200, description="明确整体替换的事实键/时点组，允许删除"
    )
    unavailable_question_ids: list[str] = Field(default_factory=list, max_length=100)
    available_question_ids: list[str] = Field(default_factory=list, max_length=100)
    replace_context_fields: list[
        Literal["timeline", "treatment_events", "onset", "course", "relationship_history"]
    ] = Field(
        default_factory=list,
        description="医生明确整体修订的病程字段，允许清空；不删除旧审计快照",
    )


class ConsultationRerunRequest(BaseModel):
    """重跑请求；expected_turn 用于避免旧页面覆盖新快照。"""

    model_config = ConfigDict(extra="forbid")
    expected_turn: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1, le=5)
    from_stage: ReasoningStage = Field(
        default="clinical_parser",
        description="从指定阶段重算；仅可复用相同事实/配置的已完成前序结果",
    )


class ConsultationSnapshot(BaseModel):
    """可刷新恢复的权威会话快照，SSE 只能作为增量提示。"""

    model_config = ConfigDict(extra="forbid")
    consultation_id: UUID
    thread_id: str
    doctor_id: str | None = None
    patient_ref: str | None = None
    patient_profile: PatientProfile = Field(
        default_factory=PatientProfile, description="医生填写的基本资料"
    )
    status: ConsultationStatus
    turn: int = Field(ge=1)
    top_k: int = Field(default=3, ge=1, le=5)
    supplementation_round: int = Field(default=0, ge=0)
    evidence_gap_context: EvidenceGapContext = Field(default_factory=EvidenceGapContext)
    excluded_path_ids: list[str] = Field(default_factory=list, description="医生明确驳回的当前路径")
    review_order: list[str] = Field(
        default_factory=list, description="医生查看顺序，不覆盖 Agent 原排名"
    )
    raw_input: str = Field(min_length=1)
    clinical_evidence: ClinicalEvidence | None = None
    reasoning: ReasoningResponse | None = None
    doctor_review: DoctorReviewRecord | None = None
    review_history: list[DoctorReviewRecord] = Field(default_factory=list)
    protocol_version: str
    knowledge_base_version: str
    scoring_config_version: str
    versions: dict[str, str] = Field(
        default_factory=dict, description="此次会话使用的模型与构建版本"
    )
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    last_run_duration_ms: int | None = Field(
        default=None, ge=0, description="最近一次实际推理耗时，不含医生等待时间"
    )
    rerun_from_stage: str = Field(default="clinical_parser", description="本轮开始重算阶段")


class ConsultationHistoryQuery(BaseModel):
    """历史快照分页过滤参数，不在搜索中包含病例全文。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: ConsultationStatus | None = Field(default=None, description="按状态过滤")
    patient_ref: str | None = Field(default=None, max_length=256, description="完整脱敏患者引用")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=10, ge=1, le=50, description="每页记录数")


class ConsultationHistoryItem(BaseModel):
    """历史列表摘要；不返回病例全文或完整患者事实，详情另行精确读取。"""

    consultation_id: UUID = Field(description="会诊 ID")
    patient_ref: str | None = Field(description="脱敏引用")
    status: ConsultationStatus = Field(description="会诊状态")
    turn: int = Field(description="当前版本")
    topk_syndromes: list[str] = Field(description="Agent 原始 Top-K 摘要")
    doctor_review: DoctorReviewRecord | None = Field(
        description="医生最后审核动作，独立于 Agent 结论"
    )
    knowledge_base_version: str = Field(description="当时知识版本")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    last_run_duration_ms: int | None = Field(description="实际推理耗时，旧记录可未知")

    @classmethod
    def from_snapshot(cls, snapshot: ConsultationSnapshot) -> "ConsultationHistoryItem":
        """只投影公开列表所需字段，不把病例全文嵌入分页响应。"""
        return cls(
            consultation_id=snapshot.consultation_id,
            patient_ref=snapshot.patient_ref,
            status=snapshot.status,
            turn=snapshot.turn,
            topk_syndromes=(
                [path.syndrome_name for path in snapshot.reasoning.reasoning_paths]
                if snapshot.reasoning
                else []
            ),
            doctor_review=snapshot.doctor_review,
            knowledge_base_version=snapshot.knowledge_base_version,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            last_run_duration_ms=snapshot.last_run_duration_ms,
        )


class ConsultationHistoryPage(BaseModel):
    """权威业务历史分页，数据来自 Repository，不从 checkpoint 拼接。"""

    items: list[ConsultationHistoryItem] = Field(description="当前页摘要")
    total: int = Field(ge=0, description="符合过滤条件的记录总数")
    page: int = Field(description="页码")
    page_size: int = Field(description="每页记录数")


class ConsultationResponse(BaseModel):
    """创建或更新会话统一返回当前快照。"""

    model_config = ConfigDict(extra="forbid")
    snapshot: ConsultationSnapshot


class ConsultationReplay(BaseModel):
    """当前业务轮次公开阶段回放；旧快照不重算或补造，不包含隐藏推理。"""

    consultation_id: UUID
    thread_id: str
    turn: int
    status: Literal["available", "unavailable"]
    stages: list[ReasoningStageSnapshot] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    reason: str
    reasoning_run_turn: int | None = Field(
        default=None, ge=1, description="最近提交推理任务的业务轮次，结果发布后仍可对应实时事件"
    )
