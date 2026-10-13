"""V2 会诊业务契约：严格保存公开事实、证据和医生审核，不接受旧规则结果。

Repository 是业务快照权威；Checkpoint 只承担图恢复，二者通过持久任务协调。
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ComparisonV2,
    RetrievalBundleV2,
)
from backend.app.schemas.review import DoctorReviewRecord

ConsultationStatus = Literal[
    "created",
    "running",
    "needs_evidence",
    "awaiting_doctor_review",
    "undetermined",
    "completed",
    "failed",
    "cancelled",
]


class PatientProfile(BaseModel):
    """医生主动填写的基本资料，不参与程序医学分类。"""

    model_config = ConfigDict(extra="forbid")
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str = "unknown"


class ConsultationSnapshot(BaseModel):
    """唯一严格 V2 业务快照，缺少中间阶段结果是合法状态。"""

    model_config = ConfigDict(extra="forbid")
    consultation_id: UUID
    thread_id: str
    doctor_id: str | None = None
    patient_ref: str | None = None
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    status: ConsultationStatus
    turn: int = Field(ge=1)
    revision: int = Field(default=0, ge=0)
    raw_input: str = Field(min_length=1, max_length=20000)
    paused_at: Literal["s2_clarification", "s5_doctor_review"] | None = None
    clinical_evidence: ClinicalEvidenceV2 | None = None
    localization: ChannelLocalizationV2 | None = None
    retrieval: RetrievalBundleV2 | None = None
    comparison: ComparisonV2 | None = None
    doctor_review: dict[str, object] | None = None
    review_history: list[DoctorReviewRecord] = Field(default_factory=list)
    protocol_version: Literal["clinical-v2.0"] = "clinical-v2.0"
    knowledge_base_version: str = "1.2"
    versions: dict[str, str] = Field(default_factory=dict)
    completed_stages: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    last_run_duration_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


ConsultationSnapshotV2 = ConsultationSnapshot


class ConsultationHistoryQuery(BaseModel):
    """历史查询只支持显式状态、业务引用和有界分页。"""

    model_config = ConfigDict(extra="forbid")
    status: ConsultationStatus | None = None
    patient_ref: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=50)


class ConsultationHistoryItem(BaseModel):
    """不含患者原文的历史摘要。"""

    consultation_id: UUID
    patient_ref: str | None = None
    status: ConsultationStatus
    turn: int
    topk_syndromes: list[str] = Field(default_factory=list)
    doctor_review: dict[str, object] | None = None
    knowledge_base_version: str
    created_at: datetime
    updated_at: datetime
    last_run_duration_ms: int | None = None

    @classmethod
    def from_snapshot(cls, snapshot: ConsultationSnapshot) -> "ConsultationHistoryItem":
        """只投影公开摘要，不能从列表推断医生最终诊断。"""
        return cls(
            consultation_id=snapshot.consultation_id,
            patient_ref=snapshot.patient_ref,
            status=snapshot.status,
            turn=snapshot.turn,
            topk_syndromes=(
                [path.formula_name or "未决" for path in snapshot.comparison.ordered_paths]
                if snapshot.comparison
                else []
            ),
            doctor_review=snapshot.doctor_review,
            knowledge_base_version=snapshot.knowledge_base_version,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            last_run_duration_ms=snapshot.last_run_duration_ms,
        )


class ConsultationHistoryPage(BaseModel):
    """分页业务历史。"""

    items: list[ConsultationHistoryItem]
    total: int
    page: int
    page_size: int
