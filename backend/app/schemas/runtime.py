"""可靠运行 DTO：保存业务任务、幂等收据和补偿意图，不代替图状态或医学 Schema。"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.schemas.events import SSEEvent


class RunJob(BaseModel):
    """一轮持久任务的输入与结果；由 Repository 管理 lease 和完成条件。"""

    run_id: UUID
    consultation_id: UUID
    turn: int
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    snapshot: ConsultationSnapshot
    evidence: ClinicalEvidence | None = None
    owner: str | None = None
    lease_until: datetime | None = None
    result: dict[str, object] | None = None
    cancel_requested: bool = False
    attempts: int = 0
    duration_ms: int | None = None


class GraphAction(BaseModel):
    """事务保存的人工动作补偿意图，业务快照是最终权威。"""

    action_id: UUID
    consultation_id: UUID
    target_turn: int
    thread_id: str
    command: dict[str, object]
    snapshot: ConsultationSnapshot
    completed: bool = False


class EventPage(BaseModel):
    """有序运行事件页，用于 SSE 与刷新后恢复进度。"""

    events: list[SSEEvent] = Field(default_factory=list)
