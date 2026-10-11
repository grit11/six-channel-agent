"""医生审核契约：保存人工动作与 Agent 原建议的差异，不生成处方。"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ReviewAction = Literal["accept", "modify", "reject", "reorder", "add_evidence", "note"]


class DoctorReviewRequest(BaseModel):
    """医生审核动作；选择的路径必须由服务端在当前快照中核验。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: ReviewAction
    selected_path_id: str | None = None
    ordered_path_ids: list[str] = Field(default_factory=list)
    modified_six_channels: list[str] = Field(default_factory=list)
    modified_syndrome: str | None = Field(default=None, max_length=500)
    modified_formula_pattern: str | None = Field(default=None, max_length=200)
    note: str | None = Field(default=None, max_length=5_000)
    expected_turn: int | None = Field(default=None, ge=1)
    rerun: bool = False


class DoctorReviewRecord(BaseModel):
    """已写入快照的审核记录，区分 Agent 原建议与医生最终动作。"""

    model_config = ConfigDict(extra="forbid")
    review_id: UUID
    consultation_id: UUID
    action: ReviewAction
    selected_path_id: str | None = None
    ordered_path_ids: list[str] = Field(default_factory=list)
    modified_six_channels: list[str] = Field(default_factory=list)
    modified_syndrome: str | None = None
    modified_formula_pattern: str | None = None
    note: str | None = None
    agent_path_ids: list[str] = Field(default_factory=list)
    doctor_adopted: bool = False
    created_at: datetime
