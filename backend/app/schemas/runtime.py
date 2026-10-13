"""持久任务契约：描述运行意图、租约和发布收据，不承载隐藏推理。"""

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from backend.app.clinical_v2.schemas import HumanGraphCommandV2


class RunJob(BaseModel):
    """与业务快照同事务保存；结果先落运行存储再发布业务快照。"""

    model_config = ConfigDict(extra="forbid")
    run_id: UUID = Field(default_factory=uuid4)
    consultation_id: UUID
    turn: int = Field(ge=1)
    command: HumanGraphCommandV2 | None = None
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    attempts: int = Field(default=0, ge=0)
    owner: str | None = None
    lease_until: datetime | None = None
    result: dict[str, object] | None = None
    duration_ms: int | None = None
