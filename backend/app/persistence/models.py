"""PostgreSQL 业务 ORM：保存会话、轮次、路径、审核、来源追踪与审计。

边界：Schema 与 ORM 分离，图 checkpoint 由官方 saver 独立管理。
表结构只通过 Alembic 创建，应用启动不隐式建表；JSON 仅承载对应实体的结构化数据。
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """业务迁移统一元数据，与 checkpoint 表互不混用。"""


class ConsultationModel(Base):
    """当前权威会话快照与乐观更新版本。"""

    __tablename__ = "consultation"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True)
    doctor_id: Mapped[str | None] = mapped_column(String(128), index=True)
    patient_ref: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(40), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict[str, object]] = mapped_column(JSONB)
    protocol_version: Mapped[str] = mapped_column(String(80))
    kb_version: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConsultationTurnModel(Base):
    """不可变业务轮次快照，避免仅保留最后一次输入。"""

    __tablename__ = "consultation_turn"
    __table_args__ = (UniqueConstraint("consultation_id", "turn"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultation.id"), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40))
    snapshot: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ReasoningPathModel(Base):
    """每次输出的原始 Agent 路径，不被医生修改覆盖。"""

    __tablename__ = "reasoning_path"
    __table_args__ = (UniqueConstraint("consultation_id", "turn", "path_id"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultation.id"), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    path_id: Mapped[str] = mapped_column(String(128))
    rank: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)


class DoctorReviewModel(Base):
    """追加人工动作日志，记录 Agent 建议与医生采用的差异。"""

    __tablename__ = "doctor_review"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultation.id"), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EvidenceTraceModel(Base):
    """保存输出引用的精确来源和检索分，后续工具审计可追加查询参数。"""

    __tablename__ = "evidence_trace"
    __table_args__ = (UniqueConstraint("consultation_id", "turn", "evidence_id"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultation.id"), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    evidence_id: Mapped[str] = mapped_column(String(200))
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)


class AuditEventModel(Base):
    """不含病例全文的关键业务状态事件，不保存模型隐藏推理。"""

    __tablename__ = "audit_event"
    __table_args__ = (Index("ix_audit_consultation_turn", "consultation_id", "turn"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    consultation_id: Mapped[UUID] = mapped_column(ForeignKey("consultation.id"))
    turn: Mapped[int] = mapped_column(Integer)
    event: Mapped[str] = mapped_column(String(80))
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RuntimeRecordModel(Base):
    """独立可靠运行记录：event/job/receipt/action，不能替代 consultation 或 checkpoint。"""

    __tablename__ = "runtime_record"
    kind: Mapped[str] = mapped_column(String(32), primary_key=True)
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB)
