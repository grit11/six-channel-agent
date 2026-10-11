"""初始六类业务表，固定迁移定义不依赖后续变化的 ORM 元数据。"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001_consultation"
down_revision = None
branch_labels = None
depends_on = None


def _id() -> sa.Column:
    """每类业务记录的独立 UUID 主键。"""
    return sa.Column("id", sa.Uuid(), primary_key=True)


def _owner() -> sa.Column:
    """归属 business consultation，不引用图 checkpoint。"""
    return sa.Column("consultation_id", sa.Uuid(), sa.ForeignKey("consultation.id"), nullable=False)


def _turn() -> sa.Column:
    """乐观业务轮次，单次输入可有 running 与结果两个快照版本。"""
    return sa.Column("turn", sa.Integer(), nullable=False)


def _json(name: str) -> sa.Column:
    """实体自己的结构化内容，不把全部领域数据混成 checkpoint。"""
    return sa.Column(name, postgresql.JSONB(), nullable=False)


def _time(name: str, nullable: bool = False) -> sa.Column:
    """时区感知的 UTC 业务时间列。"""
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    """创建 consultation 与五类追加业务实体，不调用 create_all。"""
    op.create_table(
        "consultation",
        _id(),
        sa.Column("thread_id", sa.String(128), unique=True, nullable=False),
        sa.Column("doctor_id", sa.String(128)),
        sa.Column("patient_ref", sa.String(256)),
        sa.Column("status", sa.String(40), nullable=False),
        _turn(),
        _json("snapshot"),
        sa.Column("protocol_version", sa.String(80), nullable=False),
        sa.Column("kb_version", sa.String(80), nullable=False),
        _time("created_at"),
        _time("updated_at"),
        _time("completed_at", True),
    )
    op.create_index("ix_consultation_doctor_id", "consultation", ["doctor_id"])
    op.create_index("ix_consultation_status", "consultation", ["status"])
    op.create_table(
        "consultation_turn",
        _id(),
        _owner(),
        _turn(),
        sa.Column("status", sa.String(40), nullable=False),
        _json("snapshot"),
        _time("created_at"),
        sa.UniqueConstraint("consultation_id", "turn"),
    )
    op.create_table(
        "reasoning_path",
        _id(),
        _owner(),
        _turn(),
        sa.Column("path_id", sa.String(128), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        _json("payload"),
        sa.UniqueConstraint("consultation_id", "turn", "path_id"),
    )
    op.create_table(
        "doctor_review",
        _id(),
        _owner(),
        _turn(),
        sa.Column("action", sa.String(32), nullable=False),
        _json("payload"),
        _time("created_at"),
    )
    op.create_table(
        "evidence_trace",
        _id(),
        _owner(),
        _turn(),
        sa.Column("evidence_id", sa.String(200), nullable=False),
        _json("payload"),
        sa.UniqueConstraint("consultation_id", "turn", "evidence_id"),
    )
    op.create_table(
        "audit_event",
        _id(),
        _owner(),
        _turn(),
        sa.Column("event", sa.String(80), nullable=False),
        _json("payload"),
        _time("created_at"),
    )
    for table in ("consultation_turn", "reasoning_path", "doctor_review", "evidence_trace"):
        op.create_index("ix_" + table + "_consultation_id", table, ["consultation_id"])
    op.create_index("ix_audit_consultation_turn", "audit_event", ["consultation_id", "turn"])


def downgrade() -> None:
    """显式回滚将删除本版本业务数据；不删除任何 checkpoint 表。"""
    for table in (
        "audit_event",
        "evidence_trace",
        "doctor_review",
        "reasoning_path",
        "consultation_turn",
        "consultation",
    ):
        op.drop_table(table)
