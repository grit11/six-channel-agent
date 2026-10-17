"""新增 QA 独立历史表，不触碰临床业务表或 LangGraph checkpoint。"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003_qa_history"
down_revision = "0002_reliable_runtime"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """按既有迁移链追加 QA 会话和消息，唯一约束防止并发双写。"""
    op.create_table(
        "qa_sessions",
        sa.Column("session_id", sa.Uuid(), primary_key=True),
        sa.Column("owner_id", sa.String(256), nullable=False),
        sa.Column("title", sa.String(160), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    for name in ("owner_id", "status", "updated_at"):
        op.create_index(f"ix_qa_sessions_{name}", "qa_sessions", [name])
    op.create_table(
        "qa_messages",
        sa.Column("message_id", sa.Uuid(), primary_key=True),
        sa.Column("session_id", sa.Uuid(), sa.ForeignKey("qa_sessions.session_id"), nullable=False),
        sa.Column("client_message_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("session_id", "sequence"),
        sa.UniqueConstraint("session_id", "client_message_id", "role"),
    )
    op.create_index("ix_qa_messages_session_sequence", "qa_messages", ["session_id", "sequence"])


def downgrade() -> None:
    """只移除 QA 新增表，保留临床和 checkpoint。"""
    op.drop_table("qa_messages")
    op.drop_table("qa_sessions")
