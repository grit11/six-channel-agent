"""S5 独立运行记录：持久事件、幂等收据、运行租约与补偿意图；不触碰 checkpoint。"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002_reliable_runtime"
down_revision = "0001_consultation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """创建运行表，业务 Repository 与其在同一 PostgreSQL 事务中提交。"""
    op.create_table(
        "runtime_record",
        sa.Column("kind", sa.String(32), primary_key=True),
        sa.Column("key", sa.String(256), primary_key=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )


def downgrade() -> None:
    """显式删除运行记录，不删除旧业务表或图状态。"""
    op.drop_table("runtime_record")
