"""${message}。业务迁移与 checkpoint 分离，升级/回滚只作用于本版本定义的表。"""

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    """升级业务结构。"""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """回滚业务结构；删除数据只在显式执行 downgrade 时发生。"""
    ${downgrades if downgrades else "pass"}
