"""业务迁移离线回归：验证冻结脚本可执行、表与索引契约一致；不代替真实 PostgreSQL。"""

import io
from pathlib import Path

from alembic.config import Config

from alembic import command
from backend.app.persistence.models import Base


def test_migration_upgrade_and_downgrade_compile_offline() -> None:
    """生成完整升级/降级 SQL，避免只检查迁移文件存在。"""
    root = Path(__file__).resolve().parents[2]
    output = io.StringIO()
    config = Config(str(root / "alembic.ini"), output_buffer=output)
    command.upgrade(config, "head", sql=True)
    upgrade = output.getvalue()
    for table in Base.metadata.tables.values():
        assert f"CREATE TABLE {table.name} (" in upgrade
        for index in table.indexes:
            assert f"INDEX {index.name}" in upgrade
    assert "checkpoint" not in upgrade
    output.seek(0)
    output.truncate()
    command.downgrade(config, "head:base", sql=True)
    downgrade = output.getvalue()
    for name in Base.metadata.tables:
        assert f"DROP TABLE {name};" in downgrade
