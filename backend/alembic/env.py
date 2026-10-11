"""业务 Alembic 环境：使用集中配置连接 PostgreSQL，支持离线 SQL 检查。

不创建 checkpoint 表，不在应用启动隐式迁移，不输出连接串或数据库密码。
"""

import asyncio

from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from alembic import context
from backend.app.core.config import Settings
from backend.app.persistence.models import Base


def run_offline() -> None:
    """以固定 PostgreSQL 方言生成迁移 SQL，无需读取生产密钥。"""
    context.configure(
        dialect_name="postgresql",
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def migrate(connection: Connection) -> None:
    """在已打开连接中执行迁移，事务失败由 Alembic 回滚。"""
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    """真实迁移只允许 asyncpg PostgreSQL，连接池在任何结果下关闭。"""
    settings = Settings()
    url = settings.database_url.get_secret_value()
    if not url.startswith("postgresql+asyncpg://"):
        raise ValueError("Alembic 需要配置 DATABASE_URL=postgresql+asyncpg://...")
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    asyncio.run(run_online())
