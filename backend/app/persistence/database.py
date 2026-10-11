"""业务数据库生命周期适配，只连接 PostgreSQL，不复用 checkpoint SQLite 表。"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from backend.app.core.config import Settings
from backend.app.core.exceptions import PersistenceError


@dataclass
class DatabaseRuntime:
    """业务数据库连接及健康状态，disabled 表示功能被显式关闭。"""

    backend: Literal["postgres", "disabled"]
    engine: AsyncEngine | None = None
    coordination_engine: AsyncEngine | None = None

    async def close(self) -> None:
        """释放业务数据库连接池。"""
        if self.engine is not None:
            await self.engine.dispose()
        if self.coordination_engine is not None:
            await self.coordination_engine.dispose()


async def open_database(settings: Settings) -> DatabaseRuntime:
    """按功能开关连接业务 PostgreSQL，不可用时明确启动失败。"""
    if not settings.database_enabled:
        return DatabaseRuntime("disabled")
    engine = create_async_engine(
        settings.database_url.get_secret_value(),
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_recycle=settings.database_pool_recycle_seconds,
    )
    try:
        async with asyncio.timeout(settings.postgres_connect_timeout_seconds):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        raise PersistenceError("业务 PostgreSQL 不可用") from exc
    # 长时图/请求互斥不能占满业务连接池，否则持锁者无法获得提交连接形成死锁。
    # 锁连接独立且不复用；争用通过 try-lock 释放等待连接，业务连接仍按配置有界。
    coordination = create_async_engine(
        settings.database_url.get_secret_value(),
        poolclass=NullPool,
        connect_args={"timeout": settings.postgres_connect_timeout_seconds},
    )
    return DatabaseRuntime("postgres", engine, coordination)
