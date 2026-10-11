"""业务 PostgreSQL 测试，明确 disabled 与不可用，避免误用 checkpoint 数据库。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import PersistenceError
from backend.app.persistence.database import open_database


@pytest.mark.asyncio
async def test_disabled_business_database() -> None:
    """关闭业务数据库时不创建 ORM Engine，不假装历史记录可用。"""
    with patch("backend.app.persistence.database.create_async_engine") as factory:
        runtime = await open_database(Settings())
        assert runtime.backend == "disabled"
        assert runtime.engine is None
        factory.assert_not_called()
        await runtime.close()


@pytest.mark.asyncio
async def test_unavailable_business_database_closes_engine() -> None:
    """连接失败必须关闭连接池并转换为安全持久化异常。"""
    engine = MagicMock()
    engine.dispose = AsyncMock()
    connection = MagicMock()
    connection.__aenter__ = AsyncMock(side_effect=ConnectionError("private-password"))
    connection.__aexit__ = AsyncMock(return_value=False)
    engine.connect.return_value = connection
    with patch("backend.app.persistence.database.create_async_engine", return_value=engine):
        with pytest.raises(PersistenceError) as caught:
            await open_database(
                Settings(database_enabled=True, database_url="postgresql+asyncpg://test")
            )
        engine.dispose.assert_awaited_once()
        assert "private-password" not in str(caught.value)
