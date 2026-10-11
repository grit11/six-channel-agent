"""PostgreSQL 运行适配器 SQL 回归，验证绑定参数/事务序列，实机验收后置。"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.exceptions import PersistenceError
from backend.app.persistence.runtime import PostgresRuntimeStore


@pytest.mark.asyncio
async def test_event_head_and_payload_share_transaction() -> None:
    """递增 head 与事件使用同一 session；只写公开载荷，SQL 无原文插值。"""
    session = MagicMock()
    session.scalar = AsyncMock(return_value={"sequence": 42})
    session.execute = AsyncMock()
    identifier = uuid4()
    await PostgresRuntimeStore.append_event(
        session, identifier, 2, "stage_started", {"stage": "eight_principles"}
    )
    head = session.scalar.call_args.args[0].compile(dialect=postgresql.dialect())
    event = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert "ON CONFLICT" in str(head) and "RETURNING" in str(head)
    assert "jsonb_build_object" in str(head) and "sequence" in head.params.values()
    payload = next(value for value in event.params.values() if isinstance(value, dict))
    assert payload["event_id"] == f"{identifier}:42" and payload["sequence"] == 42


@pytest.mark.asyncio
async def test_mutex_uses_bound_signed_key_and_releases_transaction() -> None:
    """图锁为 PostgreSQL 事务 advisory lock，等待者不长期占用连接池。"""
    store = PostgresRuntimeStore(MagicMock())
    session = MagicMock()
    session.scalar = AsyncMock(return_value=True)
    scope = MagicMock()
    scope.__aenter__ = AsyncMock(return_value=session)
    scope.__aexit__ = AsyncMock(return_value=False)
    store.lock_sessions = MagicMock()
    store.lock_sessions.begin.return_value = scope
    async with store.mutex("graph:测试线程"):
        statement, params = session.scalar.call_args.args
        assert "pg_try_advisory_xact_lock(:key)" in str(statement)
        assert isinstance(params["key"], int)
        assert -(2**63) <= params["key"] < 2**63
    scope.__aexit__.assert_awaited_once()
    session.scalar.side_effect = SQLAlchemyError("敏感连接")
    with pytest.raises(PersistenceError, match="运行协调"):
        async with store.mutex("graph:测试线程"):
            raise AssertionError("不可进入")


@pytest.mark.asyncio
async def test_recovery_scan_is_bounded_and_filters_expired_not_terminal() -> None:
    """队列不扫描终态历史；JSON 内部租约按 UTC 转换，查询使用绑定值。"""
    store = PostgresRuntimeStore(MagicMock())
    session = MagicMock()
    session.scalars = AsyncMock(return_value=[])
    scope = MagicMock()
    scope.__aenter__ = AsyncMock(return_value=session)
    scope.__aexit__ = AsyncMock(return_value=False)
    store.sessions = MagicMock(return_value=scope)
    await store.scan("job")
    query = session.scalars.call_args.args[0].compile(dialect=postgresql.dialect())
    assert "LIMIT" in str(query) and 256 in query.params.values()
    assert ["pending", "running"] in query.params.values()
    assert "TIMESTAMP WITH TIME ZONE" in str(query)
    identifier = uuid4()
    await store.scan("action", identifier)
    query = session.scalars.call_args.args[0].compile(dialect=postgresql.dialect())
    assert str(identifier) in query.params.values() and "false" in query.params.values()
