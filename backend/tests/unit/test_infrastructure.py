"""基础设施策略单测：用明确的测试替身覆盖远程与降级分支。"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.persistence.checkpoint import (
    CheckpointBackendStatus,
    CheckpointRuntime,
    build_checkpoint_runtime,
)
from backend.app.rag.vector_store import build_milvus_store


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "remote"])
async def test_remote_milvus_is_probed(mode: str) -> None:
    """Remote 对象构造成功后必须执行真实探测调用。"""
    client = MagicMock()
    with patch("backend.app.rag.vector_store.MilvusClient", return_value=client):
        store = await build_milvus_store(Settings(milvus_mode=mode))
        assert store.status.backend == "remote"
        client.list_collections.assert_called_once()
        await store.close()
        client.close.assert_called_once()


@pytest.mark.asyncio
async def test_auto_milvus_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Remote 失败时使用 Lite 同一客户端入口，日志不泄露 Token。"""
    lite_client = MagicMock()
    settings = Settings(
        milvus_token="secret-token",
        milvus_lite_path=str(tmp_path / "nested" / "milvus.db"),
    )
    with patch(
        "backend.app.rag.vector_store.MilvusClient",
        side_effect=[RuntimeError("secret-token"), lite_client],
    ) as factory:
        store = await build_milvus_store(settings)
        assert store.status.backend == "lite"
        assert store.status.degraded
        assert Path(settings.milvus_lite_path).parent.exists()
        assert factory.call_args.kwargs["uri"] == settings.milvus_lite_path
        assert "secret-token" not in caplog.text
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,fallback", [("remote", True), ("auto", False)])
async def test_milvus_disallowed_fallback(mode: str, fallback: bool) -> None:
    """强制 Remote 或关闭 fallback 时不得尝试 Lite。"""
    with patch(
        "backend.app.rag.vector_store.MilvusClient", side_effect=RuntimeError("failure")
    ) as factory:
        with pytest.raises(ProviderUnavailableError):
            await build_milvus_store(Settings(milvus_mode=mode, milvus_fallback_to_lite=fallback))
        assert factory.call_count == 1


@pytest.mark.asyncio
async def test_lite_failure_is_explicit(tmp_path: Path) -> None:
    """Lite 本身不可用时必须明确失败，不虚报 degraded 可用状态。"""
    with patch("backend.app.rag.vector_store.MilvusClient", side_effect=ImportError("missing")):
        with pytest.raises(ProviderUnavailableError, match="Milvus Lite") as caught:
            await build_milvus_store(
                Settings(milvus_mode="lite", milvus_lite_path=str(tmp_path / "milvus.db"))
            )
        assert isinstance(caught.value.__cause__, ImportError)


@pytest.mark.asyncio
async def test_postgres_checkpoint_is_selected() -> None:
    """可用 PostgreSQL saver 必须优先于 SQLite。"""
    runtime = MagicMock(spec=CheckpointRuntime)
    runtime.status = CheckpointBackendStatus("postgres", False, "正常")
    with patch("backend.app.persistence.checkpoint._open_postgres", return_value=runtime):
        result = await build_checkpoint_runtime(
            Settings(langgraph_database_url="postgresql://test")
        )
        assert result.status.backend == "postgres"


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_checkpoint_auto_fallback(tmp_path: Path, configured: bool) -> None:
    """PostgreSQL 缺少配置或连接失败，按配置打开真实 SQLite。"""
    settings = Settings(
        langgraph_database_url="postgresql://secret" if configured else "",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.sqlite3"),
    )
    with patch(
        "backend.app.persistence.checkpoint._open_postgres",
        side_effect=ConnectionError("secret"),
    ):
        runtime = await build_checkpoint_runtime(settings)
        assert runtime.status.backend == "sqlite"
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,fallback", [("postgres", True), ("auto", False)])
async def test_checkpoint_disallowed_fallback(mode: str, fallback: bool) -> None:
    """强制 PostgreSQL 或禁用降级时必须显式失败。"""
    with pytest.raises(ProviderUnavailableError):
        await build_checkpoint_runtime(
            Settings(
                checkpoint_backend=mode,
                checkpoint_fallback_to_sqlite=fallback,
                langgraph_database_url="postgresql://test",
            )
        )


@pytest.mark.asyncio
async def test_postgres_context_cleanup_on_failure() -> None:
    """PostgreSQL setup 失败时关闭 saver 上下文后才允许降级。"""
    context = MagicMock()
    saver = MagicMock()
    saver.setup = AsyncMock(side_effect=ConnectionError("connection failed"))
    context.__aenter__ = AsyncMock(return_value=saver)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch(
        "backend.app.persistence.checkpoint.AsyncPostgresSaver.from_conn_string",
        return_value=context,
    ):
        with pytest.raises(ProviderUnavailableError):
            await build_checkpoint_runtime(
                Settings(
                    checkpoint_backend="postgres",
                    langgraph_database_url="postgresql://test",
                )
            )
        context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_postgres_saver_setup_and_probe() -> None:
    """可用 saver 要执行 setup 和探测，并在关闭 runtime 时释放上下文。"""
    context = MagicMock()
    saver = MagicMock()
    saver.setup = AsyncMock()
    saver.aget_tuple = AsyncMock(return_value=None)
    context.__aenter__ = AsyncMock(return_value=saver)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch(
        "backend.app.persistence.checkpoint.AsyncPostgresSaver.from_conn_string",
        return_value=context,
    ):
        runtime = await build_checkpoint_runtime(
            Settings(
                checkpoint_backend="postgres",
                langgraph_database_url="postgresql://test",
            )
        )
        assert runtime.status.backend == "postgres"
        saver.setup.assert_awaited_once()
        saver.aget_tuple.assert_awaited_once()
        await runtime.close()
        context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_container_cleanup_after_checkpoint_failure() -> None:
    """后续资源创建失败时也关闭已连接的向量资源。"""
    from backend.app.core.dependencies import open_container

    store = MagicMock()
    store.close = AsyncMock()
    with (
        patch("backend.app.core.dependencies.build_milvus_store", return_value=store),
        patch(
            "backend.app.core.dependencies.build_checkpoint_runtime",
            side_effect=ProviderUnavailableError("checkpoint 失败"),
        ),
        pytest.raises(ProviderUnavailableError),
    ):
        async with open_container(Settings()):
            pytest.fail("失败的容器不应进入服务阶段")
    store.close.assert_awaited_once()
