"""Milvus 生命周期回归：验证保活、专用通道、路径引用与取消时的关闭顺序。

使用同步 SDK 替身验证真实并发边界；不调用在线模型、不改临床图或知识数据。
"""

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.rag.vector_store import (
    MilvusVectorStore,
    VectorBackendStatus,
    build_milvus_store,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["remote", "lite"])
async def test_connection_keepalive_and_dedicated_close(mode: str, tmp_path: Path) -> None:
    """两种后端都禁止空闲 ping，close 必须真正关闭专用通道且可重复调用。"""
    client = MagicMock()
    settings = Settings(
        _env_file=None, milvus_mode=mode, milvus_lite_path=str(tmp_path / "index.db")
    )
    with (
        patch("backend.app.rag.vector_store.MilvusClient", return_value=client) as factory,
        patch("backend.app.rag.vector_store._release_lite_server") as release,
    ):
        store = await build_milvus_store(settings)
        options = factory.call_args.kwargs
        assert options["dedicated"] is True
        assert options["grpc_options"] == {
            "grpc.keepalive_time_ms": 300000,
            "grpc.keepalive_timeout_ms": 20000,
            "grpc.keepalive_permit_without_calls": False,
        }
        await store.close()
        await store.close()
        client.close.assert_called_once()
        if mode == "lite":
            release.assert_called_once_with(
                str(await asyncio.to_thread(Path(settings.milvus_lite_path).resolve))
            )
        else:
            release.assert_not_called()


@pytest.mark.asyncio
async def test_lite_releases_only_last_path_reference(tmp_path: Path) -> None:
    """多个工厂客户端共用同一 Lite 服务时，关闭一个不能中断其他客户端。"""
    settings = Settings(_env_file=None, milvus_mode="lite", milvus_lite_path=str(tmp_path / "a.db"))
    clients = [MagicMock(), MagicMock()]
    with (
        patch("backend.app.rag.vector_store.MilvusClient", side_effect=clients),
        patch("backend.app.rag.vector_store._release_lite_server") as release,
    ):
        first = await build_milvus_store(settings)
        second = await build_milvus_store(settings)
        await first.close()
        release.assert_not_called()
        await second.close()
        release.assert_called_once_with(
            str(await asyncio.to_thread(Path(settings.milvus_lite_path).resolve))
        )
        for client in clients:
            client.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_constructor", [False, True])
async def test_lite_startup_failure_releases_orphan_server(
    tmp_path: Path, fail_constructor: bool
) -> None:
    """构造或探测失败都回收启动过的服务；不留下数据库锁给下一次启动。"""
    client = MagicMock()
    client.list_collections.side_effect = RuntimeError("test probe failure")
    settings = Settings(_env_file=None, milvus_mode="lite", milvus_lite_path=str(tmp_path / "a.db"))
    with (
        patch(
            "backend.app.rag.vector_store.MilvusClient",
            side_effect=RuntimeError("test constructor failure") if fail_constructor else None,
            return_value=client,
        ),
        patch("backend.app.rag.vector_store._release_lite_server") as release,
        pytest.raises(ProviderUnavailableError),
    ):
        await build_milvus_store(settings)
    release.assert_called_once_with(
        str(await asyncio.to_thread(Path(settings.milvus_lite_path).resolve))
    )
    assert client.close.call_count == (0 if fail_constructor else 1)


@pytest.mark.asyncio
async def test_cancelled_search_drains_thread_before_close() -> None:
    """取消不杀死 SDK 线程；close 在真实查询结束前不得关闭连接。"""
    started, finish = threading.Event(), threading.Event()
    client = MagicMock()
    store = MilvusVectorStore(
        client, VectorBackendStatus("remote", False, "test"), Settings(_env_file=None)
    )

    def blocking_search(*args: object) -> list[object]:
        """制造在途 RPC，直到测试显式释放后才允许资源关闭。"""
        started.set()
        assert finish.wait(5)
        return []

    with patch.object(store, "_get_index", side_effect=blocking_search):
        request = asyncio.create_task(store.get_index(MagicMock()))
        assert await asyncio.to_thread(started.wait, 5)
        request.cancel()
        closing = asyncio.create_task(store.close())
        try:
            await asyncio.sleep(0.02)
            assert not request.done()
            assert not closing.done()
            client.close.assert_not_called()
        finally:
            finish.set()
            results = await asyncio.gather(request, closing, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1] is None
    client.close.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_lite_startup_closes_created_store(tmp_path: Path) -> None:
    """热重载取消启动时，线程创建成功的客户端也必须关闭并释放服务。"""
    started, finish = threading.Event(), threading.Event()
    client = MagicMock()
    settings = Settings(_env_file=None, milvus_mode="lite", milvus_lite_path=str(tmp_path / "a.db"))

    def probe(*args: object, **kwargs: object) -> list[str]:
        """模拟耗时连接探测，让测试取消等待中的应用启动。"""
        started.set()
        assert finish.wait(5)
        return []

    client.list_collections.side_effect = probe
    with (
        patch("backend.app.rag.vector_store.MilvusClient", return_value=client),
        patch("backend.app.rag.vector_store._release_lite_server") as release,
    ):
        request = asyncio.create_task(build_milvus_store(settings))
        assert await asyncio.to_thread(started.wait, 5)
        request.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        client.close.assert_called_once()
        release.assert_called_once_with(
            str(await asyncio.to_thread(Path(settings.milvus_lite_path).resolve))
        )
