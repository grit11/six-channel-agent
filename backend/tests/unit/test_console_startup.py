"""Console 启动回归：验证 Windows 服务在 Selector 循环执行并关闭生命周期资源。

仅启动隔离的 FastAPI 生命周期，不读取 .env、不监听端口、不连接真实数据库或模型。
"""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from scripts import start_console


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 专用事件循环兼容回归")
def test_windows_lifespan_uses_selector_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实运行 Selector：服务生命周期与异步连接所在循环一致，退出执行清理。"""
    stages: list[str] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """模拟数据库资源生命周期，检查 psycopg 所需的循环类型。"""
        assert isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop)
        stages.append("startup")
        try:
            yield
        finally:
            stages.append("shutdown")

    app = FastAPI(lifespan=lifespan)

    async def serve(server: start_console.uvicorn.Server) -> None:
        """进入真实 FastAPI lifespan，避免监听端口和外部基础设施。"""
        assert server.config.app is app
        assert server.config.port == 8001
        configured_app = server.config.app
        assert isinstance(configured_app, FastAPI)
        async with configured_app.router.lifespan_context(configured_app):
            await asyncio.sleep(0)
            stages.append("serving")

    monkeypatch.setattr(start_console.uvicorn.Server, "serve", serve)
    start_console.run_console(app, 8001)
    assert stages == ["startup", "serving", "shutdown"]


def test_other_platforms_keep_uvicorn_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux 等部署环境继续由 Uvicorn 选择平台支持的事件循环。"""
    monkeypatch.setattr(start_console.sys, "platform", "linux")
    run = Mock()
    monkeypatch.setattr(start_console.uvicorn, "run", run)
    app = FastAPI()
    start_console.run_console(app, 8002)
    run.assert_called_once_with(app, host="127.0.0.1", port=8002)
