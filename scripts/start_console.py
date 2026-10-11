"""启动独立 FastAPI Console；有 .env 时使用部署配置，否则使用本地模板。

只负责启动与配置选择，不复制密钥、不改动原系统，医学流程由后端业务层负责。
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings
from backend.app.main import create_app


def run_console(app: FastAPI, port: int) -> None:
    """在 Windows 显式使用 psycopg 兼容循环，其他平台沿用 Uvicorn 默认入口。"""
    if sys.platform != "win32":
        uvicorn.run(app, host="127.0.0.1", port=port)
        return
    # 新版 Uvicorn 会显式选择 Proactor，单独设置全局 policy 无法确保生效。
    # 由 Runner 创建 Selector 并直接 await serve，保证 lifespan 与数据库连接共用该循环。
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, loop="asyncio"))
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(server.serve())


def main() -> None:
    """使用副本内显式配置启动；模型故障不自动降级为固定演示结果。"""
    parser = argparse.ArgumentParser(description="六经辨证 Agent Console")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.chdir(ROOT)
    configuration = ROOT / ".env"
    if not configuration.is_file():
        configuration = ROOT / ".env.example"
    # _env_file 是 BaseSettings 的运行时参数，Pydantic 静态生成签名未包含它。
    settings = Settings(_env_file=configuration)  # type: ignore[call-arg]
    print(f"Console: http://127.0.0.1:{args.port}/dashboard", flush=True)
    print(f"Configuration: {configuration.name}", flush=True)
    run_console(create_app(settings), args.port)


if __name__ == "__main__":
    main()
