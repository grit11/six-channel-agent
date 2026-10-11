"""开发环境检查工具：只读取环境并可探测真实向量后端，不安装依赖或生成医学数据。"""

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

# 从仓库根执行或任意工作目录执行均保持同一项目导入边界。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def probe_vector() -> bool:
    """按配置探测真实 Milvus，失败只报告安全的业务消息。"""
    from backend.app.core.config import Settings
    from backend.app.core.exceptions import ProviderUnavailableError
    from backend.app.rag.vector_store import build_milvus_store

    try:
        store = await build_milvus_store(Settings())
    except ProviderUnavailableError as exc:
        print(json.dumps({"vector_available": False, "message": exc.message}, ensure_ascii=False))
        return False
    try:
        print(json.dumps({"vector_backend": store.status.backend}, ensure_ascii=False))
        return True
    finally:
        await store.close()


def main() -> int:
    """报告缺少的声明依赖；未通过检查时返回非零退出码。"""
    parser = argparse.ArgumentParser(description="检查 langgraph 开发环境")
    parser.add_argument("--probe-vector", action="store_true", help="额外探测真实向量后端")
    args = parser.parse_args()
    modules = [
        "fastapi",
        "pydantic_settings",
        "pymilvus",
        "milvus_lite",
        "langgraph",
        "langgraph.checkpoint.sqlite.aio",
        "langgraph.checkpoint.postgres.aio",
        "sqlalchemy",
        "asyncpg",
        "pytest",
        "ruff",
        "black",
        "mypy",
    ]
    availability = {name: importlib.util.find_spec(name) is not None for name in modules}
    in_environment = Path(sys.executable).parent.name.lower() == "langgraph"
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "langgraph_environment": in_environment,
                "dependencies": availability,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    vector_ok = asyncio.run(probe_vector()) if args.probe_vector else True
    return 0 if in_environment and all(availability.values()) and vector_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
