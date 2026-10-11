"""浏览器回归专用后端：隔离数据库和模型配置，使用真实知识、Lite 和图，不读取密钥。"""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.main import create_app


def main() -> None:
    """以离线 BM25 和临时基础设施运行后端，退出后清理测试数据。"""
    parser = argparse.ArgumentParser(description="运行隔离的 E2E API")
    parser.add_argument("--port", type=int, default=8017)
    arguments = parser.parse_args()
    # 允许验收入口注入短路径；默认使用系统临时目录，规避 Windows Lite 内部路径过长。
    runtime = os.environ.get("CONSOLE_TEST_RUNTIME")
    with TemporaryDirectory(prefix="sc_e2e_", dir=runtime) as directory:
        settings = Settings(
            _env_file=None,
            clinical_parser_mode="rules",
            reasoning_mode="rules",
            knowledge_qa_mode="extractive",
            retrieval_mode="bm25",
            reranker_enabled=False,
            database_enabled=False,
            consultation_storage="memory",
            milvus_mode="lite",
            milvus_lite_path=str(Path(directory) / "milvus.db"),
            checkpoint_backend="sqlite",
            langgraph_sqlite_path=str(Path(directory) / "graph.db"),
        )
        uvicorn.run(create_app(settings), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
