"""浏览器回归专用后端：隔离数据库和模型配置，使用真实知识、Lite 和图，不读取密钥。"""

import argparse
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

import uvicorn
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.clinical_v2.schemas import ChannelLocalizationV2, S1ExtractionDraft
from backend.app.core import dependencies
from backend.app.core.config import Settings
from backend.app.core.ports import StructuredLLM
from backend.app.main import create_app
from backend.app.observability.progress import model_scope
from backend.tests.v2.test_end_to_end_workflow import FullLLM
from backend.tests.v2.test_s2_hitl_resume import UncertainLLM

T = TypeVar("T", bound=BaseModel)


class BrowserFixtureLLM(FullLLM):
    """仅供隔离浏览器测试使用的模型替身，不进入生产依赖工厂。"""

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """已知测试输入走对应夹具，其他临床语义不在此处实现。"""
        # 仅浏览器工程夹具使用可识别的合成用量，生产仍以供应商 usage 为准。
        async with model_scope("chat") as usage:
            usage.update(usage_status="available", prompt_tokens=100, completion_tokens=20)
            payload = json.loads(messages[-1]["content"])
            if schema is S1ExtractionDraft and payload["clinical_text"] == "难受":
                return await UncertainLLM().ainvoke_structured(messages, schema)
            if schema is ChannelLocalizationV2 and payload["raw_text"] == "难受":
                return await UncertainLLM().ainvoke_structured(messages, schema)
            return await super().ainvoke_structured(messages, schema)


@asynccontextmanager
async def fixture_llm(settings: Settings) -> AsyncIterator[StructuredLLM]:
    """替代网络工厂，仍让真实应用 lifespan、Graph 和 Repository 完整执行。"""
    yield BrowserFixtureLLM()


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
            llm_base_url="http://model.test/v1",
            llm_model="e2e-fixture",
            consultation_run_mode="async",
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
        dependencies.open_llm = fixture_llm
        uvicorn.run(create_app(settings), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
