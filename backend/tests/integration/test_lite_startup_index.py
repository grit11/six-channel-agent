"""真实 Lite 启动索引集成：验证自动发布、关闭重开复用及知识检索。

使用原始五类知识和隔离 Lite 数据；HTTP 测试向量只验证工程流程，不调用在线模型。
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.core.dependencies import configure_knowledge_service
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.embedding import HTTPEmbeddingProvider
from backend.app.rag.vector_store import build_milvus_store
from backend.tests.integration.test_knowledge_vector import mock_model


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_lite_startup_build_and_restart_reuse(tmp_path: Path) -> None:
    """首次启动缺失索引时发布五类知识，重开数据库不再调用模型并能检索。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "index.db"),
        retrieval_mode="hybrid",
        embedding_provider="test-http",
        embedding_base_url="https://model.test/v1",
        embedding_model="test-hash-vector",
        embedding_model_version="test-only",
        embedding_dimension=4,
        embedding_batch_size=64,
        reranker_enabled=False,
        database_enabled=False,
        consultation_storage="memory",
        checkpoint_backend="sqlite",
    )
    requests: list[str] = []

    def model(request: httpx.Request) -> httpx.Response:
        """记录测试模型请求，确认重启不重复向量化全库。"""
        requests.append(request.url.path)
        return mock_model(request)

    @asynccontextmanager
    async def provider(configuration: Settings) -> AsyncIterator[HTTPEmbeddingProvider]:
        """保留真实 HTTP 适配器，仅替换网络传输为离线测试响应。"""
        async with httpx.AsyncClient(
            base_url=configuration.embedding_base_url.rstrip("/") + "/",
            transport=httpx.MockTransport(model),
        ) as client:
            yield HTTPEmbeddingProvider(configuration, client)

    previous_collections: dict[str, str] | None = None
    with patch("backend.app.core.dependencies.open_embedding_provider", provider):
        for restart in (False, True):
            service = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
            async with AsyncExitStack() as stack:
                vector = await build_milvus_store(settings)
                stack.push_async_callback(vector.close)
                before = len(requests)
                await configure_knowledge_service(settings, service, vector, stack)
                report = await service.check_index()
                assert report is not None and report.counts == service.counts()
                assert sum(report.counts.values()) == 659
                if restart:
                    assert len(requests) == before
                    assert report.collections == previous_collections
                else:
                    assert len(requests) > before
                    previous_collections = report.collections
                assert await service.search_classic("脉浮恶寒", top_k=3)
