"""启动索引回归：通过真实 KnowledgeService 验证补建、复用、版本隔离及失败释放。

向量存储使用内存测试端口，不写业务数据库，不访问在线 Embedding；不评测医学质量。
"""

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.app.core.config import Settings
from backend.app.core.dependencies import configure_knowledge_service
from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.core.ports import EmbeddingProvider
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.vector_store import MilvusVectorStore, VectorBackendStatus
from backend.app.schemas.knowledge import IndexProfile, IndexReport, KnowledgeRecord, VectorDocument


def startup_fixture() -> tuple[Settings, KnowledgeService, Mock, Mock, dict[str, IndexReport]]:
    """装配五类真实结构知识与可发布内存端口，身份变化不能复用旧报告。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="hybrid",
        milvus_mode="auto",
        embedding_provider="test",
        embedding_base_url="https://model.test/v1",
        embedding_model="test-vector",
        embedding_model_version="v1",
        embedding_dimension=2,
        reranker_enabled=False,
        database_enabled=False,
        consultation_storage="memory",
        checkpoint_backend="sqlite",
    )
    types = [
        "classic_clause",
        "formula_pattern",
        "differential",
        "medical_case",
        "six_channel_framework",
    ]
    records = [
        KnowledgeRecord(
            knowledge_id=f"test_{kind}",
            knowledge_type=kind,
            knowledge_version="1.2",
            source="测试来源",
            source_level="L1",
            content="工程测试",
            embedding_text="工程测试",
            metadata={"clause_no": "1"} if kind == "classic_clause" else {},
        )
        for kind in types
    ]
    service = KnowledgeService(records, "1.2")
    reports: dict[str, IndexReport] = {}
    vector = Mock(spec=MilvusVectorStore)
    vector.status = VectorBackendStatus("lite", True, "测试 auto 降级")

    async def get_index(profile: IndexProfile) -> IndexReport | None:
        """按当前快照身份读取，模拟持久化发布记录。"""
        return reports.get(profile.fingerprint())

    async def publish(
        documents: list[VectorDocument], profile: IndexProfile, *, rebuild: bool = False
    ) -> IndexReport:
        """完整文档才发布五类集合报告，模拟适配器发布契约。"""
        assert len(documents) == 5 and not rebuild
        report = IndexReport(
            backend="lite",
            index_type="FLAT",
            profile=profile,
            counts=service.counts(),
            collections={kind: kind for kind in service.counts()},
            reused=False,
        )
        reports[profile.fingerprint()] = report
        return report

    vector.get_index = AsyncMock(side_effect=get_index)
    vector.upsert = AsyncMock(side_effect=publish)
    embedding = Mock(spec=EmbeddingProvider)
    embedding.dimension = 2
    embedding.embed_documents = AsyncMock(return_value=[[1.0, 0.5]] * 5)
    embedding.embed_query = AsyncMock(return_value=[1.0, 0.5])
    return settings, service, vector, embedding, reports


@pytest.mark.asyncio
async def test_lite_auto_fallback_builds_once_and_reuses_after_restart() -> None:
    """配置 auto、实际 Lite 也自动建立；换服务实例模拟重启，不重复付费向量化。"""
    settings, service, vector, embedding, reports = startup_fixture()
    with patch("backend.app.core.dependencies.open_embedding_provider") as provider:
        provider.return_value.__aenter__ = AsyncMock(return_value=embedding)
        provider.return_value.__aexit__ = AsyncMock(return_value=False)
        for _ in range(2):
            async with AsyncExitStack() as stack:
                await configure_knowledge_service(settings, service, vector, stack)
                assert await service.check_index() is not None
            service = KnowledgeService(list(service._records.values()), "1.2")
    assert len(reports) == 1
    embedding.embed_documents.assert_awaited_once()
    vector.upsert.assert_awaited_once()
    assert provider.return_value.__aexit__.await_count == 2


@pytest.mark.asyncio
async def test_new_model_version_builds_separate_index() -> None:
    """模型修订变更补建新身份，原发布版本保留且不被误用。"""
    settings, service, vector, embedding, reports = startup_fixture()
    with patch("backend.app.core.dependencies.open_embedding_provider") as provider:
        provider.return_value.__aenter__ = AsyncMock(return_value=embedding)
        provider.return_value.__aexit__ = AsyncMock(return_value=False)
        for version in ("v1", "v2"):
            async with AsyncExitStack() as stack:
                await configure_knowledge_service(
                    settings.model_copy(update={"embedding_model_version": version}),
                    service,
                    vector,
                    stack,
                )
    assert len(reports) == 2 and embedding.embed_documents.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("remote,enabled", [(True, True), (False, False)])
async def test_remote_or_disabled_does_not_build(remote: bool, enabled: bool) -> None:
    """Remote 或关闭自动补建时缺失索引明确失败，不私自重建。"""
    settings, service, vector, embedding, reports = startup_fixture()
    settings = settings.model_copy(update={"milvus_lite_auto_index": enabled})
    if remote:
        vector.status = VectorBackendStatus("remote", False, "测试远程")
    with patch("backend.app.core.dependencies.open_embedding_provider") as provider:
        provider.return_value.__aenter__ = AsyncMock(return_value=embedding)
        provider.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(KnowledgeRetrievalError, match="尚未完整索引"):
            async with AsyncExitStack() as stack:
                await configure_knowledge_service(settings, service, vector, stack)
    embedding.embed_documents.assert_not_awaited()
    assert not reports


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["embedding", "publish"])
async def test_failed_build_closes_provider_and_next_start_retries(phase: str) -> None:
    """模型或发布失败不标记就绪，释放资源；下次启动自动重试完整构建。"""
    settings, service, vector, embedding, reports = startup_fixture()
    failing = embedding.embed_documents if phase == "embedding" else vector.upsert
    original = failing.side_effect
    failing.side_effect = KnowledgeRetrievalError("模拟构建不可用")
    with patch("backend.app.core.dependencies.open_embedding_provider") as provider:
        provider.return_value.__aenter__ = AsyncMock(return_value=embedding)
        provider.return_value.__aexit__ = AsyncMock(return_value=False)
        with pytest.raises(KnowledgeRetrievalError, match="自动索引失败"):
            async with AsyncExitStack() as stack:
                await configure_knowledge_service(settings, service, vector, stack)
        provider.return_value.__aexit__.assert_awaited_once()
        assert not reports
        if phase == "embedding":
            vector.upsert.assert_not_awaited()
        failing.side_effect = original
        async with AsyncExitStack() as stack:
            await configure_knowledge_service(settings, service, vector, stack)
            assert await service.check_index() is not None
        assert provider.return_value.__aexit__.await_count == 2
    assert len(reports) == 1


@pytest.mark.asyncio
async def test_bm25_does_not_open_embedding_or_index() -> None:
    """离线测评和 BM25 模式不创建模型客户端、不读写向量索引。"""
    settings, service, vector, _, _ = startup_fixture()
    with patch("backend.app.core.dependencies.open_embedding_provider") as provider:
        async with AsyncExitStack() as stack:
            await configure_knowledge_service(
                settings.model_copy(update={"retrieval_mode": "bm25"}), service, vector, stack
            )
    provider.assert_not_called()
    vector.get_index.assert_not_awaited()
    vector.upsert.assert_not_awaited()
