"""真实 Milvus Lite 集成：HTTP 测试向量只验证工程链路，医疗知识使用原始五类主库。"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.embedding import HTTPEmbeddingProvider
from backend.app.rag.reranker import HTTPReranker
from backend.app.rag.vector_store import build_milvus_store


def mock_model(request: httpx.Request) -> httpx.Response:
    """生成标明为测试的确定向量和重排响应，不能用于语义/临床质量评价。"""
    payload = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        data = []
        for index, text in enumerate(payload["input"]):
            digest = hashlib.sha256(text.encode()).digest()
            data.append({"index": index, "embedding": [value / 255 for value in digest[:4]]})
        return httpx.Response(200, json={"data": data})
    return httpx.Response(
        200,
        json={
            "results": [
                {"index": index, "relevance_score": 1.0 / (index + 1)}
                for index in range(min(payload["top_n"], len(payload["documents"])))
            ]
        },
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_lite_full_knowledge_pipeline(tmp_path: Path) -> None:
    """全量索引、重启读回、过滤、融合重排与失败回滚均使用真实 Lite。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "index.db"),
        embedding_dimension=4,
        embedding_batch_size=64,
    )
    service = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    profile = service.index_profile(
        provider="test-http", model="test-hash-vector", model_version="test-only", dimension=4
    )
    vector = await build_milvus_store(settings)
    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(mock_model)
    ) as client:
        embedding = HTTPEmbeddingProvider(settings, client)
        reranker = HTTPReranker(settings, client)
        try:
            report = await service.build_index(embedding, vector, profile)
            assert report.backend == "lite" and report.index_type == "FLAT"
            assert report.counts == service.counts()
            assert sum(report.counts.values()) == 659
            assert not report.reused
            repeated = await service.build_index(embedding, vector, profile)
            assert repeated.reused and repeated.collections == report.collections
        finally:
            await vector.close()
        vector = await build_milvus_store(settings)
        try:
            assert (await vector.get_index(profile)).counts == service.counts()
            service.configure_retrieval(
                mode="hybrid",
                embedding=embedding,
                vector_store=vector,
                profile=profile,
                reranker=reranker,
            )
            await service.check_index()
            classic = await service.search_classic("脉浮恶寒", semantic_channels=["太阳"], top_k=4)
            assert classic and len(classic) <= 4
            assert all("太阳" in evidence.metadata["semantic_six_channels"] for evidence in classic)
            assert all(
                e.retrieval_method == "hybrid_rrf" and e.rerank_score is not None for e in classic
            )
            assert any("dense" in e.channel_scores for e in classic)
            assert len({e.evidence_id for e in classic}) == len(classic)
            assert (await service.get_clause_by_id("35")).retrieval_method == "exact"
            assert (await service.get_formula_by_name("麻黄汤")).retrieval_method == "exact"
            for scope in ("classic_clause_based", "case_derived_historical"):
                formulas = await service.search_formula("恶寒", association_scope=[scope])
                assert formulas and all(e.metadata["association_scope"] == scope for e in formulas)
            rules = await service.search_differential(["麻黄汤证", "桂枝汤证"])
            assert rules[0].retrieval_method == "candidate_pair_exact"
            bundle = await service.retrieve_evidence_bundle("恶寒脉浮")
            assert (
                bundle.framework
                and bundle.classic
                and bundle.formulas
                and bundle.differential
                and bundle.cases
            )
            assert all(e.source_level == "L5" for e in bundle.cases)
            # 对未知过滤值和含引号的字符串做真实表达式解析验证。
            no_hits = await vector.search(
                query_vector=await embedding.embed_query("恶寒"),
                knowledge_type="classic_clause",
                top_k=3,
                profile=profile,
                filters={"channels": ['不存在" or canonical == true']},
            )
            assert no_hits == []
            incompatible = profile.model_copy(update={"embedding_model_version": "other-revision"})
            with pytest.raises(KnowledgeRetrievalError):
                await vector.search(
                    query_vector=[1, 0, 0, 0],
                    knowledge_type="classic_clause",
                    top_k=3,
                    profile=incompatible,
                )
            # 第二类集合构建失败后，上一完整发布版本仍可查询。
            original_create = vector._create_collection
            calls = []

            def fail_second(name: str, current_profile: object) -> None:
                """模拟暂存构建中途故障，保留真正 SDK 对第一集合的写入。"""
                calls.append(name)
                if len(calls) == 2:
                    raise RuntimeError("test staging failure")
                original_create(name, current_profile)

            with (
                patch.object(vector, "_create_collection", side_effect=fail_second),
                pytest.raises(KnowledgeRetrievalError),
            ):
                await service.build_index(embedding, vector, profile, rebuild=True)
            assert (await vector.get_index(profile)).collections == report.collections
            assert await service.search_classic("恶寒")
            rebuilt = await service.build_index(embedding, vector, profile, rebuild=True)
            assert rebuilt.collections != report.collections and not rebuilt.reused
            await vector.delete_by_version("unrelated")
            assert await vector.get_index(profile) is not None
            await vector.delete_by_version("1.2")
            assert await vector.get_index(profile) is None
        finally:
            await vector.close()
