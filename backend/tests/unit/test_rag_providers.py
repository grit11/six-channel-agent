"""模型协议单测：注入 HTTP 响应检验真实适配代码，不访问付费服务或编造医疗知识。"""

import json

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import ConfigurationError, ProviderUnavailableError
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.embedding import HTTPEmbeddingProvider, validate_embedding_config
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.reranker import HTTPReranker, validate_reranker_config


def provider_settings() -> Settings:
    """生成不含真实密钥的协议测试配置。"""
    return Settings(
        _env_file=None,
        embedding_base_url="https://embedding.test/v1",
        embedding_model="test-model",
        embedding_model_version="test-revision",
        embedding_dimension=3,
        embedding_batch_size=2,
        embedding_max_retries=1,
        reranker_model="test-reranker",
        reranker_base_url="https://rerank.test/v1",
        reranker_max_retries=0,
    )


@pytest.mark.asyncio
async def test_embedding_batch_response_order() -> None:
    """服务返回乱序也应恢复输入顺序，并按配置分批。"""
    batches = []

    def handle(request: httpx.Request) -> httpx.Response:
        """构造协议响应，不代表模型语义向量。"""
        payload = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert payload["model"] == "test-model"
        batch = payload["input"]
        batches.append(batch)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(text), 1.0, 0.0]}
                    for index, text in reversed(list(enumerate(batch)))
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="https://embedding.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        provider = HTTPEmbeddingProvider(provider_settings(), client)
        assert await provider.embed_documents(["1", "2", "3"]) == [[1, 1, 0], [2, 1, 0], [3, 1, 0]]
        assert batches == [["1", "2"], ["3"]]
        assert await provider.embed_query("4") == [4, 1, 0]
        assert await provider.embed_documents([]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 0, "embedding": [1.0, 2.0]}],
        [{"index": 0, "embedding": [0.0, 0.0, 0.0]}],
        [{"index": 1, "embedding": [1.0, 0.0, 0.0]}],
        [{"index": 0, "embedding": [1.0, 0.0, 0.0]}, {"index": 0, "embedding": [1.0, 0.0, 0.0]}],
        [{"index": 0, "embedding": ["NaN", 0.0, 1.0]}],
    ],
)
async def test_embedding_invalid_response(data: list[dict[str, object]]) -> None:
    """响应数量、序号、维度和数值错误不得当作成功。"""
    async with httpx.AsyncClient(
        base_url="https://embedding.test/v1/",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": data})),
    ) as client:
        with pytest.raises(ProviderUnavailableError) as caught:
            await HTTPEmbeddingProvider(provider_settings(), client).embed_query("测试")
        assert caught.value.__cause__ is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected_calls", [(429, 2), (503, 2), (401, 1), (400, 1)])
async def test_provider_retry_policy(status: int, expected_calls: int) -> None:
    """只重试暂时故障，鉴权和参数错误立即失败。"""
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        """记录有限重试次数。"""
        calls.append(request)
        return httpx.Response(status, json={"error": "unavailable"})

    async with httpx.AsyncClient(
        base_url="https://embedding.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ProviderUnavailableError):
            await HTTPEmbeddingProvider(provider_settings(), client).embed_query("测试")
    assert len(calls) == expected_calls


@pytest.mark.asyncio
async def test_rerank_preserves_source_and_separate_scores() -> None:
    """重排只引用原始来源，不接受响应内的替换文本。"""
    service = await KnowledgeService.from_directory(Settings().knowledge_directory, "1.2")
    documents = [await service.get_clause_by_id("1"), await service.get_clause_by_id("2")]
    assert all(document is not None for document in documents)

    def handle(request: httpx.Request) -> httpx.Response:
        """返回乱序评分及应忽略的响应文本。"""
        assert request.url.path == "/v1/rerank"
        payload = json.loads(request.content)
        assert payload["top_n"] == 2 and len(payload["documents"]) == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.2, "document": "不可用的服务改写"},
                    {"index": 1, "relevance_score": 0.9},
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="https://rerank.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPReranker(provider_settings(), client).rerank("汗出恶风", documents, 2)
    assert result[0].knowledge_id == "clause_2"
    assert result[0].content == documents[1].content
    assert result[0].rerank_score == 0.9
    assert result[0].retrieval_score == documents[1].retrieval_score
    assert documents[1].rerank_score is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 20, "relevance_score": 0.8}],
        [{"index": 0, "relevance_score": "NaN"}],
        [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}],
    ],
)
async def test_invalid_rerank_response(results: list[dict[str, object]]) -> None:
    """重排序号错误时不能引用错误医疗来源。"""
    service = await KnowledgeService.from_directory(Settings().knowledge_directory, "1.2")
    evidence = await service.get_clause_by_id("1")
    assert evidence is not None
    async with httpx.AsyncClient(
        base_url="https://rerank.test/v1/",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": results})),
    ) as client:
        with pytest.raises(ProviderUnavailableError):
            await HTTPReranker(provider_settings(), client).rerank("恶寒", [evidence], 1)


@pytest.mark.asyncio
async def test_fusion_dedup_and_channel_scores() -> None:
    """融合按排名而非分数相加，同一通道重复来源不增加权重。"""
    service = await KnowledgeService.from_directory(Settings().knowledge_directory, "1.2")
    first = await service.get_clause_by_id("1")
    second = await service.get_clause_by_id("2")
    assert first is not None and second is not None
    first.retrieval_score = 1000
    second.retrieval_score = 0.8
    result = reciprocal_rank_fusion({"bm25": [first, first, second], "dense": [second]}, 3)
    assert len(result) == 2 and result[0].knowledge_id == "clause_2"
    assert result[0].channel_scores == {"bm25": 0.8, "dense": 0.8}
    assert result[0].retrieval_method == "hybrid_rrf"
    assert result[0].rerank_score is None
    assert first.retrieval_method == "exact"


@pytest.mark.parametrize("provider", ["embedding", "reranker"])
def test_missing_provider_configuration(provider: str) -> None:
    """缺少模型配置时明确失败，不构造伪模型。"""
    with pytest.raises(ConfigurationError):
        if provider == "embedding":
            validate_embedding_config(Settings(_env_file=None))
        else:
            validate_reranker_config(Settings(_env_file=None))
