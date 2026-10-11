"""重排器端口及 HTTP 适配：依赖配置与异步客户端，仅改变证据排序，保留来源。"""

import math
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from backend.app.core.config import Settings
from backend.app.core.exceptions import ConfigurationError, ProviderUnavailableError
from backend.app.observability.progress import model_scope
from backend.app.rag.embedding import post_json
from backend.app.schemas.evidence import RetrievedDocument


class Reranker(Protocol):
    """厂商无关的证据重排接口，具体适配在 Phase 1 接入。"""

    async def rerank(
        self, query: str, candidates: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        """保留 Evidence 来源和召回分，独立记录重排分。"""
        ...


class _RerankItem(BaseModel):
    """只允许响应引用输入索引，不接受服务生成的证据内容。"""

    index: int = Field(ge=0, strict=True)
    relevance_score: float


class _RerankResponse(BaseModel):
    """兼容 results/index/relevance_score 的重排响应。"""

    results: list[_RerankItem]


class HTTPReranker:
    """兼容 /rerank 协议的适配器，重排分与召回分严格分开。"""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """注入配置和客户端，不硬编码厂商或模型。"""
        self._settings = settings
        self._client = client

    async def rerank(
        self, query: str, candidates: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        """只按合法输入索引选择原始证据，拒绝缺项、重复和非有限分数。"""
        if top_k < 1 or not query.strip():
            raise ValueError("重排数量必须大于零且查询不能为空")
        if not candidates:
            return []
        count = min(top_k, len(candidates))
        try:
            async with model_scope("reranker"):
                raw = await post_json(
                    self._client,
                    "rerank",
                    {
                        "model": self._settings.reranker_model,
                        "query": query,
                        "documents": [candidate.content for candidate in candidates],
                        "top_n": count,
                    },
                    self._settings.reranker_max_retries,
                )
                items = _RerankResponse.model_validate(raw).results
                if (
                    len(items) != count
                    or len({item.index for item in items}) != count
                    or any(
                        item.index >= len(candidates) or not math.isfinite(item.relevance_score)
                        for item in items
                    )
                ):
                    raise ValueError("重排序号或分数无效")
                ordered = sorted(items, key=lambda item: (-item.relevance_score, item.index))
                return [
                    candidates[item.index].model_copy(
                        deep=True, update={"rerank_score": item.relevance_score}
                    )
                    for item in ordered
                ]
        except Exception as exc:
            raise ProviderUnavailableError("Reranker 调用或响应校验失败") from exc


def validate_reranker_config(settings: Settings) -> None:
    """构造时校验服务协议与 API 根路径，允许本地开发无密钥。"""
    if settings.reranker_provider not in {"http", "siliconflow"}:
        raise ConfigurationError("当前仅支持 http 或 siliconflow Reranker 协议")
    if not settings.reranker_model or not settings.reranker_base_url.startswith(
        ("http://", "https://")
    ):
        raise ConfigurationError("请配置 Reranker 模型和 HTTP(S) API 根路径")
