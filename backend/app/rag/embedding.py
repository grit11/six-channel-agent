"""Embedding HTTP 适配：依赖配置和异步客户端，校验响应，不提供伪造的生产向量。"""

import asyncio
import math

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import Settings
from backend.app.core.exceptions import ConfigurationError, ProviderUnavailableError
from backend.app.observability.progress import model_scope


def validate_vectors(vectors: list[list[float]], count: int, dimension: int) -> None:
    """拒绝缺条、维度错误、非有限数和零向量，避免不可解释的相似度。"""
    if len(vectors) != count or any(
        len(vector) != dimension
        or not all(math.isfinite(value) for value in vector)
        or not any(value != 0 for value in vector)
        for vector in vectors
    ):
        raise ValueError("Embedding 响应数量、维度或数值无效")


class _EmbeddingItem(BaseModel):
    """按响应序号恢复输入顺序，禁止依赖服务返回顺序。"""

    index: int = Field(ge=0, strict=True)
    embedding: list[float]


class _EmbeddingResponse(BaseModel):
    """仅解析向量响应，附加厂商字段不进入知识元数据。"""

    model_config = ConfigDict(extra="ignore")
    data: list[_EmbeddingItem]


async def post_json(
    client: httpx.AsyncClient, endpoint: str, payload: dict[str, object], retries: int
) -> object:
    """只对网络错误、限流和服务暂时错误有限重试，不记录凭据或输入原文。"""
    for attempt in range(retries + 1):
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
        except httpx.TransportError:
            if attempt == retries:
                raise
        await asyncio.sleep(min(0.25 * 2**attempt, 2.0))
    raise RuntimeError("不可达的 HTTP 重试状态")


class HTTPEmbeddingProvider:
    """兼容 /embeddings 协议的批量向量化适配器，生命周期由工厂管理。"""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """绑定配置与注入客户端，模型名、地址和密钥均不硬编码。"""
        self._settings = settings
        self._client = client

    @property
    def dimension(self) -> int:
        """返回已配置并在每次响应中校验的向量维度。"""
        return self._settings.embedding_dimension

    async def embed_query(self, text: str) -> list[float]:
        """在线查询采用同一模型和数值契约。"""
        if not text.strip():
            raise ValueError("查询文本不能为空")
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批调用服务并恢复响应序号；任意批次失败则整体失败。"""
        if any(not text.strip() for text in texts):
            raise ValueError("索引文本不能为空")
        vectors: list[list[float]] = []
        try:
            async with model_scope("embedding"):
                for start in range(0, len(texts), self._settings.embedding_batch_size):
                    batch = texts[start : start + self._settings.embedding_batch_size]
                    raw = await post_json(
                        self._client,
                        "embeddings",
                        {"model": self._settings.embedding_model, "input": batch},
                        self._settings.embedding_max_retries,
                    )
                    items = _EmbeddingResponse.model_validate(raw).data
                    if sorted(item.index for item in items) != list(range(len(batch))):
                        raise ValueError("Embedding 响应序号重复或缺失")
                    ordered = [
                        item.embedding for item in sorted(items, key=lambda item: item.index)
                    ]
                    validate_vectors(ordered, len(batch), self.dimension)
                    vectors.extend(ordered)
        except Exception as exc:
            raise ProviderUnavailableError("Embedding 调用或响应校验失败") from exc
        return vectors


def validate_embedding_config(settings: Settings) -> None:
    """构造时拒绝未配置模型，开发态允许本地无密钥服务。"""
    if settings.embedding_provider not in {"openai_compatible", "siliconflow"}:
        raise ConfigurationError("当前仅支持 openai_compatible 或 siliconflow Embedding 协议")
    if not all(
        (settings.embedding_base_url, settings.embedding_model, settings.embedding_model_version)
    ):
        raise ConfigurationError("请配置 Embedding 地址、模型名和模型版本")
    if not settings.embedding_base_url.startswith(("http://", "https://")):
        raise ConfigurationError("Embedding 地址必须是 HTTP(S) API 根路径")
