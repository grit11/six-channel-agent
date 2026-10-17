"""模型与向量化能力端口，应用模块只依赖抽象，具体 Provider 由工厂注入。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


@dataclass(frozen=True)
class S4StreamPolicy:
    """S4 流式调用预算；只约束模型等待，不改变 S1/S2 的通用超时。"""

    total_seconds: float = 110.0
    first_content_seconds: float = 40.0
    idle_seconds: float = 25.0
    max_output_tokens: int = 3000
    connect_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class StreamUsage:
    """S4 流式公开进度元数据，不包含模型正文或隐藏推理。"""

    output_chars: int
    content_chunks: int
    usage_status: str = "unavailable"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: int = 0


class StreamingStructuredLLM(Protocol):
    """S4 专用流式结构化端口，业务层不依赖具体 HTTP Provider。"""

    async def ainvoke_structured_stream(
        self,
        messages: list[dict[str, str]],
        schema: type[StructuredResult],
        *,
        policy: S4StreamPolicy,
        on_progress: Callable[[StreamUsage], Awaitable[None]] | None = None,
    ) -> StructuredResult:
        """在预算内消费 SSE 并仅在完整 JSON 结束后返回 Schema 对象。"""
        ...


class StructuredLLM(Protocol):
    """大模型文本与结构化调用端口，具体 Provider 由依赖工厂注入。"""

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """调用文本模型，Adapter 必须实现超时和有限重试。"""
        ...

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """按 Schema 返回结构化结果，禁止伪造调用成功。"""
        ...


@dataclass(frozen=True)
class ClinicalStageStreamPolicy:
    """S1/S2 独立工程预算，无输出 token 截断，不改变医学消息。"""

    total_seconds: float = 180.0
    first_content_seconds: float = 90.0
    idle_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0


class ClinicalStreamingLLM(Protocol):
    """临床入口独立流端口，S4 继续使用原端口。"""

    async def ainvoke_clinical_structured_stream(
        self,
        messages: list[dict[str, str]],
        schema: type[StructuredResult],
        *,
        stage: str,
        deadline: float | None = None,
    ) -> StructuredResult:
        """仅完整流结束并通过原 Schema 才返回结果。"""
        ...


class EmbeddingProvider(Protocol):
    """向量化抽象，具体模型在 Phase 1 由工厂配置。"""

    @property
    def dimension(self) -> int:
        """返回索引维度，模型维度变化必须新建版本索引。"""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """生成单条查询向量。"""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成知识文档向量。"""
        ...
