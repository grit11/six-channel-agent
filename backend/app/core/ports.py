"""模型与向量化能力端口，应用模块只依赖抽象，具体 Provider 由工厂注入。"""

from typing import Protocol, TypeVar

from pydantic import BaseModel

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


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
