"""结构化 Chat HTTP 适配器。

职责：通过配置和注入的异步客户端调用兼容协议，有限重试并验证 Pydantic 输出。
边界：不含医学规则、不记录原文或隐藏推理，不负责业务写入；客户端由容器关闭。
"""

import asyncio
import json

import httpx
from pydantic import BaseModel, ValidationError

from backend.app.core.config import Settings
from backend.app.core.exceptions import ConfigurationError, ProviderUnavailableError
from backend.app.core.ports import StructuredResult
from backend.app.observability.progress import model_scope


def validate_llm_config(settings: Settings) -> None:
    """启动时验证协议、服务地址与模型，允许明确配置的本地无密钥服务。"""
    if settings.llm_provider not in {"openai_compatible", "deepseek"}:
        raise ConfigurationError("LLM Provider 必须使用已支持的兼容 Chat 协议")
    if not settings.llm_base_url.startswith(("https://", "http://")) or not settings.llm_model:
        raise ConfigurationError("请配置 LLM HTTP(S) 地址和模型名")


class HTTPStructuredLLM:
    """兼容 Chat Completions 的结构化适配器，不绑定厂商 SDK 或模型。"""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """保存配置和容器持有的客户端，密钥只由工厂放入请求头。"""
        self._settings = settings
        self._client = client

    async def _call(self, messages: list[dict[str, str]], *, structured: bool) -> str:
        """执行单次协议请求，拒绝拒答、截断、缺少正文和非标准内容。"""
        payload: dict[str, object] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "stream": False,
        }
        if structured:
            payload["response_format"] = {"type": "json_object"}
        response = await self._client.post("chat/completions", json=payload)
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list):
            raise ValueError("Chat 响应缺少 choices")
        choices = raw["choices"]
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Chat 响应选择数量不合法")
        choice = choices[0]
        message = choice.get("message")
        if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
            raise ValueError("Chat 响应未正常完成")
        content = message.get("content")
        if message.get("refusal") or not isinstance(content, str) or not content.strip():
            raise ValueError("Chat 拒答或缺少内容")
        # 忽略厂商 reasoning_content 等字段，不保存或返回隐藏推理。
        return content

    async def _invoke(
        self, messages: list[dict[str, str]], schema: type[StructuredResult] | None
    ) -> str | StructuredResult:
        """网络暂态及结构错误有限重试；错误摘要不含原文、密钥或模型响应。"""
        prepared = [dict(message) for message in messages]
        if schema is not None:
            prepared.insert(
                0,
                {
                    "role": "system",
                    "content": "只返回 JSON 对象，遵守以下 JSON Schema；不输出隐藏推理："
                    + json.dumps(schema.model_json_schema(), ensure_ascii=False),
                },
            )
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                async with model_scope("chat"):
                    content = await self._call(prepared, structured=schema is not None)
                    return schema.model_validate_json(content) if schema is not None else content
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise ProviderUnavailableError("LLM 服务拒绝请求") from exc
                if attempt == self._settings.llm_max_retries:
                    raise ProviderUnavailableError("LLM 暂时不可用，重试次数已耗尽") from exc
            except (httpx.TransportError, ValueError, ValidationError) as exc:
                if attempt == self._settings.llm_max_retries:
                    raise ProviderUnavailableError("LLM 调用或结构化响应校验失败") from exc
            await asyncio.sleep(min(0.25 * 2**attempt, 2.0))
        raise RuntimeError("不可达的 LLM 重试状态")

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """返回正常完成的正文，模型或协议失败时保留明确错误。"""
        result = await self._invoke(messages, None)
        if not isinstance(result, str):
            raise TypeError("LLM 文本响应类型错误")
        return result

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """返回经指定 Schema 验证的结构化对象，业务来源校验由调用节点负责。"""
        result = await self._invoke(messages, schema)
        if not isinstance(result, BaseModel):
            raise TypeError("LLM 结构化响应类型错误")
        return schema.model_validate(result)
