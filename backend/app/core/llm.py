"""结构化 Chat HTTP 适配器。

职责：通过配置和注入的异步客户端调用兼容协议，有限重试并验证 Pydantic 输出。
边界：不含医学规则、不记录原文或隐藏推理，不负责业务写入；客户端由容器关闭。
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from time import perf_counter

import httpx
from pydantic import BaseModel, ValidationError

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ConfigurationError,
    ProviderUnavailableError,
    S4StreamError,
)
from backend.app.core.ports import S4StreamPolicy, StreamUsage, StructuredResult
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

    async def _call(
        self, messages: list[dict[str, str]], *, structured: bool, usage: dict[str, object]
    ) -> str:
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
        supplied = raw.get("usage")
        if isinstance(supplied, dict):
            keys = ("prompt_tokens", "completion_tokens")
            if all(type(supplied.get(key)) is int and supplied[key] >= 0 for key in keys):
                usage.update({key: supplied[key] for key in keys})
                usage["usage_status"] = "available"
        choices = raw["choices"]
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Chat 响应选择数量不合法")
        choice = choices[0]
        message = choice.get("message")
        if choice.get("finish_reason") == "length":
            usage["error_category"] = "truncation"
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
                async with model_scope("chat", attempt=attempt) as usage:
                    content = await self._call(prepared, structured=schema is not None, usage=usage)
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

    @staticmethod
    def _sse_event(lines: list[str]) -> str | None:
        """合并一个 SSE 事件的多行 data；注释和无 data 事件被忽略。"""
        values = [line[5:].lstrip() for line in lines if line.startswith("data:")]
        return "\n".join(values) if values else None

    async def ainvoke_structured_stream(
        self,
        messages: list[dict[str, str]],
        schema: type[StructuredResult],
        *,
        policy: S4StreamPolicy,
        on_progress: Callable[[StreamUsage], Awaitable[None]] | None = None,
    ) -> StructuredResult:
        """仅供 S4 消费标准 Chat SSE，在总预算内返回一次完整结构化结果。

        已产生正文后任何断流都直接失败，避免把第二次完整请求拼接到第一次的部分 JSON；
        无正文的连接瞬时失败最多重试一次，S1/S2 的非流式端口不经过此路径。
        """
        prepared = [dict(message) for message in messages]
        prepared.insert(
            0,
            {
                "role": "system",
                "content": "只返回 JSON 对象，遵守以下 JSON Schema；不输出隐藏推理："
                + json.dumps(schema.model_json_schema(), ensure_ascii=False),
            },
        )
        payload: dict[str, object] = {
            "model": self._settings.llm_model,
            "messages": prepared,
            "temperature": self._settings.llm_temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"},
            "max_tokens": policy.max_output_tokens,
        }
        loop = asyncio.get_running_loop()
        started = perf_counter()
        deadline = loop.time() + policy.total_seconds
        last_progress = 0.0
        content = ""
        chunks = 0
        finish_reason: str | None = None
        usage_values: dict[str, int] = {}
        saw_done = False

        async def report() -> None:
            """按时间节流公开元数据，不泄露增量正文。"""
            nonlocal last_progress
            now = loop.time()
            if on_progress is None or now - last_progress < 2.0:
                return
            last_progress = now
            await on_progress(
                StreamUsage(
                    output_chars=len(content),
                    content_chunks=chunks,
                    usage_status="available" if usage_values else "unavailable",
                    prompt_tokens=usage_values.get("prompt_tokens"),
                    completion_tokens=usage_values.get("completion_tokens"),
                    elapsed_ms=int((perf_counter() - started) * 1000),
                )
            )

        for attempt in range(2):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise S4StreamError("S4 阶段预算已耗尽", failure_code="S4_STAGE_TIMEOUT")
            try:
                timeout = httpx.Timeout(
                    connect=min(policy.connect_timeout_seconds, remaining),
                    read=min(policy.idle_seconds, remaining),
                    write=min(policy.connect_timeout_seconds, remaining),
                    pool=min(policy.connect_timeout_seconds, remaining),
                )
                async with self._client.stream(
                    "POST", "chat/completions", json=payload, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    event_lines: list[str] = []
                    iterator = response.aiter_lines().__aiter__()
                    while True:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise S4StreamError(
                                "S4 阶段预算已耗尽", failure_code="S4_STAGE_TIMEOUT"
                            )
                        wait_budget = min(
                            remaining,
                            policy.idle_seconds if content else policy.first_content_seconds,
                        )
                        try:
                            line = await asyncio.wait_for(iterator.__anext__(), wait_budget)
                        except StopAsyncIteration:
                            break
                        except TimeoutError as exc:
                            code = "S4_STREAM_IDLE_TIMEOUT" if content else "S4_FIRST_EVENT_TIMEOUT"
                            raise S4StreamError("S4 流等待正文超时", failure_code=code) from exc
                        if line == "":
                            data = self._sse_event(event_lines)
                            event_lines = []
                            if data is None:
                                continue
                            if data == "[DONE]":
                                saw_done = True
                                break
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError as exc:
                                raise S4StreamError(
                                    "S4 流事件不是合法 JSON", failure_code="S4_INVALID_RESPONSE"
                                ) from exc
                            if not isinstance(event, dict):
                                raise S4StreamError(
                                    "S4 流事件结构无效", failure_code="S4_INVALID_RESPONSE"
                                )
                            supplied = event.get("usage")
                            if isinstance(supplied, dict):
                                for key in ("prompt_tokens", "completion_tokens"):
                                    value = supplied.get(key)
                                    if type(value) is int and value >= 0:
                                        usage_values[key] = value
                            choices = event.get("choices")
                            if not isinstance(choices, list) or not choices:
                                await report()
                                continue
                            choice = choices[0]
                            if not isinstance(choice, dict):
                                raise S4StreamError(
                                    "S4 choices 结构无效", failure_code="S4_INVALID_RESPONSE"
                                )
                            finish_reason = choice.get("finish_reason") or finish_reason
                            if finish_reason == "length":
                                raise S4StreamError(
                                    "S4 输出被截断", failure_code="S4_INVALID_RESPONSE"
                                )
                            delta = choice.get("delta")
                            piece = delta.get("content") if isinstance(delta, dict) else None
                            if isinstance(piece, str) and piece:
                                content += piece
                                chunks += 1
                                if len(content) > policy.max_output_tokens * 8:
                                    raise S4StreamError(
                                        "S4 输出超过安全上限", failure_code="S4_INVALID_RESPONSE"
                                    )
                            await report()
                        elif line.startswith(":"):
                            # SSE 注释是心跳，不延长总预算，也不算正文进展。
                            continue
                        else:
                            event_lines.append(line)
                    if event_lines and not saw_done:
                        data = self._sse_event(event_lines)
                        if data and data != "[DONE]":
                            raise S4StreamError(
                                "S4 流在事件边界前结束", failure_code="S4_STREAM_INTERRUPTED"
                            )
                    if not saw_done:
                        raise S4StreamError(
                            "S4 流未收到结束帧", failure_code="S4_STREAM_INTERRUPTED"
                        )
                    if not content.strip():
                        raise S4StreamError("S4 流没有正文", failure_code="S4_INVALID_RESPONSE")
                    try:
                        result = schema.model_validate_json(content)
                    except Exception as exc:
                        raise S4StreamError(
                            "S4 完整正文未通过结构校验", failure_code="S4_INVALID_RESPONSE"
                        ) from exc
                    if on_progress is not None:
                        await on_progress(
                            StreamUsage(
                                output_chars=len(content),
                                content_chunks=chunks,
                                usage_status="available" if usage_values else "unavailable",
                                prompt_tokens=usage_values.get("prompt_tokens"),
                                completion_tokens=usage_values.get("completion_tokens"),
                                elapsed_ms=int((perf_counter() - started) * 1000),
                            )
                        )
                    return result
            except S4StreamError:
                raise
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise S4StreamError(
                        "S4 服务拒绝请求", failure_code="S4_PROVIDER_UNAVAILABLE"
                    ) from exc
                if content or attempt:
                    raise S4StreamError(
                        "S4 服务暂时不可用", failure_code="S4_PROVIDER_UNAVAILABLE"
                    ) from exc
            except httpx.TransportError as exc:
                if content or attempt:
                    raise S4StreamError(
                        "S4 流连接失败", failure_code="S4_STREAM_INTERRUPTED"
                    ) from exc
            if deadline - loop.time() <= 0:
                raise S4StreamError("S4 阶段预算已耗尽", failure_code="S4_STAGE_TIMEOUT")
            await asyncio.sleep(min(0.25, max(0.0, deadline - loop.time())))
        raise S4StreamError("S4 流调用失败", failure_code="S4_PROVIDER_UNAVAILABLE")
