"""结构化 Chat HTTP 适配器。

职责：通过配置和注入的异步客户端调用兼容协议，有限重试并验证 Pydantic 输出。
边界：不含医学规则、不记录原文或隐藏推理，不负责业务写入；客户端由容器关闭。
"""

import asyncio
import hashlib
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
from backend.app.observability.progress import emit_progress, model_scope


class ClinicalStageStreamError(ProviderUnavailableError):
    """S1/S2 安全技术错误，禁止携带部分正文，不影响 S4 错误契约。"""

    def __init__(self, stage: str, category: str) -> None:
        """保存阶段失败码及可安全恢复的技术分类。"""
        super().__init__("临床模型传输未完成", details={"failure_code": f"{stage}_{category}"})


def validate_llm_config(settings: Settings) -> None:
    """启动时验证协议、服务地址与模型，允许明确配置的本地无密钥服务。"""
    if settings.llm_provider not in {"openai_compatible", "deepseek"}:
        raise ConfigurationError("LLM Provider 必须使用已支持的兼容 Chat 协议")
    if not settings.llm_base_url.startswith(("https://", "http://")) or not settings.llm_model:
        raise ConfigurationError("请配置 LLM HTTP(S) 地址和模型名")


def build_llm_provider_options(settings: Settings) -> dict[str, object]:
    """在初始化边界统一选择供应商参数，供临床适配器与 QA 工厂复用。

    GLM-5.3 系列只允许开启思考，强度来自显式配置；空值保留供应商默认。
    不根据域名向其他模型猜测注入参数，避免切换 API 后破坏原有请求。
    """
    model = settings.llm_model.lower()
    if settings.llm_reasoning_effort and (model == "glm-5.3" or model.startswith("glm-5.3-")):
        return {
            "thinking": {"type": "enabled"},
            "reasoning_effort": settings.llm_reasoning_effort,
        }
    return {}


class HTTPStructuredLLM:
    """兼容 Chat Completions 的结构化适配器，不绑定厂商 SDK 或模型。"""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """保存配置和容器持有的客户端，密钥只由工厂放入请求头。"""
        self._settings = settings
        self._client = client
        self._provider_options = build_llm_provider_options(settings)

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
        payload.update(self._provider_options)
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

    def clinical_deadline(self, stage: str) -> float:
        """由原调用层共享 S1 来源修复预算，不将修复视作新阶段。"""
        return asyncio.get_running_loop().time() + (
            self._settings.s1_total_seconds if stage == "S1" else self._settings.s2_total_seconds
        )

    @property
    def clinical_runtime_digest(self) -> str:
        """指纹绑定模型端点及医学资源，不含密钥，用于技术恢复版本校验。"""
        from pathlib import Path

        root = Path(__file__).parents[1] / "clinical_v2"
        values = [
            self._settings.llm_base_url,
            self._settings.llm_model,
            str(self._settings.llm_temperature),
            json.dumps(self._provider_options, sort_keys=True),
        ]
        for name in (
            "s1_clinical_extract.md",
            "s2_channel_localize.md",
            "six_channel_reference.md",
        ):
            values.append(hashlib.sha256((root / "prompts" / name).read_bytes()).hexdigest())
        return hashlib.sha256(json.dumps(values).encode()).hexdigest()

    async def ainvoke_clinical_structured_stream(
        self,
        messages: list[dict[str, str]],
        schema: type[StructuredResult],
        *,
        stage: str,
        deadline: float | None = None,
    ) -> StructuredResult:
        """增量 S1/S2 传输；不重用或修改 S4 路径，消息与原 Schema 插入逐字相同。"""
        if not self._settings.clinical_stream_enabled:
            return await self.ainvoke_structured(messages, schema)
        loop = asyncio.get_running_loop()
        deadline = deadline if deadline is not None else self.clinical_deadline(stage)
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
        }
        payload.update(self._provider_options)
        started = loop.time()
        for attempt in range(2):
            try:
                async with (
                    asyncio.timeout_at(deadline),
                    model_scope("clinical_stream", attempt=attempt) as usage,
                ):
                    usage["request_payload_digest"] = hashlib.sha256(
                        json.dumps(prepared, ensure_ascii=False).encode()
                    ).hexdigest()
                    usage["prompt_digest"] = hashlib.sha256(
                        messages[0]["content"].encode()
                    ).hexdigest()
                    content, chunks, finish, done = "", 0, None, False
                    progress_at, last_content = started, started
                    timeout = httpx.Timeout(
                        connect=min(10, max(0.001, deadline - loop.time())),
                        read=None,
                        write=10,
                        pool=10,
                    )
                    async with self._client.stream(
                        "POST", "chat/completions", json=payload, timeout=timeout
                    ) as response:
                        response.raise_for_status()
                        iterator = response.aiter_lines().__aiter__()
                        lines: list[str] = []
                        while True:
                            # 注释、空增量和隐藏推理不能延长首正文或正文空闲预算。
                            limit = (
                                last_content + self._settings.clinical_idle_seconds
                                if chunks
                                else started + self._settings.clinical_first_content_seconds
                            )
                            remaining = min(deadline, limit) - loop.time()
                            if remaining <= 0:
                                raise ClinicalStageStreamError(
                                    stage,
                                    (
                                        "STAGE_TIMEOUT"
                                        if loop.time() >= deadline
                                        else (
                                            "STREAM_INTERRUPTED"
                                            if chunks
                                            else "FIRST_EVENT_TIMEOUT"
                                        )
                                    ),
                                )
                            try:
                                line = await asyncio.wait_for(iterator.__anext__(), remaining)
                            except StopAsyncIteration:
                                break
                            except TimeoutError as exc:
                                raise ClinicalStageStreamError(
                                    stage,
                                    (
                                        "STAGE_TIMEOUT"
                                        if loop.time() >= deadline
                                        else (
                                            "STREAM_INTERRUPTED"
                                            if chunks
                                            else "FIRST_EVENT_TIMEOUT"
                                        )
                                    ),
                                ) from exc
                            if line:
                                if not line.startswith(":"):
                                    lines.append(line)
                                continue
                            data = self._sse_event(lines)
                            lines = []
                            if data is None:
                                continue
                            if data == "[DONE]":
                                done = True
                                break
                            event = json.loads(data)
                            if not isinstance(event, dict):
                                raise ValueError("无效 SSE")
                            supplied = event.get("usage")
                            if isinstance(supplied, dict) and all(
                                type(supplied.get(k)) is int and supplied[k] >= 0
                                for k in ("prompt_tokens", "completion_tokens")
                            ):
                                usage.update(
                                    {k: supplied[k] for k in ("prompt_tokens", "completion_tokens")}
                                )
                                usage["usage_status"] = "available"
                            choices = event.get("choices")
                            if choices == []:
                                continue
                            if (
                                not isinstance(choices, list)
                                or len(choices) != 1
                                or not isinstance(choices[0], dict)
                            ):
                                raise ValueError("无效 choices")
                            choice = choices[0]
                            finish = choice.get("finish_reason") or finish
                            delta = choice.get("delta", {})
                            if not isinstance(delta, dict) or delta.get("refusal"):
                                raise ValueError("响应拒绝")
                            piece = delta.get("content")
                            if piece is not None and not isinstance(piece, str):
                                raise ValueError("无效正文类型")
                            if piece:
                                if not chunks:
                                    usage["ttft_ms"] = int((loop.time() - started) * 1000)
                                content += piece
                                chunks += 1
                                last_content = loop.time()
                            if loop.time() - progress_at >= 2:
                                progress_at = loop.time()
                                await emit_progress(
                                    "model_started",
                                    {
                                        "capability": "clinical_stream",
                                        "elapsed_ms": int((loop.time() - started) * 1000),
                                        "first_content_seen": bool(chunks),
                                        "output_chars": len(content),
                                        "content_chunks": chunks,
                                        "attempt": attempt,
                                    },
                                )
                        if not done:
                            raise ClinicalStageStreamError(stage, "STREAM_INTERRUPTED")
                        if finish != "stop" or not content.strip():
                            raise ValueError("响应未正常完成")
                        result = schema.model_validate_json(content)
                        usage.update(content_chunks=chunks, output_chars=len(content))
                        return result
            except asyncio.CancelledError:
                raise
            except ClinicalStageStreamError:
                raise
            except TimeoutError as exc:
                raise ClinicalStageStreamError(stage, "STAGE_TIMEOUT") from exc
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                # 只重试确定尚未发送的连接建立错误；ReadTimeout/429/5xx 不推断未生成。
                if attempt or loop.time() >= deadline:
                    raise ClinicalStageStreamError(stage, "PROVIDER_UNAVAILABLE") from exc
            except httpx.HTTPStatusError as exc:
                raise ClinicalStageStreamError(stage, "PROVIDER_UNAVAILABLE") from exc
            except httpx.TransportError as exc:
                raise ClinicalStageStreamError(stage, "STREAM_INTERRUPTED") from exc
            except (ValueError, ValidationError, TypeError) as exc:
                raise ClinicalStageStreamError(stage, "RESPONSE_INVALID") from exc
        raise ClinicalStageStreamError(stage, "PROVIDER_UNAVAILABLE")

    @staticmethod
    def _sse_event(lines: list[str]) -> str | None:
        """合并一个 SSE 事件的多行 data；注释和无 data 事件被忽略。"""
        values = [line[5:].lstrip() for line in lines if line.startswith("data:")]
        return "\n".join(values) if values else None

    def _s4_stream_capabilities(self) -> tuple[bool, bool]:
        """根据兼容端点选择可选参数，保留标准 OpenAI 端点的现有行为。

        保留已有 GLM 参数回避策略，避免把部分兼容网关的可选扩展当作强制能力。
        正文仍由提示词约束并由本地 Pydantic Schema 校验；思考能力另按型号处理。
        """
        endpoint = self._settings.llm_base_url.lower()
        model = self._settings.llm_model.lower()
        glm_endpoint = "bigmodel.cn" in endpoint or model.startswith("glm-")
        return (not glm_endpoint, not glm_endpoint)

    @staticmethod
    async def _s4_http_error_metadata(response: httpx.Response) -> dict[str, str]:
        """只保留错误码与白名单参数名，不把供应商原文或其回显的病历写入日志。"""
        body = bytearray()
        async for piece in response.aiter_bytes():
            if len(body) + len(piece) > 16384:
                return {}
            body.extend(piece)
        try:
            raw = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return {}
        error = raw.get("error") if isinstance(raw, dict) else None
        if not isinstance(error, dict):
            return {}
        metadata: dict[str, str] = {}
        code = error.get("code")
        if (
            (type(code) is int or isinstance(code, str))
            and str(code).isascii()
            and str(code).isdigit()
            and len(str(code)) <= 12
        ):
            metadata["provider_error_code"] = str(code)
        # 错误 message 可能包含密钥、患者输入，只提取已知协议参数名，绝不转发正文。
        param = error.get("param")
        message = error.get("message")
        for name in (
            "thinking",
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "response_format",
            "stream_options",
            "messages",
            "model",
            "stream",
        ):
            if param == name or (isinstance(message, str) and name in message.lower()):
                metadata["provider_error_param"] = name
                break
        return metadata

    @staticmethod
    def _s4_content_piece(choice: dict[str, object]) -> str | None:
        """读取兼容 Chat SSE 的正文增量，不读取或保存隐藏推理字段。

        部分兼容端点会把 content 作为多模态片段数组返回；只提取其中的 text，
        丢弃 reasoning_content、tool_calls 等非正文字段。
        """

        def text_piece(value: object) -> str | None:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "".join(parts) or None
            return None

        delta = choice.get("delta")
        if isinstance(delta, dict):
            piece = text_piece(delta.get("content"))
            if piece is not None:
                return piece
        message = choice.get("message")
        if isinstance(message, dict):
            piece = text_piece(message.get("content"))
            if piece is not None:
                return piece
        return text_piece(choice.get("content"))

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
        response_format_enabled, stream_options_enabled = self._s4_stream_capabilities()
        payload: dict[str, object] = {
            "model": self._settings.llm_model,
            "messages": prepared,
            "temperature": self._settings.llm_temperature,
            "stream": True,
            "max_tokens": policy.max_output_tokens,
        }
        if stream_options_enabled:
            payload["stream_options"] = {"include_usage": True}
        if response_format_enabled:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self._provider_options)
        loop = asyncio.get_running_loop()
        started = perf_counter()
        deadline = loop.time() + policy.total_seconds
        last_progress = 0.0
        content = ""
        chunks = 0
        finish_reason: str | None = None
        usage_values: dict[str, int] = {}
        saw_done = False
        http_error_metadata: dict[str, str] = {}

        def stream_error(
            message: str,
            failure_code: str,
            *,
            error_category: str | None = None,
            http_status: int | None = None,
        ) -> S4StreamError:
            """只附加传输诊断元数据，避免错误日志丢失 Provider 兼容性线索。"""
            return S4StreamError(
                message,
                failure_code=failure_code,
                http_status=http_status,
                error_category=error_category,
                provider=self._settings.llm_provider,
                model=self._settings.llm_model,
                response_format_enabled=response_format_enabled,
                stream_options_enabled=stream_options_enabled,
                finish_reason=finish_reason,
                saw_done=saw_done,
                content_chars=len(content),
                thinking_type=("enabled" if "thinking" in payload else None),
                reasoning_effort=(
                    self._settings.llm_reasoning_effort if "reasoning_effort" in payload else None
                ),
                provider_error_code=http_error_metadata.get("provider_error_code"),
                provider_error_param=http_error_metadata.get("provider_error_param"),
            )

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
                raise stream_error(
                    "S4 阶段预算已耗尽", "S4_STAGE_TIMEOUT", error_category="timeout"
                )
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
                    if response.is_error:
                        async with asyncio.timeout(max(0.001, deadline - loop.time())):
                            http_error_metadata = await self._s4_http_error_metadata(response)
                    response.raise_for_status()
                    event_lines: list[str] = []
                    iterator = response.aiter_lines().__aiter__()
                    while True:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise stream_error(
                                "S4 阶段预算已耗尽", "S4_STAGE_TIMEOUT", error_category="timeout"
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
                            raise stream_error(
                                "S4 流等待正文超时", code, error_category="timeout"
                            ) from exc
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
                                raise stream_error(
                                    "S4 流事件不是合法 JSON",
                                    "S4_INVALID_RESPONSE",
                                    error_category="event_json",
                                ) from exc
                            if not isinstance(event, dict):
                                raise stream_error(
                                    "S4 流事件结构无效",
                                    "S4_INVALID_RESPONSE",
                                    error_category="event_shape",
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
                                raise stream_error(
                                    "S4 choices 结构无效",
                                    "S4_INVALID_RESPONSE",
                                    error_category="choices_shape",
                                )
                            finish_reason = choice.get("finish_reason") or finish_reason
                            piece = self._s4_content_piece(choice)
                            if piece:
                                content += piece
                                chunks += 1
                                if len(content) > policy.max_output_tokens * 8:
                                    raise stream_error(
                                        "S4 输出超过安全上限",
                                        "S4_INVALID_RESPONSE",
                                        error_category="output_limit",
                                    )
                            if finish_reason == "length":
                                raise stream_error(
                                    "S4 输出被截断",
                                    "S4_INVALID_RESPONSE",
                                    error_category="truncation",
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
                            raise stream_error(
                                "S4 流在事件边界前结束",
                                "S4_STREAM_INTERRUPTED",
                                error_category="event_boundary",
                            )
                    if not saw_done:
                        raise stream_error(
                            "S4 流未收到结束帧",
                            "S4_STREAM_INTERRUPTED",
                            error_category="missing_done",
                        )
                    if not content.strip():
                        raise stream_error(
                            "S4 流没有正文", "S4_INVALID_RESPONSE", error_category="empty_content"
                        )
                    try:
                        result = schema.model_validate_json(content)
                    except Exception as exc:
                        raise stream_error(
                            "S4 完整正文未通过结构校验",
                            "S4_INVALID_RESPONSE",
                            error_category="schema_validation",
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
                    raise stream_error(
                        "S4 服务拒绝请求",
                        "S4_PROVIDER_UNAVAILABLE",
                        http_status=exc.response.status_code,
                        error_category="http_status",
                    ) from exc
                if content or attempt:
                    raise stream_error(
                        "S4 服务暂时不可用",
                        "S4_PROVIDER_UNAVAILABLE",
                        http_status=exc.response.status_code,
                        error_category="http_status",
                    ) from exc
            except httpx.TransportError as exc:
                if content or attempt:
                    raise stream_error(
                        "S4 流连接失败",
                        "S4_STREAM_INTERRUPTED",
                        error_category="network",
                    ) from exc
            except TimeoutError as exc:
                raise stream_error(
                    "S4 流读取超时",
                    "S4_STREAM_IDLE_TIMEOUT" if content else "S4_FIRST_EVENT_TIMEOUT",
                    error_category="timeout",
                ) from exc
            if deadline - loop.time() <= 0:
                raise stream_error(
                    "S4 阶段预算已耗尽", "S4_STAGE_TIMEOUT", error_category="timeout"
                )
            await asyncio.sleep(min(0.25, max(0.0, deadline - loop.time())))
        raise stream_error("S4 流调用失败", "S4_PROVIDER_UNAVAILABLE", error_category="provider")
