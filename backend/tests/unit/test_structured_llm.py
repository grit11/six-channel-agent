"""结构化 LLM 协议与解析约束测试；用 HTTP 替身验证实际适配，不调用付费模型。"""

import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError, S4StreamError
from backend.app.core.llm import HTTPStructuredLLM
from backend.app.core.ports import S4StreamPolicy


class Payload(BaseModel):
    """测试协议契约，不包含医学判断。"""

    model_config = ConfigDict(extra="forbid")
    value: int


def settings(retries: int = 0) -> Settings:
    """不读取项目密钥的模型测试配置。"""
    return Settings(
        _env_file=None,
        llm_base_url="https://model.test/v1",
        llm_model="test",
        llm_max_retries=retries,
    )


def response(content: str, finish: str = "stop") -> httpx.Response:
    """构造兼容服务响应；隐藏字段只用于验证它不会出现在返回结果。"""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {
                        "content": content,
                        "reasoning_content": "private hidden field",
                    },
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_structured_adapter_retries_schema_and_ignores_hidden_fields() -> None:
    """第一次 Schema 错误允许有限重试，正常响应只返回指定对象。"""
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """检查路径、模型和 JSON 输出约束，正文不依赖真实网络。"""
        nonlocal attempts
        attempts += 1
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "test"
        assert payload["response_format"] == {"type": "json_object"}
        return response('{"value":"invalid"}' if attempts == 1 else '{"value":7}')

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPStructuredLLM(settings(1), client).ainvoke_structured(
            [{"role": "user", "content": "protocol input"}], Payload
        )
    assert result == Payload(value=7)
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,finish", [(401, "stop"), (200, "length")])
async def test_permanent_http_error_or_truncated_content_fails(status: int, finish: str) -> None:
    """鉴权永久错误不重试，截断 JSON 不得作为完整对象成功返回。"""
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """计数服务请求，避免重试造成无限调用。"""
        nonlocal attempts
        attempts += 1
        return httpx.Response(status) if status != 200 else response('{"value":7}', finish)

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ProviderUnavailableError):
            await HTTPStructuredLLM(settings(), client).ainvoke_structured([], Payload)
    assert attempts == 1


@pytest.mark.asyncio
async def test_s4_stream_parses_delta_usage_and_done() -> None:
    """S4 解析增量正文、usage-only 事件和结束帧，不返回隐藏字段。"""
    body = (
        'data: {"choices":[{"delta":{"content":"{\\"value\\":"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"7}"},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        """返回标准 Chat SSE，验证请求确实启用 stream。"""
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPStructuredLLM(settings(), client).ainvoke_structured_stream(
            [],
            Payload,
            policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
        )
    assert result == Payload(value=7)


@pytest.mark.asyncio
async def test_s4_stream_does_not_retry_after_partial_content() -> None:
    """正文已经产生后提前断流只能失败，不能重发整份请求。"""
    attempts = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        """返回没有 DONE 的不完整流。"""
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"{"}}]}\n\n',
        )

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(Exception) as caught:
            await HTTPStructuredLLM(settings(), client).ainvoke_structured_stream(
                [],
                Payload,
                policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
            )
    assert getattr(caught.value, "failure_code", None) == "S4_STREAM_INTERRUPTED"
    assert attempts == 1


@pytest.mark.asyncio
async def test_s4_stream_handles_crlf_multiline_data_and_usage_only() -> None:
    """兼容 Windows CRLF 和一个事件拆成多行 data 的标准 SSE。"""
    body = (
        'data: {"choices":[{"delta":{"content":"{\\"value\\":7}"}}]}\r\n\r\n'
        'data: {"choices":[],\r\n'
        'data: "usage":{"prompt_tokens":3,"completion_tokens":1}}\r\n\r\n'
        "data: [DONE]\r\n\r\n"
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        """返回含 CRLF 的合成 SSE。"""
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPStructuredLLM(settings(), client).ainvoke_structured_stream(
            [],
            Payload,
            policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
        )
    assert result == Payload(value=7)


@pytest.mark.asyncio
async def test_s4_stream_rejects_finish_length_without_retry() -> None:
    """模型输出达到上限时必须失败，不能把截断 JSON 当作成功。"""
    attempts = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        """返回 finish_reason=length。"""
        nonlocal attempts
        attempts += 1
        body = (
            'data: {"choices":[{"delta":{"content":"{\\"value\\":1"},'
            '"finish_reason":"length"}]}\n\n'
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(S4StreamError) as caught:
            await HTTPStructuredLLM(settings(), client).ainvoke_structured_stream(
                [],
                Payload,
                policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
            )
    assert caught.value.failure_code == "S4_INVALID_RESPONSE"
    assert attempts == 1


@pytest.mark.asyncio
async def test_s4_stream_classifies_unsupported_request() -> None:
    """供应商拒绝 stream/response_format/max_tokens 时返回兼容性失败分类。"""

    async def handle(request: httpx.Request) -> httpx.Response:
        """用 400 模拟供应商不支持流式参数。"""
        return httpx.Response(400, json={"error": {"message": "stream not supported"}})

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(S4StreamError) as caught:
            await HTTPStructuredLLM(settings(), client).ainvoke_structured_stream(
                [],
                Payload,
                policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
            )
    assert caught.value.failure_code == "S4_PROVIDER_UNAVAILABLE"
    assert caught.value.details["http_status"] == 400
