"""结构化 LLM 协议与解析约束测试；用 HTTP 替身验证实际适配，不调用付费模型。"""

import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError, S4StreamError
from backend.app.core.llm import HTTPStructuredLLM, build_llm_provider_options
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


def glm_settings(model: str = "glm-5.3-flash") -> Settings:
    """构造智谱兼容端点配置，验证可选 OpenAI 扩展不会发送。"""
    return Settings(
        _env_file=None,
        llm_base_url="https://open.bigmodel.cn/api/paas/v4/",
        llm_model=model,
        llm_reasoning_effort="low",
        llm_max_retries=0,
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
        assert "thinking" not in payload
        assert "reasoning_effort" not in payload
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
@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"])
async def test_glm_s4_stream_uses_required_thinking_mode(model: str) -> None:
    """GLM-5.3 全系列只能开启思考，使用 low 并验证最终 JSON。"""
    body = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"{\\"value\\":7}"},'
        '"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        """确保 GLM 不收到未声明的 response_format/stream_options。"""
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert "response_format" not in payload
        assert "stream_options" not in payload
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "low"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(
        base_url="https://open.bigmodel.cn/api/paas/v4/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPStructuredLLM(glm_settings(model), client).ainvoke_structured_stream(
            [],
            Payload,
            policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
        )
    assert result == Payload(value=7)


@pytest.mark.asyncio
async def test_other_glm_models_do_not_receive_unverified_thinking_options() -> None:
    """旧 GLM 型号及未知模型不根据端点域名猜测思考参数能力。"""
    async with httpx.AsyncClient() as client:
        adapter = HTTPStructuredLLM(glm_settings("glm-4-flash-250414"), client)
        assert build_llm_provider_options(adapter._settings) == {}


@pytest.mark.asyncio
async def test_glm_s4_stream_reads_content_parts_without_reasoning_content() -> None:
    """智谱可能以内容片段数组返回正文；解析器只合并 text，不采集隐藏推理。"""
    body = (
        'data: {"choices":[{"delta":{"content":[{"type":"text","text":"{\\"value\\":"}]}}]}\n\n'
        'data: {"choices":[{"delta":{"content":[{"type":"text","text":"7}"}]},'
        '"finish_reason":"stop","reasoning_content":"hidden"}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        """返回分段正文事件，确认请求使用智谱支持的思考模式。"""
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "low"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(
        base_url="https://open.bigmodel.cn/api/paas/v4/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPStructuredLLM(glm_settings(), client).ainvoke_structured_stream(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "text", "oversized"])
async def test_s4_http_error_retains_safe_metadata_without_body_or_retry(kind: str) -> None:
    """400 错误保留数值错误码和参数名，日志不能携带响应回显的密钥或病历。"""
    calls = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        """模拟 GLM 参数拒绝及不能解析的网关错误。"""
        nonlocal calls
        calls += 1
        if kind == "text":
            return httpx.Response(400, text="private-patient sk-private")
        if kind == "oversized":
            return httpx.Response(400, content=b"private" * 4096)
        return httpx.Response(
            400,
            json={"error": {"code": "1210", "message": "thinking: private-patient sk-private"}},
        )

    async with httpx.AsyncClient(
        base_url="https://open.bigmodel.cn/api/paas/v4/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(S4StreamError) as caught:
            await HTTPStructuredLLM(glm_settings("glm-5.3"), client).ainvoke_structured_stream(
                [],
                Payload,
                policy=S4StreamPolicy(total_seconds=2, first_content_seconds=1, idle_seconds=1),
            )
    details = caught.value.details
    assert calls == 1
    assert details["http_status"] == 400
    assert details["thinking_type"] == "enabled"
    assert details["reasoning_effort"] == "low"
    if kind == "json":
        assert details["provider_error_code"] == "1210"
        assert details["provider_error_param"] == "thinking"
    else:
        assert "provider_error_code" not in details
    assert "private-patient" not in str(details)
    assert "sk-private" not in str(details)
