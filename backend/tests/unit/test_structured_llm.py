"""结构化 LLM 协议与解析约束测试；用 HTTP 替身验证实际适配，不调用付费模型。"""

import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.llm import HTTPStructuredLLM


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
