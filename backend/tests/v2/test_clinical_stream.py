"""S1/S2 传输等价、完整性与故障测试；使用合成输入，不调用付费模型。"""

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from backend.app.core.config import Settings
from backend.app.core.llm import ClinicalStageStreamError, HTTPStructuredLLM


class Output(BaseModel):
    """合成协议结果，不代表医学判断。"""

    value: int


def sse(content: str, done: bool = True, finish: str = "stop") -> bytes:
    """构造 CRLF、多行 data 和 usage-only 标准事件。"""
    event = json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": finish}]})
    return (
        ": heartbeat\r\n\r\ndata: "
        + event
        + "\r\n\r\n"
        + 'data: {"choices":[],\r\ndata: "usage":{"prompt_tokens":2,"completion_tokens":3}}\r\n\r\n'
        + ("data: [DONE]\r\n\r\n" if done else "")
    ).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["S1", "S2"])
async def test_messages_and_schema_identical(stage: str) -> None:
    """只有传输字段变化，医学消息、模型、温度与 Schema 逐字一致。"""
    captured: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """记录合成请求，不记录用户病例。"""
        body = json.loads(request.content)
        captured.append(body)
        if body["stream"]:
            return httpx.Response(200, content=sse('{"value":7}'))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value":7}'}, "finish_reason": "stop"}]},
        )

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handle)
    ) as client:
        llm = HTTPStructuredLLM(
            Settings(_env_file=None, llm_model="test", llm_max_retries=0), client
        )
        messages = [
            {"role": "system", "content": "完整冻结说明"},
            {"role": "user", "content": "完整合成原文"},
        ]
        assert await llm.ainvoke_structured(
            messages, Output
        ) == await llm.ainvoke_clinical_structured_stream(messages, Output, stage=stage)
    assert captured[0]["messages"] == captured[1]["messages"]
    for field in ("model", "temperature", "response_format"):
        assert captured[0][field] == captured[1][field]
    assert "max_tokens" not in captured[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,done,finish,code",
    [
        ('{"value":7}', False, "stop", "STREAM_INTERRUPTED"),
        ('{"value":7}', True, "length", "RESPONSE_INVALID"),
        ('{"value":', True, "stop", "RESPONSE_INVALID"),
    ],
)
async def test_incomplete_never_retried(content: str, done: bool, finish: str, code: str) -> None:
    """半截流不得提交或整请求自动重发。"""
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """模拟请求已到达供应商后的失败。"""
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=sse(content, done, finish))

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handle)
    ) as client:
        llm = HTTPStructuredLLM(Settings(_env_file=None), client)
        with pytest.raises(ClinicalStageStreamError) as caught:
            await llm.ainvoke_clinical_structured_stream(
                [{"role": "user", "content": "test"}], Output, stage="S1"
            )
    assert caught.value.details["failure_code"] == "S1_" + code
    assert calls == 1


@pytest.mark.asyncio
async def test_cancel_propagates() -> None:
    """取消必须传播，不伪造 JSON 或业务成功。"""

    async def handle(request: httpx.Request) -> httpx.Response:
        """模拟被取消的模型请求。"""
        raise asyncio.CancelledError

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await HTTPStructuredLLM(
                Settings(_env_file=None), client
            ).ainvoke_clinical_structured_stream(
                [{"role": "user", "content": "test"}], Output, stage="S2"
            )
