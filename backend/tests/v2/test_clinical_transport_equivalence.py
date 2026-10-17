"""S1/S2 真实 DTO 与原始医学请求传输等价测试；冻结 Prompt，不调用在线模型。"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from backend.app.clinical_v2.channel_localization import localize_channels
from backend.app.clinical_v2.extraction import extract_clinical_facts
from backend.app.clinical_v2.schemas import ChannelLocalizationV2, S1ExtractionDraft
from backend.app.core.config import Settings
from backend.app.core.llm import ClinicalStageStreamError, HTTPStructuredLLM
from backend.tests.v2.test_clinical_stream import Output, sse


@pytest.mark.asyncio
async def test_actual_s1_s2_messages_schema_and_results_identical() -> None:
    """用同一预设医学 JSON 对比原非流式与流式的消息、Schema、事实 ID 和路由。"""
    captured: list[dict] = []
    draft = S1ExtractionDraft(
        findings=[
            {
                "local_id": "a",
                "concept": "发热",
                "polarity": "positive",
                "time_scope": "past",
                "source_text": "昨日发热",
            },
            {
                "local_id": "b",
                "concept": "发热",
                "polarity": "negative",
                "time_scope": "current",
                "source_text": "今日退热",
            },
            {
                "local_id": "c",
                "concept": "口渴",
                "polarity": "unknown",
                "time_scope": "current",
                "source_text": "口渴不明",
                "needs_clarification": True,
            },
        ]
    )
    localization = ChannelLocalizationV2(
        status="unknown",
        candidates=[],
        proceed_to_retrieval=False,
        clarification_questions=[],
        note="合成未决结果",
    )

    def handle(request: httpx.Request) -> httpx.Response:
        """同一响应 JSON，不让随机模型差异掩盖传输语义差异。"""
        payload = json.loads(request.content)
        captured.append(payload)
        body = (
            localization if "ChannelLocalizationV2" in payload["messages"][0]["content"] else draft
        ).model_dump_json()
        if payload["stream"]:
            return httpx.Response(200, content=sse(body))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": body}, "finish_reason": "stop"}],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handle)
    ) as client:
        settings = Settings(_env_file=None, llm_model="synthetic", llm_max_retries=0)
        legacy = HTTPStructuredLLM(
            settings.model_copy(update={"clinical_stream_enabled": False}), client
        )
        stream = HTTPStructuredLLM(settings, client)
        text = "昨日发热，今日退热，口渴不明"
        a = await extract_clinical_facts(text, legacy, terminology={"发热": "fever"})
        b = await extract_clinical_facts(text, stream, terminology={"发热": "fever"})
        assert a == b
        assert await localize_channels(a, legacy) == await localize_channels(b, stream)
    for first, second in ((captured[0], captured[1]), (captured[2], captured[3])):
        for name in ("messages", "model", "temperature", "response_format"):
            assert first[name] == second[name]
        assert "max_tokens" not in second
    assert captured[2]["messages"][-1]["content"] == a.model_dump_json()
    assert [f.polarity for f in b.findings] == ["positive", "negative", "unknown"]
    assert localization.status == "unknown" and not localization.proceed_to_retrieval


class Heartbeats(httpx.AsyncByteStream):
    """持续发送注释模拟无正文的上游，不允许心跳重置首内容预算。"""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """有限时间内只输出注释，测试客户端自己的首内容截止。"""
        for _ in range(100):
            await asyncio.sleep(0.01)
            yield b": heartbeat\n\n"


@pytest.mark.asyncio
async def test_heartbeats_do_not_extend_first_content_budget() -> None:
    """有 SSE 事件而无正文也必须超时，且不整请求重试。"""
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """只启动一次响应。"""
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=Heartbeats())

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handle)
    ) as client:
        settings = Settings(_env_file=None, clinical_first_content_seconds=0.03)
        with pytest.raises(ClinicalStageStreamError) as error:
            await HTTPStructuredLLM(settings, client).ainvoke_clinical_structured_stream(
                [{"role": "user", "content": "合成"}], Output, stage="S2"
            )
    assert error.value.details["failure_code"] == "S2_FIRST_EVENT_TIMEOUT"
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_http_failure_not_reissued(status: int) -> None:
    """已收到 HTTP 失败不推断服务器未生成；保留错误类别和单次请求。"""
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """模拟 provider 限流/不可用。"""
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ClinicalStageStreamError) as error:
            await HTTPStructuredLLM(
                Settings(_env_file=None), client
            ).ainvoke_clinical_structured_stream(
                [{"role": "user", "content": "合成"}], Output, stage="S1"
            )
    assert error.value.details["failure_code"] == "S1_PROVIDER_UNAVAILABLE"
    assert calls == 1
