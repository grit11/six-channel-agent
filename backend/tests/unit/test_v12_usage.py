"""Provider usage 回归：HTTP 替身验证重试真实次数与缺失报告，不请求真实模型。"""

import httpx
import pytest
from eval.system_eval import usage_summary

from backend.app.core.config import Settings
from backend.app.core.llm import HTTPStructuredLLM
from backend.app.observability.progress import progress_scope
from backend.app.schemas.events import StreamEventName


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 11, "completion_tokens": 5},
        None,
        {"prompt_tokens": True, "completion_tokens": 2},
    ],
)
async def test_provider_usage_preserves_available_and_missing(usage: object) -> None:
    """原生整数 usage 才可用于成本；缺失与布尔异常不可估算补齐。"""
    events: list[dict[str, object]] = []

    async def sink(event: StreamEventName, payload: dict[str, object]) -> None:
        """仅保存安全监控字段。"""
        events.append({"event": event, **payload})

    def handle(request: httpx.Request) -> httpx.Response:
        """模拟兼容响应，没有患者数据。"""
        return httpx.Response(
            200,
            json={
                "usage": usage,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "ok", "reasoning_content": "private"},
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None, llm_model="fixture", llm_base_url="https://model.test", llm_max_retries=0
    )
    async with (
        httpx.AsyncClient(
            base_url="https://model.test/", transport=httpx.MockTransport(handle)
        ) as client,
        progress_scope(sink),
    ):
        assert await HTTPStructuredLLM(settings, client).ainvoke_text([]) == "ok"
    summary = usage_summary({"events": events})
    assert summary["llm_calls"] == 1
    assert summary["status"] == (
        "available"
        if isinstance(usage, dict) and type(usage["prompt_tokens"]) is int
        else "unavailable"
    )
    assert "private" not in str(events)
