"""进度上下文回归：模型校验失败不报完成、任务隔离及公开事件安全。"""

import asyncio

import pytest

from backend.app.observability.progress import (
    emit_progress,
    model_scope,
    progress_scope,
    stage_scope,
)
from backend.app.schemas.events import StreamEventName


@pytest.mark.asyncio
async def test_model_validation_failure_and_concurrent_scope_isolation() -> None:
    """实际模型失败产生 started/failed；其他会诊上下文不被阶段或回调污染。"""
    first: list[tuple[str, dict[str, object]]] = []
    second: list[tuple[str, dict[str, object]]] = []

    async def sink_first(event: StreamEventName, payload: dict[str, object]) -> None:
        """只接收第一会诊公开事件。"""
        first.append((event, payload))

    async def sink_second(event: StreamEventName, payload: dict[str, object]) -> None:
        """只接收第二会诊公开事件。"""
        second.append((event, payload))

    async def fail() -> None:
        """模拟返回内容未通过响应 Schema 校验。"""
        with pytest.raises(ValueError):
            async with progress_scope(sink_first), stage_scope("eight_principles"):
                async with model_scope("chat"):
                    await asyncio.sleep(0)
                    raise ValueError("内部响应不能输出")

    async def succeed() -> None:
        """模拟另一会诊在独立阶段完成模型调用。"""
        async with (
            progress_scope(sink_second),
            stage_scope("channel_candidates"),
            model_scope("embedding"),
        ):
            await asyncio.sleep(0)

    await asyncio.gather(fail(), succeed())
    assert [item[0] for item in first] == ["model_started", "model_failed", "stage_failed"]
    assert [item[0] for item in second] == ["model_started", "model_completed"]
    assert all(item[1]["stage"] == "eight_principles" for item in first)
    assert all(item[1]["stage"] == "channel_candidates" for item in second)
    assert "内部响应" not in str(first)
    await emit_progress("stage_started", {"stage": "outside"})
    assert first[-1][1]["duration_ms"] >= 0
    assert len(first) == 3 and len(second) == 2


async def test_stage_timing_and_usage_are_preserved() -> None:
    """阶段耗时独立于模型耗时，供应商真实 usage 保留；上下文退出不泄漏阶段。"""
    events: list[tuple[str, dict[str, object]]] = []

    async def sink(event: StreamEventName, payload: dict[str, object]) -> None:
        """收集注入的公开事件，不记录模型正文。"""
        events.append((event, payload))

    async with progress_scope(sink):
        async with stage_scope("s1_extract"):
            await emit_progress("stage_started", {})
            async with model_scope("chat") as usage:
                usage.update(usage_status="available", prompt_tokens=12, completion_tokens=8)
                await asyncio.sleep(0.01)
            await emit_progress("stage_completed", {})
        await emit_progress("stage_completed", {})
    assert events[2][1]["prompt_tokens"] == 12
    assert events[2][1]["completion_tokens"] == 8
    assert events[3][1]["duration_ms"] >= events[2][1]["duration_ms"]
    assert events[-1][1]["stage"] is None and "duration_ms" not in events[-1][1]
