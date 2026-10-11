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
        async with progress_scope(sink_first), stage_scope("eight_principles"):
            with pytest.raises(ValueError):
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
    assert [item[0] for item in first] == ["model_started", "model_failed"]
    assert [item[0] for item in second] == ["model_started", "model_completed"]
    assert all(item[1]["stage"] == "eight_principles" for item in first)
    assert all(item[1]["stage"] == "channel_candidates" for item in second)
    assert "内部响应" not in str(first)
    await emit_progress("stage_started", {"stage": "outside"})
    assert len(first) == len(second) == 2
