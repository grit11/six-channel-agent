"""运行进度上下文：将节点/检索/模型的真实事件交给注入的异步接收器。

上下文仅存进程内回调，不进入 checkpoint；无病例、查询原文、模型正文或隐藏推理。
独立 QA/原型未绑定会诊时不会写入临床事件序列。
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import perf_counter

from backend.app.schemas.events import StreamEventName

ProgressSink = Callable[[StreamEventName, dict[str, object]], Awaitable[None]]
_sink: ContextVar[ProgressSink | None] = ContextVar("consultation_progress", default=None)
_stage: ContextVar[str | None] = ContextVar("consultation_stage", default=None)


@asynccontextmanager
async def progress_scope(sink: ProgressSink) -> AsyncIterator[None]:
    """绑定单轮会诊接收器；嵌套调用与并发任务各自隔离。"""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


@asynccontextmanager
async def stage_scope(stage: str) -> AsyncIterator[None]:
    """标注当前协议节点，检索/模型事件只附固定阶段名。"""
    token = _stage.set(stage)
    try:
        yield
    finally:
        _stage.reset(token)


async def emit_progress(event: StreamEventName, payload: dict[str, object]) -> None:
    """等待事件落库后继续执行；持久化失败不能假装进度已发布。"""
    sink = _sink.get()
    if sink is not None:
        await sink(event, {**payload, "stage": _stage.get()})


@asynccontextmanager
async def model_scope(capability: str) -> AsyncIterator[None]:
    """包围模型调用与响应校验，失败不提前报告完成，不记录正文。"""
    started = perf_counter()
    await emit_progress("model_started", {"capability": capability})
    try:
        yield
    except Exception:
        await emit_progress(
            "model_failed",
            {"capability": capability, "duration_ms": int((perf_counter() - started) * 1000)},
        )
        raise
    else:
        await emit_progress(
            "model_completed",
            {"capability": capability, "duration_ms": int((perf_counter() - started) * 1000)},
        )
