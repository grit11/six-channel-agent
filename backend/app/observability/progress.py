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
_stage_started: ContextVar[float | None] = ContextVar("stage_started", default=None)


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
    """隔离阶段与单调时钟；仅自动执行耗时，不把人工等待计入推理。"""
    token = _stage.set(stage)
    clock_token = _stage_started.set(perf_counter())
    try:
        yield
    except Exception as exc:
        await emit_progress("stage_failed", {"error_type": type(exc).__name__})
        raise
    finally:
        _stage_started.reset(clock_token)
        _stage.reset(token)


async def emit_progress(event: StreamEventName, payload: dict[str, object]) -> None:
    """等待事件落库后继续执行；持久化失败不能假装进度已发布。"""
    sink = _sink.get()
    if sink is not None:
        public = {**payload, "stage": _stage.get()}
        started = _stage_started.get()
        if event in {"stage_completed", "stage_failed"} and started is not None:
            public["duration_ms"] = int((perf_counter() - started) * 1000)
        await sink(event, public)


@asynccontextmanager
async def model_scope(capability: str, *, attempt: int = 0) -> AsyncIterator[dict[str, object]]:
    """包围模型调用与响应校验，失败不提前报告完成，不记录正文。"""
    started = perf_counter()
    metadata: dict[str, object] = {"attempt": attempt, "usage_status": "unavailable"}
    await emit_progress("model_started", {"capability": capability, **metadata})
    try:
        yield metadata
    except Exception as exc:
        metadata["error_type"] = type(exc).__name__
        details = getattr(exc, "details", {})
        if isinstance(details, dict):
            for key in ("failure_code", "error_category", "http_status"):
                if details.get(key) is not None:
                    metadata[key] = details[key]
        metadata.setdefault(
            "error_category",
            (
                "schema_validation"
                if type(exc).__name__ == "ValidationError"
                else (
                    "http_status"
                    if hasattr(exc, "response")
                    else (
                        "network"
                        if type(exc).__module__.startswith("httpx")
                        else "invalid_response"
                    )
                )
            ),
        )
        response = getattr(exc, "response", None)
        if response is not None:
            metadata["http_status"] = response.status_code
        await emit_progress(
            "model_failed",
            {
                "capability": capability,
                "duration_ms": int((perf_counter() - started) * 1000),
                **metadata,
            },
        )
        raise
    else:
        await emit_progress(
            "model_completed",
            {
                "capability": capability,
                "duration_ms": int((perf_counter() - started) * 1000),
                **metadata,
            },
        )
