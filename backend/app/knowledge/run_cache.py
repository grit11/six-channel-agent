"""单轮知识缓存：依赖调用方建立作用域，按知识与事实签名隔离，不跨患者共享。"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeVar, cast

T = TypeVar("T")


@dataclass
class RunCache:
    """只保存本轮已验证来源的副本及命中计数，不进入 checkpoint。"""

    values: dict[tuple[str, ...], object] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


_active: ContextVar[RunCache | None] = ContextVar("knowledge_run_cache", default=None)


@contextmanager
def run_cache_scope() -> Iterator[RunCache]:
    """为单次逻辑运行创建隔离缓存，嵌套保持同轮，结束后清理。"""
    existing = _active.get()
    if existing is not None:
        yield existing
        return
    cache = RunCache()
    token = _active.set(cache)
    try:
        yield cache
    finally:
        cache.values.clear()
        _active.reset(token)


def cached(key: tuple[str, ...], expected: type[T]) -> T | None:
    """读取深拷贝，禁止节点修改共享证据或其他候选条件。"""
    cache = _active.get()
    if cache is None:
        return None
    if key not in cache.values:
        cache.misses += 1
        return None
    cache.hits += 1
    return cast(T, deepcopy(cache.values[key]))


def cache_metrics() -> dict[str, int]:
    """投影当前轮的命中计数，不暴露患者事实或来源正文。"""
    cache = _active.get()
    return {"cache_hits": cache.hits if cache else 0, "cache_misses": cache.misses if cache else 0}


def remember(key: tuple[str, ...], value: object) -> None:
    """仅在已验证后写入当前轮缓存，外部独立查询不留患者状态。"""
    cache = _active.get()
    if cache is not None:
        cache.values[key] = deepcopy(value)
