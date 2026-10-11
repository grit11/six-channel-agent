"""LangGraph checkpoint 工厂，仅管理图执行状态，不替代业务 Repository。"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.core.config import Settings
from backend.app.core.exceptions import ConfigurationError, ProviderUnavailableError
from backend.app.core.logging import log_backend_event


def checkpoint_serializer() -> JsonPlusSerializer:
    """只允许业务图实际使用的 DTO 类型，禁止任意模块反序列化或 pickle 降级。"""
    modules = [
        ("backend.app.schemas.clinical", "ClinicalEvidence"),
        ("backend.app.schemas.evidence", "Evidence"),
        *[
            ("backend.app.schemas.reasoning", name)
            for name in (
                "EightPrinciplesState",
                "ChannelCandidate",
                "SyndromeRelationship",
                "FormulaCandidate",
                "DifferentialResult",
                "EvidenceGap",
                "ReasoningPath",
                "EvidenceGapContext",
                "EvidenceGapDecision",
                "SimilarCaseAssessment",
                "ReasoningStageSnapshot",
            )
        ],
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=modules, allowed_json_modules=modules)


@dataclass(frozen=True)
class CheckpointBackendStatus:
    """Checkpoint 实际后端和降级状态。"""

    backend: Literal["postgres", "sqlite"]
    degraded: bool
    detail: str


class CheckpointRuntime:
    """持有应用级 saver 上下文，保证关闭所有数据库连接。"""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        status: CheckpointBackendStatus,
        stack: AsyncExitStack,
    ) -> None:
        """绑定 saver、实际状态和资源清理栈。"""
        self.checkpointer = checkpointer
        self.status = status
        self._stack = stack

    async def close(self) -> None:
        """退出 saver 上下文，释放 SQLite 或 PostgreSQL 连接。"""
        await self._stack.aclose()


async def _open_sqlite(settings: Settings, reason: str) -> CheckpointRuntime:
    """创建持久化 SQLite saver；创建失败时保留异常链。"""
    stack = AsyncExitStack()
    try:
        path = Path(settings.langgraph_sqlite_path)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(path)))
        saver.serde = checkpoint_serializer()
        await saver.setup()
    except Exception as exc:
        await stack.aclose()
        raise ProviderUnavailableError("SQLite Checkpointer 初始化失败") from exc
    return CheckpointRuntime(saver, CheckpointBackendStatus("sqlite", True, reason), stack)


async def _open_postgres(settings: Settings) -> CheckpointRuntime:
    """连接并探测 PostgreSQL，初始化独立 checkpoint 表。"""
    if not settings.langgraph_database_url.get_secret_value():
        raise ConfigurationError("PostgreSQL Checkpointer 缺少连接配置")
    stack = AsyncExitStack()
    try:
        # 外部连接与 setup 均受超时限制，失败后先关闭连接才允许降级。
        async with asyncio.timeout(settings.postgres_connect_timeout_seconds):
            saver = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(
                    settings.langgraph_database_url.get_secret_value()
                )
            )
            saver.serde = checkpoint_serializer()
            await saver.setup()
            await saver.aget_tuple({"configurable": {"thread_id": "__startup_probe__"}})
    except Exception:
        await stack.aclose()
        raise
    return CheckpointRuntime(
        saver, CheckpointBackendStatus("postgres", False, "PostgreSQL 连接正常"), stack
    )


async def build_checkpoint_runtime(settings: Settings) -> CheckpointRuntime:
    """按配置选择 PostgreSQL 或 SQLite，禁止强制 postgres 时静默降级。"""
    if settings.checkpoint_backend == "sqlite":
        return await _open_sqlite(settings, "显式 SQLite 模式")
    try:
        return await _open_postgres(settings)
    except Exception as exc:
        if settings.checkpoint_backend == "postgres" or not settings.checkpoint_fallback_to_sqlite:
            raise ProviderUnavailableError("PostgreSQL Checkpointer 不可用，禁止降级") from exc
        # 不记录异常文本或连接 URL，避免数据库密码经日志泄露。
        log_backend_event("checkpoint_fallback", "sqlite", type(exc).__name__)
        return await _open_sqlite(settings, "PostgreSQL 未配置或不可用，已降级 SQLite")
