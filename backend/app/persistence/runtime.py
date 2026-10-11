"""可靠运行存储：事件、任务、请求收据与 checkpoint 补偿；不包含医学规则。

PostgreSQL 运行意图与业务快照同事务提交；memory 只是无重启持久性的开发替身。
图锁和业务锁分离，租约过期后仍必须获取图锁，防止旧进程同时写 checkpoint。
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from sqlalchemy import DateTime, Integer, cast, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.core.exceptions import InvalidBusinessRequestError, PersistenceError
from backend.app.persistence.models import RuntimeRecordModel
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.schemas.events import SSEEvent, StreamEventName

RuntimeWrite = tuple[str, str, dict[str, object]]
receipt_context: ContextVar[tuple[str, str] | None] = ContextVar("request_receipt", default=None)
operation_context: ContextVar[UUID | None] = ContextVar("operation_id", default=None)


class RuntimeStore(Protocol):
    """运行记录、事件页与跨进程互斥端口。"""

    def mutex(self, key: str) -> AbstractAsyncContextManager[None]:
        """返回资源互斥上下文，禁止嵌套获取相同键。"""

    async def read(self, kind: str, key: str) -> dict[str, object] | None:
        """读取独立记录。"""

    async def write(self, kind: str, key: str, payload: dict[str, object]) -> None:
        """提交运行记录，调用方负责状态转换的互斥。"""

    async def scan(self, kind: str, identifier: UUID | None = None) -> list[dict[str, object]]:
        """有界查询可恢复任务/补偿，终态历史不参与队列。"""

    async def events(self, identifier: UUID, after: int, limit: int = 100) -> list[SSEEvent]:
        """严格按会诊与序列分页读取。"""

    async def emit(
        self, identifier: UUID, turn: int, event: StreamEventName, payload: dict[str, object]
    ) -> None:
        """追加实际发生的公开进度事件。"""


def cursor_sequence(identifier: UUID, cursor: str | None) -> int:
    """校验游标归属和序列，非法游标不能静默回放其他会诊。"""
    if cursor is None:
        return 0
    parts = cursor.split(":")
    if len(parts) != 2 or parts[0] != str(identifier) or not parts[1].isdigit():
        raise InvalidBusinessRequestError("事件游标无效或不属于当前会诊")
    value = int(parts[1])
    if value < 1 or value > 9223372036854775807:
        raise InvalidBusinessRequestError("事件游标序列无效")
    return value


def event_record(
    identifier: UUID, turn: int, sequence: int, event: StreamEventName, payload: dict[str, object]
) -> SSEEvent:
    """封装公开状态，患者事实和模型正文不得放入事件载荷。"""
    return SSEEvent(
        event_id=f"{identifier}:{sequence}",
        consultation_id=identifier,
        sequence=sequence,
        turn=turn,
        event=event,
        payload=payload,
        created_at=datetime.now(UTC),
    )


def snapshot_records(
    snapshot: ConsultationSnapshot, records: list[RuntimeWrite]
) -> list[RuntimeWrite]:
    """业务事务内保存请求结果；响应丢失后重试不重复人工审核。"""
    result = list(records)
    receipt = receipt_context.get()
    if receipt:
        key, digest = receipt
        result.append(
            ("receipt", key, {"digest": digest, "result": snapshot.model_dump(mode="json")})
        )
    return result


class MemoryRuntimeStore:
    """共享开发 Repository 的运行数据，不冒充 PostgreSQL 持久队列。"""

    def __init__(self) -> None:
        """创建无阻塞内存数据与资源锁。"""
        self.records: dict[tuple[str, str], dict[str, object]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def mutex(self, key: str) -> AsyncIterator[None]:
        """按资源串行共享此存储的多个 Service。"""
        async with self.locks.setdefault(key, asyncio.Lock()):
            yield

    async def read(self, kind: str, key: str) -> dict[str, object] | None:
        """返回深拷贝，不允许偷偷更新已提交租约。"""
        return deepcopy(self.records.get((kind, key)))

    async def write(self, kind: str, key: str, payload: dict[str, object]) -> None:
        """保存独立运行记录。"""
        self.records[(kind, key)] = deepcopy(payload)

    def commit(self, snapshot: ConsultationSnapshot, records: list[RuntimeWrite]) -> None:
        """无 await 的开发原子提交，业务校验后保存意图和状态事件。"""
        for kind, key, payload in snapshot_records(snapshot, records):
            self.records[(kind, key)] = deepcopy(payload)
        self._emit(
            snapshot.consultation_id, snapshot.turn, "snapshot_updated", {"status": snapshot.status}
        )
        if snapshot.status in {"completed", "cancelled", "failed"}:
            self._emit(
                snapshot.consultation_id,
                snapshot.turn,
                "failed" if snapshot.status == "failed" else "completed",
                {"status": snapshot.status},
            )

    def _emit(
        self, identifier: UUID, turn: int, event: StreamEventName, payload: dict[str, object]
    ) -> None:
        """同步递增 head 与事件，协程切换不会拆开此提交。"""
        sequence = int(str(self.records.get(("head", str(identifier)), {}).get("sequence", 0))) + 1
        item = event_record(identifier, turn, sequence, event, payload)
        self.records[("head", str(identifier))] = {"sequence": sequence}
        self.records[("event", f"{identifier}:{sequence:020d}")] = item.model_dump(mode="json")

    async def emit(
        self, identifier: UUID, turn: int, event: StreamEventName, payload: dict[str, object]
    ) -> None:
        """追加实际节点、检索或模型事件。"""
        self._emit(identifier, turn, event, payload)

    async def scan(self, kind: str, identifier: UUID | None = None) -> list[dict[str, object]]:
        """每轮限量读取过期/待执行任务与未完成补偿。"""
        values = []
        for (record_kind, _), value in self.records.items():
            if record_kind != kind:
                continue
            if identifier is not None and value.get("consultation_id") != str(identifier):
                continue
            if kind == "job" and (
                value.get("status") not in {"pending", "running"}
                or (
                    value.get("lease_until")
                    and datetime.fromisoformat(str(value["lease_until"])) > datetime.now(UTC)
                )
            ):
                continue
            if kind == "action" and value.get("completed"):
                continue
            values.append(deepcopy(value))
        return values[:256]

    async def events(self, identifier: UUID, after: int, limit: int = 100) -> list[SSEEvent]:
        """读取当前开发存储的有序事件页。"""
        items = sorted(
            (
                SSEEvent.model_validate(value)
                for (kind, key), value in self.records.items()
                if kind == "event" and key.startswith(f"{identifier}:")
            ),
            key=lambda item: item.sequence,
        )
        return [item for item in items if item.sequence > after][:limit]


class PostgresRuntimeStore:
    """同业务库的运行适配器；独立锁连接与业务事务分工明确。"""

    def __init__(self, engine: AsyncEngine, coordination_engine: AsyncEngine | None = None) -> None:
        """绑定应用级连接池，不自动建表。"""
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.lock_sessions = async_sessionmaker(
            coordination_engine or engine, expire_on_commit=False
        )

    @asynccontextmanager
    async def mutex(self, key: str) -> AsyncIterator[None]:
        """事务 advisory lock 跨进程串行，断开连接自动释放。"""
        number = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], signed=True)
        try:
            # 等待者不长期占用连接；否则大量重试请求可耗尽池并阻塞持锁者提交。
            while True:
                async with self.lock_sessions.begin() as session:
                    acquired = await session.scalar(
                        text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": number}
                    )
                    if acquired:
                        yield
                        return
                await asyncio.sleep(0.05)
        except SQLAlchemyError as exc:
            raise PersistenceError("运行协调存储不可用") from exc

    async def read(self, kind: str, key: str) -> dict[str, object] | None:
        """精确读取运行数据，异常不暴露连接或患者信息。"""
        try:
            async with self.sessions() as session:
                row = await session.get(RuntimeRecordModel, (kind, key))
                return deepcopy(row.payload) if row else None
        except SQLAlchemyError as exc:
            raise PersistenceError("读取运行记录失败") from exc

    @staticmethod
    async def put(session: AsyncSession, kind: str, key: str, payload: dict[str, object]) -> None:
        """在传入事务内 upsert，供业务 Repository 原子提交。"""
        statement = insert(RuntimeRecordModel).values(kind=kind, key=key, payload=payload)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=["kind", "key"], set_={"payload": payload}
            )
        )

    async def write(self, kind: str, key: str, payload: dict[str, object]) -> None:
        """提交独立运行记录，调用方负责对应资源互斥。"""
        try:
            async with self.sessions.begin() as session:
                await self.put(session, kind, key, payload)
        except SQLAlchemyError as exc:
            raise PersistenceError("保存运行记录失败") from exc

    @staticmethod
    async def append_event(
        session: AsyncSession,
        identifier: UUID,
        turn: int,
        event: StreamEventName,
        payload: dict[str, object],
    ) -> None:
        """head 与事件同事务递增；回滚不留下丢失的序列。"""
        statement = insert(RuntimeRecordModel).values(
            kind="head", key=str(identifier), payload={"sequence": 1}
        )
        returning = statement.on_conflict_do_update(
            index_elements=["kind", "key"],
            set_={
                "payload": func.jsonb_build_object(
                    "sequence", cast(RuntimeRecordModel.payload["sequence"].astext, Integer) + 1
                )
            },
        ).returning(RuntimeRecordModel.payload)
        head = await session.scalar(returning)
        assert isinstance(head, dict)
        sequence = int(str(head["sequence"]))
        item = event_record(identifier, turn, sequence, event, payload)
        await PostgresRuntimeStore.put(
            session, "event", f"{identifier}:{sequence:020d}", item.model_dump(mode="json")
        )

    async def commit(
        self, session: AsyncSession, snapshot: ConsultationSnapshot, records: list[RuntimeWrite]
    ) -> None:
        """业务快照、任务/收据/补偿意图及状态事件一起提交。"""
        for kind, key, payload in snapshot_records(snapshot, records):
            await self.put(session, kind, key, payload)
        await self.append_event(
            session,
            snapshot.consultation_id,
            snapshot.turn,
            "snapshot_updated",
            {"status": snapshot.status},
        )
        if snapshot.status in {"completed", "cancelled", "failed"}:
            await self.append_event(
                session,
                snapshot.consultation_id,
                snapshot.turn,
                "failed" if snapshot.status == "failed" else "completed",
                {"status": snapshot.status},
            )

    async def emit(
        self, identifier: UUID, turn: int, event: StreamEventName, payload: dict[str, object]
    ) -> None:
        """持久追加公开进度，保存失败明确停止本次运行。"""
        try:
            async with self.sessions.begin() as session:
                await self.append_event(session, identifier, turn, event, payload)
        except SQLAlchemyError as exc:
            raise PersistenceError("保存运行事件失败") from exc

    async def events(self, identifier: UUID, after: int, limit: int = 100) -> list[SSEEvent]:
        """按精确会诊前缀和序列分页，不使用全局事件列表。"""
        try:
            async with self.sessions() as session:
                rows = await session.scalars(
                    select(RuntimeRecordModel.payload)
                    .where(
                        RuntimeRecordModel.kind == "event",
                        RuntimeRecordModel.key.startswith(f"{identifier}:"),
                        RuntimeRecordModel.key > f"{identifier}:{after:020d}",
                    )
                    .order_by(RuntimeRecordModel.key)
                    .limit(limit)
                )
                return [SSEEvent.model_validate(item) for item in rows]
        except SQLAlchemyError as exc:
            raise PersistenceError("读取事件页失败") from exc

    async def scan(self, kind: str, identifier: UUID | None = None) -> list[dict[str, object]]:
        """只查询可接管任务与待补偿动作，终态记录不阻塞队列。"""
        conditions = [RuntimeRecordModel.kind == kind]
        if identifier is not None:
            conditions.append(
                RuntimeRecordModel.payload["consultation_id"].astext == str(identifier)
            )
        if kind == "job":
            conditions.extend(
                [
                    RuntimeRecordModel.payload["status"].astext.in_(["pending", "running"]),
                    or_(
                        RuntimeRecordModel.payload["lease_until"].astext.is_(None),
                        cast(
                            RuntimeRecordModel.payload["lease_until"].astext,
                            DateTime(timezone=True),
                        )
                        <= datetime.now(UTC),
                    ),
                ]
            )
        elif kind == "action":
            conditions.append(RuntimeRecordModel.payload["completed"].astext == "false")
        try:
            async with self.sessions() as session:
                rows = await session.scalars(
                    select(RuntimeRecordModel.payload)
                    .where(*conditions)
                    .order_by(RuntimeRecordModel.key)
                    .limit(256)
                )
                return [dict(item) for item in rows]
        except SQLAlchemyError as exc:
            raise PersistenceError("读取恢复任务失败") from exc


def next_lease(seconds: float) -> datetime:
    """生成 UTC 租约截止时间。"""
    return datetime.now(UTC) + timedelta(seconds=seconds)
