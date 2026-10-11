"""会话 Repository 抽象与开发态内存实现。

职责：提供 consultation 快照的并发版本边界，隔离 Service 与存储实现。
边界：内存实现不提供重启持久化；PostgreSQL 已实现事务适配，须迁移并验证后启用。
"""

from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.core.exceptions import (
    ConsultationNotFoundError,
    InvalidStateTransitionError,
    PersistenceError,
)
from backend.app.persistence.models import (
    AuditEventModel,
    ConsultationModel,
    ConsultationTurnModel,
    DoctorReviewModel,
    EvidenceTraceModel,
    ReasoningPathModel,
)
from backend.app.persistence.runtime import (
    MemoryRuntimeStore,
    PostgresRuntimeStore,
    RuntimeStore,
    RuntimeWrite,
)
from backend.app.schemas.consultation import (
    ConsultationHistoryItem,
    ConsultationHistoryPage,
    ConsultationHistoryQuery,
    ConsultationSnapshot,
)


class ConsultationRepository(Protocol):
    """业务会话存储端口，Service 只依赖这些异步方法。"""

    @property
    def runtime(self) -> RuntimeStore:
        """读取共享运行存储，Service 不得替换 Repository 的运行后端。"""

    async def create(
        self, snapshot: ConsultationSnapshot, *, runtime_records: list[RuntimeWrite] | None = None
    ) -> ConsultationSnapshot:
        """创建新快照。"""

    async def get(self, consultation_id: UUID) -> ConsultationSnapshot:
        """按业务 ID 获取快照。"""

    async def list_history(self, query: ConsultationHistoryQuery) -> ConsultationHistoryPage:
        """按明确过滤和稳定排序返回业务摘要分页。"""

    async def save(
        self,
        snapshot: ConsultationSnapshot,
        *,
        expected_turn: int | None,
        runtime_records: list[RuntimeWrite] | None = None,
    ) -> ConsultationSnapshot:
        """按轮次条件更新快照，版本冲突必须显式失败。"""


class InMemoryConsultationRepository:
    """开发和单测使用的进程内 Repository，不冒充业务数据库。"""

    def __init__(self) -> None:
        """创建隔离存储；应用重启后数据自然失效。"""
        self._items: MutableMapping[UUID, ConsultationSnapshot] = {}
        self.runtime = MemoryRuntimeStore()

    async def create(
        self, snapshot: ConsultationSnapshot, *, runtime_records: list[RuntimeWrite] | None = None
    ) -> ConsultationSnapshot:
        """保存初始快照并拒绝重复业务 ID。"""
        if snapshot.consultation_id in self._items:
            raise InvalidStateTransitionError("辨证会话已存在")
        self._items[snapshot.consultation_id] = snapshot.model_copy(deep=True)
        self.runtime.commit(snapshot, runtime_records or [])
        return snapshot.model_copy(deep=True)

    async def get(self, consultation_id: UUID) -> ConsultationSnapshot:
        """读取当前快照，缺失转为领域错误。"""
        try:
            return self._items[consultation_id].model_copy(deep=True)
        except KeyError as exc:
            raise ConsultationNotFoundError("未找到指定辨证会话") from exc

    async def list_history(self, query: ConsultationHistoryQuery) -> ConsultationHistoryPage:
        """分页当前进程内记录；不宣称重启后可找回 memory 历史。"""
        items = [
            item
            for item in self._items.values()
            if (query.status is None or item.status == query.status)
            and (query.patient_ref is None or item.patient_ref == query.patient_ref)
        ]
        items.sort(key=lambda item: (item.updated_at, str(item.consultation_id)), reverse=True)
        offset = (query.page - 1) * query.page_size
        return ConsultationHistoryPage(
            items=[
                ConsultationHistoryItem.from_snapshot(item)
                for item in items[offset : offset + query.page_size]
            ],
            total=len(items),
            page=query.page,
            page_size=query.page_size,
        )

    async def save(
        self,
        snapshot: ConsultationSnapshot,
        *,
        expected_turn: int | None,
        runtime_records: list[RuntimeWrite] | None = None,
    ) -> ConsultationSnapshot:
        """以乐观轮次检查保存，避免旧请求覆盖医生新编辑。"""
        current = await self.get(snapshot.consultation_id)
        if expected_turn is not None and current.turn != expected_turn:
            raise InvalidStateTransitionError("会话版本已更新，请刷新后重试")
        if snapshot.turn <= current.turn:
            raise InvalidStateTransitionError("新快照版本必须递增")
        now = datetime.now(UTC)
        saved = snapshot.model_copy(update={"updated_at": now}, deep=True)
        self._items[snapshot.consultation_id] = saved
        self.runtime.commit(saved, runtime_records or [])
        return saved.model_copy(deep=True)


class DisabledConsultationRepository:
    """业务能力关闭时显式失败，不偷偷用 checkpoint 保存历史。"""

    def __init__(self) -> None:
        """业务操作仍明确失败；运行替身不会保存临床业务。"""
        self.runtime = MemoryRuntimeStore()

    async def create(
        self, snapshot: ConsultationSnapshot, *, runtime_records: list[RuntimeWrite] | None = None
    ) -> ConsultationSnapshot:
        """创建需要启用业务存储。"""
        raise PersistenceError("会话业务存储未启用；开发验证可显式配置 CONSULTATION_STORAGE=memory")

    async def get(self, consultation_id: UUID) -> ConsultationSnapshot:
        """查询需要启用业务存储。"""
        raise PersistenceError("会话业务存储未启用")

    async def list_history(self, query: ConsultationHistoryQuery) -> ConsultationHistoryPage:
        """未启用业务历史时明确失败，不能伪装空列表或读取 checkpoint。"""
        raise PersistenceError("会话业务存储未启用")

    async def save(
        self,
        snapshot: ConsultationSnapshot,
        *,
        expected_turn: int | None,
        runtime_records: list[RuntimeWrite] | None = None,
    ) -> ConsultationSnapshot:
        """更新需要启用业务存储。"""
        raise PersistenceError("会话业务存储未启用")


class PostgresConsultationRepository:
    """事务性业务 Repository：行锁/CAS 与追加历史，不感知图 checkpoint。"""

    def __init__(self, engine: AsyncEngine, coordination_engine: AsyncEngine | None = None) -> None:
        """绑定应用级 PostgreSQL Engine，不隐式创建或修改业务表。"""
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self.runtime = PostgresRuntimeStore(engine, coordination_engine)

    @staticmethod
    async def _append_records(session: AsyncSession, snapshot: ConsultationSnapshot) -> None:
        """在同一事务追加轮次、路径、精确引用及新审核动作。"""
        payload = snapshot.model_dump(mode="json")
        session.add(
            ConsultationTurnModel(
                id=uuid4(),
                consultation_id=snapshot.consultation_id,
                turn=snapshot.turn,
                status=snapshot.status,
                snapshot=payload,
                created_at=snapshot.updated_at,
            )
        )
        session.add(
            AuditEventModel(
                id=uuid4(),
                consultation_id=snapshot.consultation_id,
                turn=snapshot.turn,
                event="consultation." + snapshot.status,
                payload={"thread_id": snapshot.thread_id, "status": snapshot.status},
                created_at=snapshot.updated_at,
            )
        )
        if snapshot.reasoning is not None and snapshot.status != "running":
            for path in snapshot.reasoning.reasoning_paths:
                session.add(
                    ReasoningPathModel(
                        id=uuid4(),
                        consultation_id=snapshot.consultation_id,
                        turn=snapshot.turn,
                        path_id=path.path_id,
                        rank=path.rank,
                        payload=path.model_dump(mode="json"),
                    )
                )
            catalog = snapshot.reasoning.evidence_catalog
            # 实机复现：大 JSONB 证据的 asyncpg executemany 会停在协议写入。
            # 显式多行 VALUES 避开该路径；按 100 条分批限制参数数目，仍绑定参数、
            # 保留完整原文且共用业务事务，任一批失败会整体回滚，绝不截断证据。
            for start in range(0, len(catalog), 100):
                records = [
                    {
                        "id": uuid4(),
                        "consultation_id": snapshot.consultation_id,
                        "turn": snapshot.turn,
                        "evidence_id": evidence.evidence_id,
                        "payload": evidence.model_dump(mode="json"),
                    }
                    for evidence in catalog[start : start + 100]
                ]
                await session.execute(insert(EvidenceTraceModel).values(records))

    async def create(
        self, snapshot: ConsultationSnapshot, *, runtime_records: list[RuntimeWrite] | None = None
    ) -> ConsultationSnapshot:
        """原子创建 consultation 与第一轮记录，重复 ID 作为明确存储错误。"""
        try:
            async with self._sessions.begin() as session:
                session.add(
                    ConsultationModel(
                        id=snapshot.consultation_id,
                        thread_id=snapshot.thread_id,
                        doctor_id=snapshot.doctor_id,
                        patient_ref=snapshot.patient_ref,
                        status=snapshot.status,
                        turn=snapshot.turn,
                        snapshot=snapshot.model_dump(mode="json"),
                        protocol_version=snapshot.protocol_version,
                        kb_version=snapshot.knowledge_base_version,
                        created_at=snapshot.created_at,
                        updated_at=snapshot.updated_at,
                        completed_at=snapshot.completed_at,
                    )
                )
                await session.flush()
                await self._append_records(session, snapshot)
                await self.runtime.commit(session, snapshot, runtime_records or [])
        except SQLAlchemyError as exc:
            raise PersistenceError("创建业务会话失败，请检查 PostgreSQL 和迁移状态") from exc
        return snapshot.model_copy(deep=True)

    async def get(self, consultation_id: UUID) -> ConsultationSnapshot:
        """读取并按正式 Schema 验证持久化快照。"""
        try:
            async with self._sessions() as session:
                row = await session.get(ConsultationModel, consultation_id)
                if row is None:
                    raise ConsultationNotFoundError("未找到指定辨证会话")
                return ConsultationSnapshot.model_validate(row.snapshot)
        except SQLAlchemyError as exc:
            raise PersistenceError("读取业务会话失败") from exc

    async def list_history(self, query: ConsultationHistoryQuery) -> ConsultationHistoryPage:
        """在一致事务快照中查询过滤、总数和分页；值全部绑定，不拼接 SQL。"""
        conditions = []
        if query.status is not None:
            conditions.append(ConsultationModel.status == query.status)
        if query.patient_ref is not None:
            conditions.append(ConsultationModel.patient_ref == query.patient_ref)
        try:
            async with self._sessions() as session:
                # 单个语句的窗口总数保证当前页与计数一致；越界页补一次纯计数。
                rows = (
                    await session.execute(
                        select(ConsultationModel.snapshot, func.count().over().label("total"))
                        .where(*conditions)
                        .order_by(ConsultationModel.updated_at.desc(), ConsultationModel.id.desc())
                        .offset((query.page - 1) * query.page_size)
                        .limit(query.page_size)
                    )
                ).all()
                total = (
                    int(rows[0].total)
                    if rows
                    else int(
                        await session.scalar(
                            select(func.count()).select_from(ConsultationModel).where(*conditions)
                        )
                        or 0
                    )
                )
                return ConsultationHistoryPage(
                    items=[
                        ConsultationHistoryItem.from_snapshot(
                            ConsultationSnapshot.model_validate(row.snapshot)
                        )
                        for row in rows
                    ],
                    total=total,
                    page=query.page,
                    page_size=query.page_size,
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("读取业务历史失败") from exc

    async def save(
        self,
        snapshot: ConsultationSnapshot,
        *,
        expected_turn: int | None,
        runtime_records: list[RuntimeWrite] | None = None,
    ) -> ConsultationSnapshot:
        """行锁与轮次检查后原子更新当前快照、追加历史/审核，冲突不触发重试写入。"""
        try:
            async with self._sessions.begin() as session:
                row = await session.scalar(
                    select(ConsultationModel)
                    .where(ConsultationModel.id == snapshot.consultation_id)
                    .with_for_update()
                )
                if row is None:
                    raise ConsultationNotFoundError("未找到指定辨证会话")
                if expected_turn is not None and row.turn != expected_turn:
                    raise InvalidStateTransitionError("会话版本已更新，请刷新后重试")
                if snapshot.turn <= row.turn:
                    raise InvalidStateTransitionError("新快照版本必须递增")
                before = ConsultationSnapshot.model_validate(row.snapshot)
                saved = snapshot.model_copy(update={"updated_at": datetime.now(UTC)}, deep=True)
                row.status, row.turn = saved.status, saved.turn
                row.snapshot = saved.model_dump(mode="json")
                row.updated_at, row.completed_at = saved.updated_at, saved.completed_at
                await self._append_records(session, saved)
                await self.runtime.commit(session, saved, runtime_records or [])
                existing = {item.review_id for item in before.review_history}
                for review in saved.review_history:
                    if review.review_id not in existing:
                        session.add(
                            DoctorReviewModel(
                                id=review.review_id,
                                consultation_id=saved.consultation_id,
                                turn=saved.turn,
                                action=review.action,
                                payload=review.model_dump(mode="json"),
                                created_at=review.created_at,
                            )
                        )
        except SQLAlchemyError as exc:
            raise PersistenceError("保存业务会话失败，事务已回滚") from exc
        return saved
