"""QA 独立历史仓储。

职责：会话归属、分页、乐观版本、单轮幂等和原子终态提交。
依赖现有异步数据库引擎；开发内存替身明确不跨重启，不读取临床表或 checkpoint。
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.agents.knowledge_qa_tools import QAError
from backend.app.persistence.models import QAMessageModel, QASessionModel
from backend.app.schemas.knowledge import (
    KnowledgeChatResponse,
    QAMessage,
    QASession,
    QASessionPatch,
)


class QARepository:
    """一个 QA 权威仓储；SQL 用行锁，开发替身用同一会话锁，不持锁等待模型。"""

    def __init__(self, engine: AsyncEngine | None = None) -> None:
        """注入已有引擎，不自动建表或迁移，不修改业务数据库配置。"""
        self.engine = engine
        self.factory = async_sessionmaker(engine, expire_on_commit=False) if engine else None
        self.sessions: dict[UUID, QASession] = {}
        self.messages: dict[UUID, QAMessage] = {}
        self.locks: dict[UUID, asyncio.Lock] = {}
        self.lock_users: dict[UUID, int] = {}
        self.storage = "postgres" if engine else "memory"

    @asynccontextmanager
    async def _transaction(self, identifier: UUID) -> AsyncIterator[AsyncSession | None]:
        """进程内锁保护内存替身；SQL 额外获取行锁保护跨 worker 序列。"""
        if self.factory:
            # PostgreSQL 的短事务和行锁负责跨进程互斥，无需永久缓存进程内会话锁。
            try:
                async with self.factory.begin() as db:
                    yield db
            except SQLAlchemyError as exc:
                raise QAError(
                    "QA_HISTORY_UNAVAILABLE",
                    "QA 历史数据库不可用，请确认 QA 迁移已执行及数据库连接正常",
                ) from exc
            return
        lock = self.locks.setdefault(identifier, asyncio.Lock())
        self.lock_users[identifier] = self.lock_users.get(identifier, 0) + 1
        try:
            async with lock:
                yield None
        finally:
            self.lock_users[identifier] -= 1
            if self.lock_users[identifier] == 0:
                del self.lock_users[identifier]
                del self.locks[identifier]

    async def _get(
        self, identifier: UUID, owner: str, db: AsyncSession | None, *, lock: bool = False
    ) -> QASession:
        if db:
            query = select(QASessionModel).where(
                QASessionModel.session_id == identifier, QASessionModel.owner_id == owner
            )
            row = await db.scalar(query.with_for_update() if lock else query)
            item = QASession.model_validate(row.payload) if row else None
        else:
            item = self.sessions.get(identifier)
        if item is None or item.owner_id != owner or item.status == "deleted":
            raise QAError("QA_SESSION_NOT_FOUND", "QA 会话不存在或无权访问", 404)
        return item.model_copy(deep=True)

    async def _save_session(self, item: QASession, db: AsyncSession | None) -> None:
        if db:
            row = await db.get(QASessionModel, item.session_id)
            values = dict(
                owner_id=item.owner_id,
                title=item.title,
                status=item.status,
                updated_at=item.updated_at,
                payload=item.model_dump(mode="json"),
            )
            if row:
                for name, value in values.items():
                    setattr(row, name, value)
            else:
                db.add(QASessionModel(session_id=item.session_id, **values))
            await db.flush()
        else:
            self.sessions[item.session_id] = item.model_copy(deep=True)

    async def _save_message(self, item: QAMessage, db: AsyncSession | None) -> None:
        if db:
            row = await db.get(QAMessageModel, item.message_id)
            if row:
                row.payload = item.model_dump(mode="json")
            else:
                db.add(
                    QAMessageModel(
                        message_id=item.message_id,
                        session_id=item.session_id,
                        client_message_id=item.client_message_id,
                        sequence=item.sequence,
                        role=item.role,
                        payload=item.model_dump(mode="json"),
                    )
                )
            await db.flush()
        else:
            self.messages[item.message_id] = item.model_copy(deep=True)

    async def _message(self, identifier: UUID, db: AsyncSession | None) -> QAMessage | None:
        if db:
            row = await db.get(QAMessageModel, identifier)
            return QAMessage.model_validate(row.payload) if row else None
        item = self.messages.get(identifier)
        return item.model_copy(deep=True) if item else None

    async def create(self, owner: str, title: str, version: str) -> QASession:
        """后端签发 QA UUID，不接受客户端 owner 或临床 thread。"""
        now = datetime.now(UTC)
        item = QASession(
            session_id=uuid4(),
            owner_id=owner,
            title=title,
            knowledge_version=version,
            storage=cast(Literal["memory", "postgres"], self.storage),
            created_at=now,
            updated_at=now,
        )
        async with self._transaction(item.session_id) as db:
            await self._save_session(item, db)
        return item

    async def get(self, identifier: UUID, owner: str) -> QASession:
        """精确归属过滤；UUID 本身不是身份。"""
        async with self._transaction(identifier) as db:
            return await self._get(identifier, owner, db)

    async def list_sessions(
        self, owner: str, page: int = 1, size: int = 20, q: str = ""
    ) -> list[QASession]:
        """数据库分页与绑定参数搜索，不把其它医生会话加载进客户端。"""
        if self.factory:
            async with self._transaction(uuid4()) as db:
                assert db is not None
                query = select(QASessionModel).where(
                    QASessionModel.owner_id == owner, QASessionModel.status != "deleted"
                )
                if q:
                    query = query.where(QASessionModel.title.contains(q, autoescape=True))
                rows = await db.scalars(
                    query.order_by(QASessionModel.updated_at.desc())
                    .offset((page - 1) * size)
                    .limit(size)
                )
                return [QASession.model_validate(row.payload) for row in rows]
        items = [
            s.model_copy(deep=True)
            for s in self.sessions.values()
            if s.owner_id == owner and s.status != "deleted" and q in s.title
        ]
        return sorted(items, key=lambda s: s.updated_at, reverse=True)[
            (page - 1) * size : page * size
        ]

    async def history(
        self, identifier: UUID, owner: str, after: int = 0, size: int = 100
    ) -> list[QAMessage]:
        """历史结果只读且分页，失败消息没有正式回答正文。"""
        async with self._transaction(identifier) as db:
            await self._get(identifier, owner, db)
            if db:
                rows = await db.scalars(
                    select(QAMessageModel)
                    .where(QAMessageModel.session_id == identifier, QAMessageModel.sequence > after)
                    .order_by(QAMessageModel.sequence)
                    .limit(size)
                )
                return [QAMessage.model_validate(row.payload) for row in rows]
            return sorted(
                [
                    m.model_copy(deep=True)
                    for m in self.messages.values()
                    if m.session_id == identifier and m.sequence > after
                ],
                key=lambda m: m.sequence,
            )[:size]

    async def recent(self, identifier: UUID, owner: str) -> list[QAMessage]:
        """只获取近期完整上下文，全部历史仍保留在权威仓储。"""
        async with self._transaction(identifier) as db:
            item = await self._get(identifier, owner, db)
        return await self.history(identifier, owner, max(0, item.next_sequence - 13), 12)

    async def patch(self, identifier: UUID, owner: str, change: QASessionPatch) -> QASession:
        """基于行锁及 revision 重命名/归档，不影响临床状态。"""
        async with self._transaction(identifier) as db:
            item = await self._get(identifier, owner, db, lock=True)
            if item.revision != change.expected_revision:
                raise QAError("QA_VERSION_CONFLICT", "QA 会话版本已更新", 409)
            if item.active_turn_id:
                raise QAError("QA_SESSION_BUSY", "当前会话正在生成", 409)
            if change.title is not None:
                item.title = change.title
            if change.status is not None:
                item.status = change.status
            item.revision += 1
            item.updated_at = datetime.now(UTC)
            await self._save_session(item, db)
            return item

    async def remove(self, identifier: UUID, owner: str, revision: int) -> None:
        """删除会话下消息并保留无内容墓碑；避免留下敏感问答历史。"""
        async with self._transaction(identifier) as db:
            item = await self._get(identifier, owner, db, lock=True)
            if item.revision != revision or item.active_turn_id:
                raise QAError("QA_VERSION_CONFLICT", "会话正在运行或版本已变化", 409)
            item.status, item.title = "deleted", "已删除"
            item.revision += 1
            await self._save_session(item, db)
            if db:
                await db.execute(
                    delete(QAMessageModel).where(QAMessageModel.session_id == identifier)
                )
            else:
                self.messages = {
                    k: v for k, v in self.messages.items() if v.session_id != identifier
                }

    async def begin(
        self,
        identifier: UUID,
        owner: str,
        client_id: UUID,
        question: str,
        *,
        seconds: float,
        model: str,
        prompt_digest: str,
        version: str,
    ) -> tuple[QAMessage, bool]:
        """事务预留 user/assistant 序列，同键复用已有运行，不重复收费。"""
        digest = hashlib.sha256(question.encode()).hexdigest()
        now = datetime.now(UTC)
        async with self._transaction(identifier) as db:
            item = await self._get(identifier, owner, db, lock=True)
            if db:
                row = await db.scalar(
                    select(QAMessageModel).where(
                        QAMessageModel.session_id == identifier,
                        QAMessageModel.client_message_id == client_id,
                        QAMessageModel.role == "assistant",
                    )
                )
                existing = QAMessage.model_validate(row.payload) if row else None
            else:
                existing = next(
                    (
                        m.model_copy(deep=True)
                        for m in self.messages.values()
                        if m.session_id == identifier
                        and m.client_message_id == client_id
                        and m.role == "assistant"
                    ),
                    None,
                )
            if existing:
                if existing.request_digest != digest:
                    raise QAError("QA_IDEMPOTENCY_CONFLICT", "幂等键已用于不同问题", 409)
                if existing.status == "running" and existing.expires_at <= now:
                    existing.status, existing.error_code = "interrupted", "QA_RUN_INTERRUPTED"
                    await self._save_message(existing, db)
                    if item.active_turn_id == existing.message_id:
                        item.active_turn_id = None
                        await self._save_session(item, db)
                return existing, False
            if item.status != "active":
                raise QAError("QA_SESSION_ARCHIVED", "请恢复归档会话后提问", 409)
            if item.active_turn_id:
                running = await self._message(item.active_turn_id, db)
                if running and running.expires_at > now:
                    raise QAError("QA_SESSION_BUSY", "该会话已有运行中的问题", 409)
                if running:
                    running.status, running.error_code = "interrupted", "QA_RUN_INTERRUPTED"
                    await self._save_message(running, db)
            fields = dict(
                session_id=identifier,
                client_message_id=client_id,
                request_digest=digest,
                knowledge_version=version,
                model_id=model,
                prompt_digest=prompt_digest,
                expires_at=now + timedelta(seconds=seconds + 10),
                created_at=now,
                updated_at=now,
            )
            user = QAMessage.model_validate(
                dict(
                    message_id=uuid4(),
                    sequence=item.next_sequence,
                    role="user",
                    content=question,
                    status="completed",
                    **fields,
                )
            )
            assistant = QAMessage.model_validate(
                dict(
                    message_id=uuid4(),
                    sequence=item.next_sequence + 1,
                    role="assistant",
                    content="",
                    status="running",
                    **fields,
                )
            )
            await self._save_message(user, db)
            await self._save_message(assistant, db)
            item.active_turn_id = assistant.message_id
            item.next_sequence += 2
            item.revision += 1
            item.updated_at = now
            await self._save_session(item, db)
            return assistant, True

    async def finish(
        self,
        identifier: UUID,
        owner: str,
        turn: UUID,
        response: KnowledgeChatResponse | None = None,
        *,
        error: str | None = None,
        interrupted: bool = False,
    ) -> None:
        """核验回答及引用一起提交；失败/断线只写安全错误，不写半截答案。"""
        async with self._transaction(identifier) as db:
            item = await self._get(identifier, owner, db, lock=True)
            message = await self._message(turn, db)
            if not message or item.active_turn_id != turn:
                raise QAError("QA_VERSION_CONFLICT", "QA 运行所有权已变化", 409)
            message.updated_at = item.updated_at = datetime.now(UTC)
            message.response = response
            message.content = response.answer if response else ""
            message.status = (
                "partial"
                if response and response.status == "partial"
                else "completed" if response else "interrupted" if interrupted else "failed"
            )
            message.error_code = error
            item.active_turn_id = None
            item.revision += 1
            await self._save_message(message, db)
            await self._save_session(item, db)

    async def turn(self, identifier: UUID, owner: str) -> QAMessage:
        """断线后从权威消息查询状态；跨 owner 永远拒绝。"""
        if self.factory:
            async with self._transaction(uuid4()) as db:
                assert db is not None
                row = await db.scalar(
                    select(QAMessageModel).where(
                        QAMessageModel.message_id == identifier,
                        QAMessageModel.role == "assistant",
                    )
                )
                message = QAMessage.model_validate(row.payload) if row else None
                if not message:
                    raise QAError("QA_TURN_NOT_FOUND", "QA 运行不存在", 404)
        else:
            message = self.messages.get(identifier)
        if not message or message.role != "assistant":
            raise QAError("QA_TURN_NOT_FOUND", "QA 运行不存在", 404)
        async with self._transaction(message.session_id) as db:
            session = await self._get(message.session_id, owner, db, lock=True)
            current = await self._message(identifier, db)
            if current is None:
                raise QAError("QA_TURN_NOT_FOUND", "QA 运行不存在", 404)
            if current.status == "running" and current.expires_at <= datetime.now(UTC):
                current.status, current.error_code = "interrupted", "QA_RUN_INTERRUPTED"
                current.updated_at = datetime.now(UTC)
                await self._save_message(current, db)
                if session.active_turn_id == identifier:
                    session.active_turn_id = None
                    session.revision += 1
                    await self._save_session(session, db)
            return current
