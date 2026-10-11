"""业务 Repository 回归：验证版本/CAS、事务追加与 PostgreSQL DDL，不冒充真实数据库。"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateTable

from backend.app.core.config import Settings
from backend.app.core.exceptions import InvalidStateTransitionError, PersistenceError
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.models import (
    AuditEventModel,
    Base,
    ConsultationModel,
    ConsultationTurnModel,
)
from backend.app.persistence.repositories import (
    DisabledConsultationRepository,
    InMemoryConsultationRepository,
    PostgresConsultationRepository,
)
from backend.app.schemas.consultation import ConsultationHistoryQuery, ConsultationSnapshot
from backend.app.services.reasoning_service import ReasoningService


def snapshot() -> ConsultationSnapshot:
    """构造没有医学结论的合法初始业务记录。"""
    now = datetime.now(UTC)
    return ConsultationSnapshot(
        consultation_id=uuid4(),
        thread_id=str(uuid4()),
        status="created",
        turn=1,
        raw_input="测试记录",
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_memory_returns_copies_and_rejects_nonincreasing_version() -> None:
    """读取后的原地编辑不能偷偷改写已保存状态，同版本写入必须拒绝。"""
    repository = InMemoryConsultationRepository()
    original = snapshot()
    await repository.create(original)
    fetched = await repository.get(original.consultation_id)
    fetched.status = "cancelled"
    assert (await repository.get(original.consultation_id)).status == "created"
    with pytest.raises(InvalidStateTransitionError):
        await repository.save(original, expected_turn=1)


def test_postgres_ddl_has_business_and_runtime_tables_without_checkpoint() -> None:
    """六张业务表和独立运行表映射，不将运行记录或 checkpoint 当临床业务替代。"""
    assert set(Base.metadata.tables) == {
        "consultation",
        "consultation_turn",
        "reasoning_path",
        "doctor_review",
        "evidence_trace",
        "audit_event",
        "runtime_record",
    }
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert "CREATE TABLE" in ddl
        assert "checkpoint" not in ddl


@pytest.mark.asyncio
async def test_postgres_create_appends_turn_and_audit_in_transaction() -> None:
    """测试替身验证 create 的事务追加行为及 flush，不替代真实 PostgreSQL 验收。"""
    session = MagicMock()
    session.flush = AsyncMock()
    session.scalar = AsyncMock(return_value={"sequence": 1})
    session.execute = AsyncMock()
    scope = MagicMock()
    scope.__aenter__ = AsyncMock(return_value=session)
    scope.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock()
    factory.begin.return_value = scope
    with patch("backend.app.persistence.repositories.async_sessionmaker", return_value=factory):
        repository = PostgresConsultationRepository(MagicMock())
        original = snapshot()
        result = await repository.create(original)
    session.flush.assert_awaited_once()
    inserted = [call.args[0] for call in session.add.call_args_list]
    assert {type(item) for item in inserted} == {
        ConsultationModel,
        ConsultationTurnModel,
        AuditEventModel,
    }
    assert result == original
    assert result is not original


@pytest.mark.asyncio
async def test_postgres_save_checks_locked_version_before_append() -> None:
    """过期请求在持有行锁时失败，不追加审核或路径，事务由上下文回滚。"""
    original = snapshot()
    row = ConsultationModel(
        id=original.consultation_id, turn=7, snapshot=original.model_dump(mode="json")
    )
    session = MagicMock()
    session.scalar = AsyncMock(return_value=row)
    scope = MagicMock()
    scope.__aenter__ = AsyncMock(return_value=session)
    scope.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock()
    factory.begin.return_value = scope
    with patch("backend.app.persistence.repositories.async_sessionmaker", return_value=factory):
        repository = PostgresConsultationRepository(MagicMock())
        with pytest.raises(InvalidStateTransitionError):
            await repository.save(original.model_copy(update={"turn": 2}), expected_turn=1)
    statement = session.scalar.call_args.args[0]
    assert "FOR UPDATE" in str(statement.compile(dialect=postgresql.dialect()))
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_postgres_history_binds_filters_and_returns_summary() -> None:
    """窗口计数与绑定分页 SQL 的适配器回归；仍不声称 PostgreSQL 实机已接入。"""
    original = snapshot()
    original.patient_ref = "P001"
    session = MagicMock()
    result = MagicMock()
    result.all.return_value = [SimpleNamespace(snapshot=original.model_dump(mode="json"), total=11)]
    session.execute = AsyncMock(return_value=result)
    session.scalar = AsyncMock(return_value=11)
    scope = MagicMock()
    scope.__aenter__ = AsyncMock(return_value=session)
    scope.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=scope)
    with patch("backend.app.persistence.repositories.async_sessionmaker", return_value=factory):
        repository = PostgresConsultationRepository(MagicMock())
        query = ConsultationHistoryQuery(page=2, page_size=10, status="created", patient_ref="P001")
        history = await repository.list_history(query)
        statement = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
        assert "count(*) OVER" in str(statement)
        assert "ORDER BY consultation.updated_at DESC, consultation.id DESC" in str(statement)
        assert "P001" in statement.params.values()
        assert history.total == 11 and history.page == 2
        assert history.items[0].consultation_id == original.consultation_id
        assert "raw_input" not in history.items[0].model_dump()
        result.all.return_value = []
        empty = await repository.list_history(query)
        assert empty.items == [] and empty.total == 11
        session.execute.side_effect = SQLAlchemyError("隔离错误")
        with pytest.raises(PersistenceError):
            await repository.list_history(query)


@pytest.mark.asyncio
async def test_disabled_history_is_explicit_error() -> None:
    """关闭业务存储不能伪造空历史或从 checkpoint 恢复列表。"""
    with pytest.raises(PersistenceError, match="业务存储未启用"):
        await DisabledConsultationRepository().list_history(ConsultationHistoryQuery())


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_second_batch", [False, True])
async def test_large_evidence_catalog_preserves_sources_and_transaction(
    fail_second_batch: bool,
) -> None:
    """跨批真实来源不丢失，后一批失败须退出同一事务，不返回成功快照。"""
    settings = Settings(_env_file=None)
    knowledge = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
    reasoning = await ReasoningService(knowledge).analyze_text("发热，恶寒，无汗，脉浮")
    catalog = []
    for number in range(1, 102):
        evidence = await knowledge.get_clause_by_id(str(number))
        assert evidence is not None
        catalog.append(evidence)
    reasoning.evidence_catalog = catalog
    original = snapshot().model_copy(update={"reasoning": reasoning, "status": "needs_evidence"})
    session = MagicMock()
    session.flush = AsyncMock()
    session.scalar = AsyncMock(return_value={"sequence": 1})
    statements = []

    async def execute(statement: object) -> MagicMock:
        """记录真实参数绑定，模拟后一批数据库失败而非吞掉异常。"""
        statements.append(statement)
        if fail_second_batch and len(statements) == 2:
            raise SQLAlchemyError("第二批测试故障")
        return MagicMock()

    session.execute = AsyncMock(side_effect=execute)
    scope = MagicMock()
    scope.__aenter__ = AsyncMock(return_value=session)
    scope.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock()
    factory.begin.return_value = scope
    with patch("backend.app.persistence.repositories.async_sessionmaker", return_value=factory):
        repository = PostgresConsultationRepository(MagicMock())
        if fail_second_batch:
            with pytest.raises(PersistenceError):
                await repository.create(original)
            assert scope.__aexit__.call_args.args[0] is SQLAlchemyError
        else:
            assert await repository.create(original) == original
            assert scope.__aexit__.call_args.args[0] is None
    # 两批 INSERT 都保留完整来源及轮次，绑定值不拼入 SQL 正文。
    stored = {}
    for statement in statements[:2]:
        compiled = statement.compile(dialect=postgresql.dialect())
        assert str(compiled).startswith("INSERT INTO evidence_trace")
        for key, value in compiled.params.items():
            if key.startswith("payload_"):
                stored[value["evidence_id"]] = value
    assert stored == {item.evidence_id: item.model_dump(mode="json") for item in catalog}
