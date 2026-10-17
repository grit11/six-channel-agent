"""QA PostgreSQL 隔离验收：仅在随机 qa_v21_test_* schema 内验证迁移、跨进程及归属。

读取既有 env 的数据库配置但不打印连接串，不修改业务/临床 schema；结束清理测试 schema。
依赖 SQLAlchemy、Alembic 和 QARepository，不调用在线模型或真实患者数据。
"""

import argparse
import asyncio
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agents.knowledge_qa_tools import QAError
from backend.app.core.config import Settings
from backend.app.persistence.qa_repository import QARepository
from backend.app.schemas.knowledge import KnowledgeChatResponse, QASessionPatch


async def migration(connection: AsyncConnection, direction: str) -> None:
    """通过真实 Alembic Operations 在隔离连接中执行仅 QA 的新迁移。"""
    path = ROOT / "backend/alembic/versions/0003_qa_history.py"
    spec = importlib.util.spec_from_file_location("qa_migration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("QA 迁移模块不存在")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def apply(sync_connection: Connection) -> None:
        """迁移由固定源码调用，不接受任意 SQL 或用户表名。"""
        context = MigrationContext.configure(sync_connection)
        with Operations.context(context):
            getattr(module, direction)()

    await connection.run_sync(apply)


async def probe(args: argparse.Namespace) -> dict[str, object]:
    """新进程恢复或主验收共用同一测试 schema，URL 仅留在内存。"""
    # BaseSettings 支持运行时 env_file 参数，Pydantic 的静态构造签名不包含此项。
    settings = Settings(_env_file=args.env_file)  # type: ignore[call-arg]
    url = settings.database_url.get_secret_value()
    if not url.startswith("postgresql+asyncpg://"):
        raise RuntimeError("未配置 asyncpg PostgreSQL")
    schema = args.schema or "qa_v21_test_" + uuid4().hex
    if not re.fullmatch(r"qa_v21_test_[0-9a-f]{32}", schema):
        raise RuntimeError("隔离 schema 名称不符合约束")
    admin = create_async_engine(url, poolclass=NullPool)
    engine = create_async_engine(
        url,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema + ",pg_catalog"}},
    )
    repository = QARepository(engine)
    if args.schema:
        try:
            session_id = UUID(args.session)
            session = await repository.get(session_id, "synthetic:a")
            history = await repository.history(session_id, "synthetic:a")
            assert session.title == "跨进程测试" and history[-1].response is not None
            assert history[-1].content == "合成工程回答，非医学结论"
            try:
                await repository.get(session_id, "synthetic:b")
            except QAError as exc:
                assert exc.code == "QA_SESSION_NOT_FOUND"
            else:
                raise AssertionError("归属隔离失败")
            return {"restart": True, "owner_isolation": True}
        finally:
            await engine.dispose()
            await admin.dispose()
    created = False
    try:
        async with admin.begin() as connection:
            await connection.execute(text('CREATE SCHEMA "' + schema + '"'))
            created = True
        async with engine.begin() as connection:
            await migration(connection, "upgrade")
        session = await repository.create("synthetic:a", "跨进程测试", "1.2")
        client_id = uuid4()
        turn, fresh = await repository.begin(
            session.session_id,
            "synthetic:a",
            client_id,
            "合成问题",
            seconds=60,
            model="synthetic",
            prompt_digest="probe",
            version="1.2",
        )
        assert fresh
        duplicate, fresh = await repository.begin(
            session.session_id,
            "synthetic:a",
            client_id,
            "合成问题",
            seconds=60,
            model="synthetic",
            prompt_digest="probe",
            version="1.2",
        )
        assert not fresh and duplicate.message_id == turn.message_id
        await repository.finish(
            session.session_id,
            "synthetic:a",
            turn.message_id,
            KnowledgeChatResponse(
                session_id=session.session_id,
                turn_id=turn.message_id,
                status="no_evidence",
                answer="合成工程回答，非医学结论",
                knowledge_version="1.2",
                duration_ms=1,
            ),
        )
        process = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--env-file",
                args.env_file,
                "--schema",
                schema,
                "--session",
                str(session.session_id),
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert process.returncode == 0, "子进程恢复失败"
        child = json.loads(process.stdout.strip().splitlines()[-1])
        current = await repository.get(session.session_id, "synthetic:a")
        await repository.patch(
            session.session_id,
            "synthetic:a",
            QASessionPatch(expected_revision=current.revision, title="已重命名"),
        )
        assert len(await repository.list_sessions("synthetic:a", q="已重命名")) == 1
        assert await repository.list_sessions("synthetic:b") == []
        current = await repository.get(session.session_id, "synthetic:a")
        await repository.remove(session.session_id, "synthetic:a", current.revision)
        async with engine.begin() as connection:
            await migration(connection, "downgrade")
            remaining = await connection.scalar(
                text("SELECT count(*) FROM information_schema.tables WHERE table_schema=:schema"),
                {"schema": schema},
            )
            assert remaining == 0
        return {"migration_up_down": True, "idempotency": True, "crud": True, **child}
    finally:
        await engine.dispose()
        if created:
            async with admin.begin() as connection:
                await connection.execute(text('DROP SCHEMA "' + schema + '" CASCADE'))
        await admin.dispose()


def main() -> None:
    """输出安全验收结果；异常只公布类型，不泄露数据库账号或密码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--session")
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(probe(args)), ensure_ascii=False))
    except (
        SQLAlchemyError,
        QAError,
        OSError,
        ValueError,
        RuntimeError,
        AssertionError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
