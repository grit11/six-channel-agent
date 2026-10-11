"""真实 Lite 和 SQLite 启动验证，不注入向量测试替身。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.mark.integration
def test_real_local_lifespan(tmp_path: Path) -> None:
    """真实本地后端启动后报告 Lite/SQLite，业务数据库明确关闭。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").json() == {
            "status": "degraded",
            "vector_backend": "lite",
            "checkpoint_backend": "sqlite",
            "business_database": "disabled",
        }
        assert client.get("/api/v1/system/versions").status_code == 200
