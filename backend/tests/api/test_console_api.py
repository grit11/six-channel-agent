"""原生 Console 路由回归测试。

职责：确认单页入口和运行状态接口已挂载到真实 FastAPI 应用。
边界：不伪造会诊结果，不调用在线模型；医学流程仍由主工程服务负责。
"""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_dashboard_is_served_by_live_app(tmp_path: Path) -> None:
    """确认 Console 页面由真实应用返回，而不是独立演示服务。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.sqlite3"),
        consultation_storage="disabled",
    )
    with TestClient(create_app(settings)) as client:
        page = client.get("/dashboard")
        assert page.status_code == 200
        assert "临床辨证工作台" in page.text
        assert "当前为固定示例 UI 演示" not in page.text
        assert client.get("/").text == page.text
        assert client.get("/console-static/console.js").status_code == 200
        assert client.get("/console-static/labels.js").status_code == 200
        icon = client.get("/favicon.ico")
        assert icon.status_code == 200
        assert icon.headers["content-type"].startswith("image/svg+xml")
        assert client.get("/console-static/favicon.svg").status_code == 200


def test_console_config_reports_real_backends(tmp_path: Path) -> None:
    """配置接口必须报告实际降级后端，不能把 SQLite/Lite 宣称为生产连接。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.sqlite3"),
        consultation_storage="disabled",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/console/config")
        assert response.status_code == 200
        payload = response.json()
        assert payload["agent_connected"] is False
        assert payload["consultation_enabled"] is False
        assert payload["vector_backend"] == "lite"
        assert payload["checkpoint_backend"] == "sqlite"
        assert payload["business_database"] == "disabled"
        assert payload["degraded"] is True


def test_console_all_native_controls_are_present() -> None:
    """检查真实交互脚本的静态 ID 均存在，避免漏接按钮使初始化崩溃。"""
    static = Path(__file__).resolve().parents[2] / "app/static"
    script = (static / "console.js").read_text(encoding="utf-8")
    page = (static / "dashboard.html").read_text(encoding="utf-8")
    controls = set(re.findall(r"\$\('([a-zA-Z][a-zA-Z0-9]*)'\)", script))
    ids = re.findall(r'id="([^\"]+)"', page)
    assert controls <= set(ids), controls - set(ids)
    assert len(ids) == len(set(ids))
