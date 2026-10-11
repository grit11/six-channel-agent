"""临床解析 API 回归：验证三值事实和冲突可审计返回。"""

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_clinical_parse_api_returns_editable_evidence() -> None:
    """接口返回结构化事实，不把解析结果包装成诊断结论。"""
    settings = Settings(_env_file=None, checkpoint_backend="sqlite")
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/clinical/parse", json={"text": "发热，无汗，舌脉未提供"})
    assert response.status_code == 200
    body = response.json()
    assert body["parser_version"] == "rules-v0.3"
    assert body["clinical_evidence"]["findings"][0]["polarity"] == "positive"
    assert any(item["polarity"] == "negative" for item in body["clinical_evidence"]["findings"])
    assert any(
        item["key"] == "tongue" and item["polarity"] == "unknown"
        for item in body["clinical_evidence"]["findings"]
    )
    assert "diagnosis" not in body


def test_clinical_parse_api_rejects_unknown_fields() -> None:
    """解析 API 不允许客户端注入未定义推理字段。"""
    settings = Settings(_env_file=None, checkpoint_backend="sqlite")
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/clinical/parse", json={"text": "恶寒", "channel": "太阳"})
    assert response.status_code == 422
