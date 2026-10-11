"""
配置单元测试。

职责：验证默认配置、边界校验和 Phase 0 的本地 fallback 配置。
"""

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def test_default_settings_use_local_fallbacks() -> None:
    """默认配置应允许本地开发使用自动降级。"""

    settings = Settings()
    assert settings.milvus_mode == "auto"
    assert settings.milvus_fallback_to_lite is True
    assert settings.checkpoint_backend == "auto"
    assert settings.checkpoint_fallback_to_sqlite is True


def test_top_k_is_bounded() -> None:
    """Top-K 超出协议允许范围时必须在配置阶段失败。"""

    with pytest.raises(ValidationError):
        Settings(top_k_default=6)


@pytest.mark.parametrize(
    "values",
    [
        {"top_k_default": 4, "top_k_max": 3},
        {"retrieval_top_k": 3, "rerank_top_k": 4},
        {"embedding_dimension": 0},
        {"database_enabled": True},
        {"checkpoint_backend": "postgres"},
        {"app_env": "prod"},
    ],
)
def test_invalid_runtime_configuration(values: dict[str, object]) -> None:
    """启动时拒绝不一致配置，避免节点执行时才出现模糊错误。"""
    with pytest.raises(ValidationError):
        Settings(**values)


def test_secrets_are_redacted() -> None:
    """配置对象展示不能泄露模型密钥或数据库密码。"""
    settings = Settings(llm_api_key="private-key", database_url="postgresql://private-password")
    assert "private-key" not in repr(settings)
    assert "private-password" not in repr(settings)
