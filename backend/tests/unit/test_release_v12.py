"""V2 发布整理回归：验证版本、验收范围与合并后的重排探测。

依赖真实知识和假重排器，不访问在线服务或用户数据库，不评价医学准确率。
"""

import json
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from scripts.engineering_acceptance import write_report
from scripts.verify_real_hybrid import probe_reranker, verify

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.main import create_app
from backend.app.schemas.evidence import RetrievedDocument

ROOT = Path(__file__).resolve().parents[3]


class FakeReranker:
    """逆序返回真实来源，验证工具不依赖指定厂商。"""

    async def rerank(
        self, query: str, candidates: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        """增加测试分数并保留输入身份和原文。"""
        assert query and top_k == 3
        return [item.model_copy(update={"rerank_score": 0.5}) for item in reversed(candidates)]


async def test_probe_only_uses_configured_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """独立探测不连接向量或 Embedding，也不枚举其他模型。"""
    from scripts import verify_real_hybrid as module

    @asynccontextmanager
    async def fake_reranker(settings: Settings) -> AsyncIterator[FakeReranker]:
        """提供不访问网络的重排器生命周期。"""
        yield FakeReranker()

    def forbidden(*args: object, **kwargs: object) -> None:
        """连接无关基础设施时直接失败。"""
        raise AssertionError("独立探测不得连接向量或 Embedding")

    monkeypatch.setattr(module, "open_reranker", fake_reranker)
    monkeypatch.setattr(module, "build_milvus_store", forbidden)
    monkeypatch.setattr(module, "open_embedding_provider", forbidden)
    result = await verify(
        Settings(_env_file=None, knowledge_directory=ROOT / "knowledge/v1.2"),
        reranker_only=True,
    )
    assert result["source_preserved"] is True
    assert [item["knowledge_id"] for item in result["results"]] == [
        "clause_35",
        "clause_2",
        "clause_1",
    ]


async def test_probe_rejects_modified_sources() -> None:
    """改写来源不能生成成功摘要。"""

    class ChangedReranker(FakeReranker):
        """模拟错误服务返回修改原文。"""

        async def rerank(
            self, query: str, candidates: list[RetrievedDocument], top_k: int
        ) -> list[RetrievedDocument]:
            """返回篡改正文，验证来源边界。"""
            results = await super().rerank(query, candidates, top_k)
            results[0] = results[0].model_copy(update={"content": "改写内容"})
            return results

    knowledge = await KnowledgeService.from_directory(ROOT / "knowledge/v1.2", "1.2")
    with pytest.raises(ValueError, match="改写"):
        await probe_reranker(knowledge, ChangedReranker())


def test_release_version_and_acceptance_scope(tmp_path: Path) -> None:
    """包与应用版本一致，验收范围包含 S6 医生审核。"""
    metadata = tomllib.loads((ROOT / "backend/pyproject.toml").read_text(encoding="utf-8"))
    assert create_app(Settings(_env_file=None)).version == metadata["project"]["version"] == "2.0.0"
    write_report(tmp_path, [{"name": "fixture", "status": "passed"}], 1)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "医生审核" in report["scope"]
    assert "S6" not in report["deferred"]
