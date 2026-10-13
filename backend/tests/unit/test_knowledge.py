"""知识服务单测，使用真实来源数据验证精确查询、层级过滤和无证据行为。"""

import json
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.knowledge.classic import load_records, normalize_record
from backend.app.knowledge.service import KnowledgeService


@pytest.fixture
async def service() -> KnowledgeService:
    """加载现有五类主库，不伪造条文和方证数据。"""
    settings = Settings()
    return await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")


@pytest.mark.asyncio
async def test_real_counts_and_exact_clause(service: KnowledgeService) -> None:
    """五类数据完整加载，条文引用必须与来源原文完全一致。"""
    assert service.counts() == {
        "classic_clause": 398,
        "formula_pattern": 128,
        "differential": 53,
        "medical_case": 69,
        "six_channel_framework": 11,
    }
    clause = await service.get_clause_by_id("35")
    assert clause is not None
    assert clause.knowledge_id == "clause_35"
    assert clause.content == clause.metadata["classic_text"]
    assert "麻黄汤" in clause.content
    assert clause.evidence_id == "1.2:clause_35"
    assert clause.rerank_score is None
    assert await service.get_clause_by_id("9999") is None


@pytest.mark.asyncio
async def test_formula_scope_and_risk(service: KnowledgeService) -> None:
    """精确方名保留风险，canonical 和历史派生过滤互不污染。"""
    formula = await service.get_formula_by_name("三物小白散")
    assert formula is not None
    assert formula.source_level == "L3"
    assert formula.metadata["risk_level"] == "high"
    assert formula.metadata["risk_flags"]
    assert await service.get_formula_by_name("不存在的方名") is None
    classic = await service.search_formula("恶寒", association_scope=["classic_clause_based"])
    historical = await service.search_formula("恶寒", association_scope=["case_derived_historical"])
    assert classic and historical
    assert all(e.metadata["association_scope"] == "classic_clause_based" for e in classic)
    assert all(e.metadata["association_scope"] == "case_derived_historical" for e in historical)


@pytest.mark.asyncio
async def test_differential_pair_priority(service: KnowledgeService) -> None:
    """候选对顺序不影响精确优先，同一规则不重复返回。"""
    rules = await service.search_differential(["麻黄汤证", "桂枝汤证"])
    assert rules[0].knowledge_id == "diff_v12_001"
    assert rules[0].retrieval_method == "candidate_pair_exact"
    assert "汗" in str(rules[0].metadata.get("key_dimensions", {}))
    assert len({r.knowledge_id for r in rules}) == len(rules)


@pytest.mark.asyncio
async def test_metadata_and_empty_search(service: KnowledgeService) -> None:
    """六经过滤使用审核后标签，未命中保持空结果。"""
    results = await service.search_classic("脉", semantic_channels=["少阴"])
    assert results
    assert all("少阴" in e.metadata["semantic_six_channels"] for e in results)
    assert await service.search_classic("unmatched_z9x8q7") == []
    exact = await service.search_classic("", clause_nos=["35"])
    assert exact[0].retrieval_method == "exact"


@pytest.mark.asyncio
async def test_bundle_preserves_layers(service: KnowledgeService) -> None:
    """框架和医案证据始终分层，召回分不冒充重排分。"""
    bundle = await service.retrieve_evidence_bundle("恶寒脉浮")
    assert bundle.classic and bundle.formulas and bundle.cases
    assert all(e.source_level == "L1" for e in bundle.classic)
    assert all(e.source_level == "L5" for e in bundle.cases)
    assert all(e.retrieval_method == "bm25" for e in bundle.classic)


def test_grounded_empty_does_not_revert_to_old_field() -> None:
    """已审核空标签不能回填旧六经标签。"""
    row = json.loads(
        next(Settings().knowledge_directory.glob("01_*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    row["semantic_six_channels"] = []
    row["six_channel"] = "太阳"
    record = normalize_record(row, "1.2")
    assert record.channels == []


def test_invalid_knowledge_directory_fails(tmp_path: Path) -> None:
    """缺少文件时不得返回部分知识或伪造来源。"""
    with pytest.raises(KnowledgeRetrievalError):
        load_records(tmp_path, "1.2")


@pytest.mark.asyncio
async def test_returned_metadata_is_isolated(service: KnowledgeService) -> None:
    """外部更改 Evidence 列表不应修改只读知识来源。"""
    evidence = await service.get_clause_by_id("1")
    assert evidence is not None
    evidence.metadata["semantic_six_channels"].clear()
    again = await service.get_clause_by_id("1")
    assert again is not None
    assert again.metadata["semantic_six_channels"] == ["太阳"]


def test_malformed_classic_text_is_rejected() -> None:
    """不得把非字符串原文转换成看似有效证据。"""
    row = json.loads(
        next(Settings().knowledge_directory.glob("01_*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    row["classic_text"] = None
    with pytest.raises(ValueError):
        normalize_record(row, "1.2")
