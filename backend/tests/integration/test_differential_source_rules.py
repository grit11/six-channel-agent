"""真实 53 条来源规则逐条节点回归。

通过 KnowledgeService 查询真实来源；仅验证归属、观察与保守执行行为，不冒充临床金标准。
"""

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.schemas.reasoning import FormulaCandidate


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """离线加载主库，节点仍使用唯一知识服务边界。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


@pytest.mark.parametrize("number", range(1, 54))
async def test_each_source_rule_applies_without_inventing_values(
    knowledge: KnowledgeService, number: int
) -> None:
    """每条规则进入节点且保留原维度；无明确患者事实时不得产生正反证。"""
    source = await knowledge.get_differential_by_id(f"diff_v12_{number:03}")
    assert source is not None
    formulas = [
        FormulaCandidate(
            formula_name=label,
            association_scope="classic_clause_based",
            evidence_ids=[source.evidence_id],
            rank_score=0.5,
        )
        for label in source.candidates
    ]
    results = await compare_formula_candidates(
        parse_clinical_text("尚未录入脉证"), formulas, knowledge
    )
    relevant = [
        result
        for result in results
        if any(dimension.source_rule_id == source.rule_id for dimension in result.dimensions)
    ]
    assert relevant, source.rule_id
    assert len(relevant) == len(source.candidates) * (len(source.candidates) - 1) // 2
    for result in relevant:
        assert not result.decisive_support
        assert not result.decisive_contradictions
        assert {item.dimension for item in result.dimensions} == set(source.key_dimensions)
        for dimension in result.dimensions:
            assert dimension.status == "unknown"
            assert set(dimension.candidate_values).issubset(source.candidates)
            assert set(dimension.condition_assessments) == set(dimension.candidate_values)
            assert all(
                item.status == "unknown" for item in dimension.condition_assessments.values()
            )


async def test_alias_retrieval_and_condition_audit(knowledge: KnowledgeService) -> None:
    """别名解析映射真实来源，来源维度标题可归属单方但不补齐另一方。"""
    aliases = await knowledge.resolve_differential_candidates(["苓桂术甘汤", "苓桂枣甘汤"])
    assert aliases == {"苓桂术甘汤": "茯苓桂枝白术甘草汤证", "苓桂枣甘汤": "茯苓桂枝甘草大枣汤证"}
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=["unused"],
            rank_score=0.5,
        )
        for name in aliases
    ]
    results = await compare_formula_candidates(parse_clinical_text("脉沉紧"), formulas, knowledge)
    assert results
    dimension = next(item for item in results[0].dimensions if item.dimension == "苓桂术甘汤")
    label = aliases["苓桂术甘汤"]
    assert dimension.candidate_values[label].startswith("心脾阳虚")
    assert dimension.condition_assessments[label].kind == "mechanism"
    assert dimension.status == "unknown"


async def test_group_exact_priority_avoids_dependence_on_semantic_rank(
    knowledge: KnowledgeService,
) -> None:
    """成组规则完整覆盖当前候选对时可精确进入节点，不扩大在线语义召回上限。"""
    sources = await knowledge.search_differential(["麻黄汤", "小青龙汤"], top_k=1)
    assert sources[0].rule_id == "diff_v12_053"
    assert sources[0].retrieval_method == "candidate_group_exact"
