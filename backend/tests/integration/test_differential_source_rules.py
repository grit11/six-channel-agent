"""真实 53 条来源规则逐条节点回归。

通过 KnowledgeService 查询真实来源；仅验证归属、观察与保守执行行为，不冒充临床金标准。
"""

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.differential import source_candidate_label
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
    # 每条真实规则都放到超过旧截断上限的池尾，检查覆盖与候选位置无关。
    distractors = await knowledge.search_formula("发热", top_k=12)
    formulas = [
        FormulaCandidate(
            formula_name=str(item.metadata["formula_name"]),
            association_scope=str(item.metadata["association_scope"]),
            evidence_ids=[item.evidence_id],
            rank_score=0.5,
        )
        for item in distractors
        if source_candidate_label(str(item.metadata["formula_name"]), source.candidates) is None
    ] + formulas
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


async def test_pool_rules_do_not_depend_on_order_or_semantic_search(
    knowledge: KnowledgeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """完整绑定规则一次读取；无关语义召回失败不得影响 S4 精确鉴别。"""

    async def reject_search(*args: object, **kwargs: object) -> list:
        """候选鉴别不应触发语义检索。"""
        raise AssertionError("S4 不应逐对触发语义检索")

    monkeypatch.setattr(knowledge, "search_differential", reject_search)
    names = ["麻黄汤", "小青龙汤", "桂枝汤", "乌梅丸"]
    forward = await knowledge.differential_rules_for_pool(names)
    backward = await knowledge.differential_rules_for_pool(list(reversed(names)))
    assert [item.rule_id for item in forward] == [item.rule_id for item in backward]
    assert {item.rule_id for item in forward}.issuperset({"diff_v12_001", "diff_v12_053"})
    assert await knowledge.differential_rules_for_pool(["桂枝汤", "乌梅丸"]) == []
    sources = [
        await knowledge.get_differential_by_id(f"diff_v12_{number:03}") for number in range(1, 54)
    ]
    all_names = [name for source in sources if source is not None for name in source.candidates]
    complete = await knowledge.differential_rules_for_pool(all_names)
    assert len(complete) == 53
    assert len({item.rule_id for item in complete}) == 53


@pytest.mark.parametrize(
    ("names", "text", "dimension", "supported"),
    [
        (["桂枝汤", "麻黄汤"], "无汗", "汗", "麻黄汤证"),
        (["太阳蓄水证", "太阳蓄血证"], "小便不利", "小便", "太阳蓄水证"),
        (["太阳蓄水证", "太阳蓄血证"], "小便自利", "小便", "太阳蓄血证"),
    ],
)
async def test_distinct_source_rules_use_current_facts(
    knowledge: KnowledgeService, names: list[str], text: str, dimension: str, supported: str
) -> None:
    """不同来源规则动态归属同一核验器，未知候选条件不被当作阴性。"""
    sources = await knowledge.differential_rules_for_pool(names)
    assert sources
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=[sources[0].evidence_id],
            rank_score=0.5,
        )
        for name in names
    ]
    results = await compare_formula_candidates(parse_clinical_text(text), formulas, knowledge)
    current = next(
        item for result in results for item in result.dimensions if item.dimension == dimension
    )
    assert current.condition_assessments[supported].status == "supports"
    assert current.status == "supports"
