"""路径适用性回归：使用真实知识来源核对极性、未知、时间及归属，不作临床评测。"""

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.topk_builder import build_topk_paths
from backend.app.schemas.reasoning import ChannelCandidate, FormulaCandidate


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """从现有五类主库加载不可变来源，测试不调用外部模型。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


@pytest.mark.parametrize(
    ("text", "grade", "supported", "contradicted"),
    [
        ("头痛，发热，骨节疼痛，恶风，无汗，腰痛，喘", "direct", True, False),
        ("发热，无汗", "indirect", True, False),
        ("发热，汗出", "contradict", True, True),
        ("昨日无汗，今日汗出不详", "background", False, False),
        ("汗出，无汗", "background", False, False),
        ("汗出不详", "background", False, False),
    ],
)
async def test_classic_current_polarity_and_unknown(
    knowledge: KnowledgeService, text: str, grade: str, supported: bool, contradicted: bool
) -> None:
    """第35条有汗与无汗不能子串混淆；未知、历史、冲突均不形成单向支持。"""
    source = await knowledge.get_clause_by_id("35")
    assert source is not None
    assessments = await knowledge.assess_candidate_evidence(
        [source.evidence_id, source.evidence_id],
        parse_clinical_text(text),
        channels=["太阳"],
        formula_name="麻黄汤",
    )
    assert len(assessments) == 1
    assessment = assessments[0]
    assert assessment.grade == grade
    assert bool(assessment.supporting_requirements) is supported
    assert bool(assessment.contradicting_requirements) is contradicted
    if contradicted:
        assert "无汗" in assessment.contradicting_requirements
    again = await knowledge.get_clause_by_id("35")
    assert again == source


async def test_scope_relevance_is_not_direct_support(knowledge: KnowledgeService) -> None:
    """跨经无关条文被过滤；不同方名和跨经路径不能借条文标签成为直接支持。"""
    clinical = parse_clinical_text("头痛，发热，骨节疼痛，恶风，无汗，腰痛，喘")
    for channels, formula, expected in [
        (["少阴"], "麻黄汤", "irrelevant"),
        (["太阳"], "桂枝汤", "indirect"),
        (["太阳", "阳明"], "麻黄汤", "indirect"),
    ]:
        result = await knowledge.assess_candidate_evidence(
            ["1.2:clause_35"], clinical, channels=channels, formula_name=formula
        )
        assert result[0].grade == expected
    with pytest.raises(KnowledgeRetrievalError):
        await knowledge.assess_candidate_evidence(["1.2:invented"], clinical, channels=["太阳"])


async def test_formula_composite_and_differential_ownership(knowledge: KnowledgeService) -> None:
    """浮脉不能补齐浮紧；同一鉴别来源在不同候选上下文有不同用途。"""
    clinical = parse_clinical_text("发热，无汗，脉浮")
    source = await knowledge.get_formula_by_name("麻黄汤")
    assert source is not None
    assessment = (
        await knowledge.assess_candidate_evidence(
            [source.evidence_id], clinical, channels=["太阳"], formula_name="麻黄汤"
        )
    )[0]
    assert "脉浮紧" in assessment.unresolved_requirements
    assert "脉浮紧" not in assessment.supporting_requirements
    for formula, grade in [("麻黄汤", "indirect"), ("桂枝汤", "contradict")]:
        result = await knowledge.assess_candidate_evidence(
            ["1.2:diff_v12_001"], clinical, channels=["太阳"], formula_name=formula
        )
        assert result[0].grade == grade
        assert ("汗出" in result[0].contradicting_requirements) == (formula == "桂枝汤")


async def test_historical_and_framework_are_not_direct(knowledge: KnowledgeService) -> None:
    """真实历史来源与框架标签均不能升级为当前患者的经典直接支持。"""
    clinical = parse_clinical_text("下利，乏力，四肢厥冷，脉细")
    sources = await knowledge.search_case("下利", channels=["少阴"], top_k=3)
    sources += await knowledge.search_core_framework("少阴", top_k=2)
    assert sources
    results = await knowledge.assess_candidate_evidence(
        [source.evidence_id for source in sources], clinical, channels=["少阴"]
    )
    assert all(item.grade != "direct" for item in results)


async def test_topk_attaches_assessments_and_filters_only_irrelevant(
    knowledge: KnowledgeService,
) -> None:
    """路径核验接入主链；反证仍供审核，无关来源从有效引用退出且留审计。"""
    clinical = parse_clinical_text("发热，汗出")
    candidate = ChannelCandidate(
        candidate_id="channel_少阴",
        six_channels=["少阴"],
        independent_score=0.4,
        supporting_findings=["发热"],
        status="active",
        evidence_completeness=0.5,
        classic_evidence_ids=["1.2:clause_35"],
    )
    paths, catalog = await build_topk_paths(
        clinical, analyze_eight_principles(clinical), [candidate], [], [], [], knowledge
    )
    assert paths[0].classic_evidence == []
    assert paths[0].evidence_assessments[0].grade == "irrelevant"
    assert paths[0].evidence_assessments[0].evidence_id == "1.2:clause_35"
    assert catalog == []
    candidate.six_channels = ["太阳"]
    paths, catalog = await build_topk_paths(
        clinical, analyze_eight_principles(clinical), [candidate], [], [], [], knowledge
    )
    assert paths[0].classic_evidence == ["1.2:clause_35"]
    assert paths[0].evidence_assessments[0].grade == "contradict"
    assert catalog[0].evidence_id == "1.2:clause_35"


async def test_same_rule_can_support_and_contradict_distinct_paths(
    knowledge: KnowledgeService,
) -> None:
    """候选特异分级不能污染共享 Evidence Catalog；同一来源按每条路径分别保存。"""
    clinical = parse_clinical_text("发热，无汗，脉浮紧")
    formulas = []
    for name in ["桂枝汤", "麻黄汤"]:
        source = await knowledge.get_formula_by_name(name)
        assert source is not None
        formulas.append(
            FormulaCandidate(
                formula_name=name,
                association_scope="classic_clause_based",
                canonical_six_channels=["太阳"],
                evidence_ids=[source.evidence_id],
                rank_score=0.5,
            )
        )
    candidate = ChannelCandidate(
        candidate_id="channel_太阳",
        six_channels=["太阳"],
        independent_score=0.8,
        supporting_findings=["发热"],
        status="active",
        evidence_completeness=0.5,
    )
    paths, catalog = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [candidate],
        [],
        formulas,
        await compare_formula_candidates(clinical, formulas, knowledge),
        knowledge,
    )
    assessments = {
        path.formula_pattern: next(
            item for item in path.evidence_assessments if item.evidence_id == "1.2:diff_v12_001"
        )
        for path in paths
    }
    assert assessments["麻黄汤"].grade == "indirect"
    assert assessments["桂枝汤"].grade == "contradict"
    assert len([item for item in catalog if item.evidence_id == "1.2:diff_v12_001"]) == 1
