"""Top-K 工程契约回归：使用真实知识来源验证配对、三值缺失、关系和引用边界，不作临床评测。"""

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.topk_builder import build_topk_paths
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    DifferentialDimension,
    DifferentialResult,
    FormulaCandidate,
    SyndromeRelationship,
)


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """加载真实五类来源，测试不调用外部模型或修改原知识库。"""
    settings = Settings(_env_file=None)
    return await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")


def channel(name: str = "太阳", score: float = 0.8) -> ChannelCandidate:
    """构建上游候选测试输入，来源引用由各测试显式附加。"""
    return ChannelCandidate(
        candidate_id=f"channel_{name}",
        six_channels=[name],
        independent_score=score,
        supporting_findings=["发热"],
        missing_key_findings=["sweating"],
        evidence_completeness=0.5,
        status="active",
    )


async def formula(knowledge: KnowledgeService) -> FormulaCandidate:
    """用真实桂枝汤来源构造方证测试输入，避免创建虚构知识 ID。"""
    evidence = await knowledge.get_formula_by_name("桂枝汤")
    assert evidence is not None
    return FormulaCandidate(
        formula_name="桂枝汤",
        association_scope="classic_clause_based",
        canonical_six_channels=["太阳"],
        evidence_ids=[evidence.evidence_id],
        rank_score=0.6,
        supporting_findings=["汗出"],
        contradicting_findings=["无汗待核对"],
    )


@pytest.mark.asyncio
async def test_stable_paths_refs_unknown_and_raw_scores(knowledge: KnowledgeService) -> None:
    """相同输入保持路径身份，未知为缺失，反证不丢失，权重可审计且不按来源数加分。"""
    clinical = parse_clinical_text("发热，汗出不详")
    candidate = channel()
    clause = await knowledge.get_clause_by_id("35")
    assert clause is not None
    candidate.classic_evidence_ids = [clause.evidence_id, clause.evidence_id]
    first_formula = await formula(knowledge)
    inputs = (
        clinical,
        analyze_eight_principles(clinical),
        [candidate],
        [],
        [first_formula],
        [],
        knowledge,
    )
    paths, catalog = await build_topk_paths(*inputs, channel_weight=0.75)
    repeat, _ = await build_topk_paths(*inputs, channel_weight=0.75)
    assert len(paths) == 1
    assert paths[0].path_id == repeat[0].path_id
    assert paths[0].ranking_score == pytest.approx(0.5)
    assert paths[0].score_components["core_factor"] == 0.8
    assert paths[0].score_components["high_penalty"] == 0
    assert paths[0].score_components["medium_penalty"] == 0
    assert paths[0].classic_evidence == [clause.evidence_id]
    assert {item.evidence_id for item in catalog} == {
        clause.evidence_id,
        *first_formula.evidence_ids,
    }
    assert "sweating" in paths[0].missing_key_findings
    # 与本路径无关的全局未知不能挤入关键缺口；原始三值事实仍完整保留。
    assert "anal_burning" not in paths[0].missing_key_findings
    assert any(item.key == "anal_burning" for item in clinical.uncertain_findings)
    assert paths[0].contradicting_findings == ["无汗待核对"]
    assert paths[0].review_status == "pending"
    assert paths[0].completeness_detail is not None
    for condition in paths[0].completeness_detail.conditions:
        if condition.status == "unknown" and condition.kind != "symptom":
            assert condition.source_text not in paths[0].missing_key_findings
    assert paths[0].treatment_principle is None


@pytest.mark.asyncio
async def test_no_formula_weak_excluded_and_k_limits(knowledge: KnowledgeService) -> None:
    """弱候选不强配方，排除候选不进入输出，返回数量不凑满 K。"""
    clinical = parse_clinical_text("发热")
    weak, excluded = channel(), channel("阳明")
    weak.status, weak.supporting_findings = "weak", []
    excluded.status = "excluded"
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [weak, excluded],
        [],
        [await formula(knowledge)],
        [],
        knowledge,
        top_k=5,
    )
    assert len(paths) == 1
    assert paths[0].formula_pattern is None
    assert paths[0].ranking_score == weak.independent_score * 0.25
    assert paths[0].candidate_status == "weak"
    assert paths[0].separation_from_next is None


@pytest.mark.asyncio
async def test_historical_association_does_not_use_canonical(knowledge: KnowledgeService) -> None:
    """历史派生范围只使用历史归属；经典字段不能给历史关联跨经背书。"""
    clinical = parse_clinical_text("发热")
    evidence = await knowledge.get_formula_by_name("不二丹")
    assert evidence is not None
    historical = FormulaCandidate(
        formula_name="不二丹",
        association_scope="case_derived_historical",
        canonical_six_channels=["太阳"],
        case_derived_six_channels=["太阴", "少阴", "厥阴"],
        evidence_ids=[evidence.evidence_id],
        rank_score=0.6,
    )
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [channel(), channel("少阴")],
        [],
        [historical],
        [],
        knowledge,
        top_k=5,
    )
    by_channel = {path.six_channels[0]: path for path in paths}
    assert by_channel["太阳"].formula_pattern is None
    assert by_channel["少阴"].association_scope == "case_derived_historical"
    assert any("历史医案" in warning for warning in by_channel["少阴"].warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relationship_type", ["uncertain", "combined_disease", "sequential_concurrent_disease"]
)
async def test_relationship_not_inferred_from_multicandidates(
    knowledge: KnowledgeService, relationship_type: str
) -> None:
    """未定多候选不能合并；明确关系保留多经，不能任意配单经方证。"""
    clinical = parse_clinical_text("发热")
    relation = SyndromeRelationship.model_validate(
        {
            "relationship_type": relationship_type,
            "primary_candidate_id": "channel_太阳",
            "secondary_candidate_ids": ["channel_阳明"],
        }
    )
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [channel(), channel("阳明")],
        [relation],
        [await formula(knowledge)],
        [],
        knowledge,
        top_k=5,
    )
    combined = [item for item in paths if len(item.six_channels) > 1]
    if relationship_type == "uncertain":
        assert not combined
    else:
        assert len(combined) == 1
        assert combined[0].relationship == relationship_type
        assert combined[0].formula_pattern is None


@pytest.mark.asyncio
async def test_differential_requires_exact_pair(knowledge: KnowledgeService) -> None:
    """相关规则不得作为其他比较对的来源；精确对保留来源与缺失但不复制无归属正反证。"""
    clinical = parse_clinical_text("发热")
    dimension = DifferentialDimension(
        dimension="汗",
        observed_polarity="unknown",
        discriminator_weight=0.8,
        source_rule_id="diff_v12_001",
        status="unknown",
    )
    first = DifferentialResult(
        candidate_ids=["桂枝汤", "麻黄汤"],
        dimensions=[dimension],
        unresolved_dimensions=["汗"],
        decisive_support=["不可复制的组级摘要"],
    )
    unrelated = first.model_copy(update={"candidate_ids": ["桂枝汤", "理中汤"]})
    args = (
        clinical,
        analyze_eight_principles(clinical),
        [channel()],
        [],
        [await formula(knowledge)],
    )
    exact, _ = await build_topk_paths(*args, [first], knowledge)
    rejected, _ = await build_topk_paths(*args, [unrelated], knowledge)
    assert exact[0].differential_evidence == ["1.2:diff_v12_001"]
    assert "汗" in exact[0].missing_key_findings
    assert "不可复制的组级摘要" not in exact[0].supporting_findings
    assert rejected[0].differential_evidence == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["invented", "1.2:invented", "1.1:clause_35"])
async def test_missing_or_cross_version_refs_fail(knowledge: KnowledgeService, bad_id: str) -> None:
    """伪造或跨版本来源不能被默默丢弃或模糊召回替代。"""
    with pytest.raises(KnowledgeRetrievalError):
        await knowledge.get_evidence_by_ids([bad_id])


@pytest.mark.asyncio
async def test_wrong_source_category_fails(knowledge: KnowledgeService) -> None:
    """真实存在的方证 ID 也不能被放进经典来源类别。"""
    clinical = parse_clinical_text("发热")
    candidate = channel()
    candidate.classic_evidence_ids = (await formula(knowledge)).evidence_ids
    with pytest.raises(KnowledgeRetrievalError):
        await build_topk_paths(
            clinical, analyze_eight_principles(clinical), [candidate], [], [], [], knowledge
        )


@pytest.mark.asyncio
async def test_formula_source_identity_mismatch_fails(knowledge: KnowledgeService) -> None:
    """方名与引用不一致时明确失败，不能用真实 ID 为其他方证背书。"""
    clinical = parse_clinical_text("发热")
    mismatched = await formula(knowledge)
    mismatched.formula_name = "麻黄汤"
    with pytest.raises(KnowledgeRetrievalError):
        await build_topk_paths(
            clinical,
            analyze_eight_principles(clinical),
            [channel()],
            [],
            [mismatched],
            [],
            knowledge,
        )


@pytest.mark.asyncio
async def test_rank_separation_and_empty_candidates(knowledge: KnowledgeService) -> None:
    """排序和下一候选分差来自同一完整列表，无候选时允许空输出。"""
    clinical = parse_clinical_text("发热")
    args = (clinical, analyze_eight_principles(clinical))
    paths, _ = await build_topk_paths(
        *args, [channel(score=0.9), channel("阳明", 0.7)], [], [], [], knowledge, top_k=1
    )
    assert paths[0].rank == 1
    assert paths[0].separation_from_next == pytest.approx(0.05)
    assert await build_topk_paths(*args, [], [], [], [], knowledge) == ([], [])
