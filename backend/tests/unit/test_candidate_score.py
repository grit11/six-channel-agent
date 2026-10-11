"""RP-13 评分需求回归：真实来源复算、Unknown/冲突不惩罚、来源核验先于 Top-K。

排除标注仅在复制的真实来源测试夹具中附加以验证机制，不修改主库或宣称专家已审核。
"""

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.topk_builder import build_topk_paths
from backend.app.schemas.knowledge import KnowledgeRecord
from backend.app.schemas.reasoning import ChannelCandidate, FormulaCandidate, ScoringPolicy


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """通过唯一知识边界加载实际 V1.2 来源。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


async def _inputs(knowledge: KnowledgeService) -> tuple[ChannelCandidate, list[FormulaCandidate]]:
    """构造同经两条方证路径，方名和 ID 均取真实来源。"""
    channel = ChannelCandidate(
        candidate_id="solar",
        six_channels=["太阳"],
        independent_score=0.8,
        pass_a_score=0.8,
        supporting_findings=["发热"],
        evidence_completeness=0.5,
        status="active",
    )
    formulas = []
    for name in ("桂枝汤", "麻黄汤"):
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
    return channel, formulas


@pytest.mark.asyncio
async def test_scores_recompute_and_source_polarity_changes_ranking(
    knowledge: KnowledgeService,
) -> None:
    """明确无汗产生桂枝方证反证且支持麻黄路径；分项可直接复算，去重不改变分。"""
    clinical = parse_clinical_text("发热，无汗，脉浮紧")
    channel, formulas = await _inputs(knowledge)
    differentials = await compare_formula_candidates(clinical, formulas, knowledge)
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [channel],
        [],
        formulas,
        differentials,
        knowledge,
    )
    assert paths[0].formula_pattern == "麻黄汤"
    by_name = {path.formula_pattern: path for path in paths}
    assert by_name["桂枝汤"].score_components["high_penalty"] > 0
    for path in paths:
        components = path.score_components
        positive = sum(
            components[f"{name}_factor"] * components[f"{name}_weight"]
            for name in (
                "core",
                "differential",
                "classic",
                "formula",
                "temporal",
                "case",
                "context",
            )
        )
        assert path.ranking_score == pytest.approx(
            max(0, min(1, positive - components["high_penalty"] - components["medium_penalty"]))
        )
    for formula in formulas:
        formula.evidence_ids *= 2
    repeated, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [channel],
        [],
        formulas,
        differentials * 2,
        knowledge,
    )
    assert [path.ranking_score for path in repeated] == [path.ranking_score for path in paths]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["汗出不详", "昨日无汗", "汗出，无汗"])
async def test_unknown_history_and_conflicts_never_become_penalty(
    knowledge: KnowledgeService,
    text: str,
) -> None:
    """未知、历史与同时间冲突均不计入明确反证。"""
    clinical = parse_clinical_text(text)
    channel, formulas = await _inputs(knowledge)
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [channel],
        [],
        formulas,
        await compare_formula_candidates(clinical, formulas, knowledge),
        knowledge,
    )
    assert all(
        path.score_components["high_penalty"] == 0 and path.score_components["medium_penalty"] == 0
        for path in paths
    )


@pytest.mark.asyncio
async def test_unknown_does_not_reduce_known_positive_factor(knowledge: KnowledgeService) -> None:
    """增加未问事实只改变缺口，不能降低已知支持比例。"""
    channel, formulas = await _inputs(knowledge)
    scores = []
    for text in ("发热，无汗", "发热，无汗，喘不详"):
        clinical = parse_clinical_text(text)
        paths, _ = await build_topk_paths(
            clinical,
            analyze_eight_principles(clinical),
            [channel],
            [],
            formulas,
            [],
            knowledge,
        )
        scores.append({p.formula_pattern: p.ranking_score for p in paths})
    assert scores[0] == scores[1]


@pytest.mark.asyncio
async def test_grounded_exclusion_is_audited_and_unknown_does_not_trigger(
    knowledge: KnowledgeService,
) -> None:
    """真实来源夹具增加审核标注后核验 gate；普通主库冲突不自行产生硬排除。"""
    clause = await knowledge.get_clause_by_id("35")
    formula_source = await knowledge.get_formula_by_name("麻黄汤")
    assert clause is not None and formula_source is not None
    records = []
    for source in (clause, formula_source):
        metadata = dict(source.metadata)
        if source is clause:
            metadata.update(
                core_conditions_source_grounded=["无汗"], exclusion_review_status="expert_approved"
            )
        records.append(
            KnowledgeRecord(
                knowledge_id=source.knowledge_id,
                knowledge_version="1.2",
                knowledge_type=source.knowledge_type,
                source=source.source,
                source_level=source.source_level,
                content=source.content,
                embedding_text=source.content,
                metadata=metadata,
            )
        )
    fixture = KnowledgeService(records, "1.2")
    channel, formulas = await _inputs(knowledge)
    channel.classic_evidence_ids = [clause.evidence_id]
    for text, expected in (("汗出", 1), ("汗出不详", 0), ("汗出，无汗", 0)):
        clinical = parse_clinical_text(text)
        excluded = []
        paths, catalog = await build_topk_paths(
            clinical,
            analyze_eight_principles(clinical),
            [channel],
            [],
            [formulas[1]],
            [],
            fixture,
            excluded_paths=excluded,
        )
        assert len(excluded) == expected
        if expected:
            assert paths == []
            assert clause.evidence_id in excluded[0].exclusion_reasons[0]
            assert {s.evidence_id for s in catalog}.issuperset({clause.evidence_id})


@pytest.mark.asyncio
async def test_lower_core_path_can_enter_topk_after_source_validation(
    knowledge: KnowledgeService,
) -> None:
    """先截断再核验会遗漏来源更充分的候选，此回归要求先对所有路径评分。"""
    clinical = parse_clinical_text("发热")
    low, _ = await _inputs(knowledge)
    low.independent_score, low.pass_a_score = 0.2, 0.2
    low.classic_evidence_ids = ["1.2:clause_35"]
    high = low.model_copy(
        update={
            "candidate_id": "higher-core",
            "pass_a_score": 0.9,
            "independent_score": 0.9,
            "classic_evidence_ids": [],
        }
    )
    paths, _ = await build_topk_paths(
        clinical, analyze_eight_principles(clinical), [high, low], [], [], [], knowledge, top_k=1
    )
    assert paths[0].channel_candidate_ids == ["solar"]


@pytest.mark.asyncio
async def test_ordinary_conflict_never_hard_excludes(knowledge: KnowledgeService) -> None:
    """主库无专家核心标注时，明确条文冲突可以扣分但不能硬排除。"""
    channel, formulas = await _inputs(knowledge)
    channel.classic_evidence_ids = ["1.2:clause_35"]
    clinical = parse_clinical_text("汗出")
    excluded = []
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [channel],
        [],
        [formulas[1]],
        [],
        knowledge,
        excluded_paths=excluded,
    )
    assert paths and excluded == []
    assert paths[0].score_components["medium_penalty"] > 0


def test_scoring_policy_rejects_bad_weights_and_penalties() -> None:
    """集中配置不允许非单位权重或高反证惩罚小于普通支持。"""
    with pytest.raises(ValidationError):
        ScoringPolicy(core=0.9)
    with pytest.raises(ValidationError):
        ScoringPolicy(high_penalty=0.1)
