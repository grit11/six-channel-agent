"""路径摘要回归：真实来源计数、次选鉴别与治法资料保守归属，不作为临床评测。"""

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.path_summary import (
    compare_next_path,
    summarize_completeness,
    summarize_treatment,
)
from backend.app.nodes.topk_builder import build_topk_paths
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import EvidenceAssessment
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    EvidenceGap,
    FormulaCandidate,
    ReasoningPath,
)


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """读取真实知识，夹具只复制 DTO 不改原始文件。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


def _channel() -> ChannelCandidate:
    """提供具有患者支持的上游候选。"""
    return ChannelCandidate(
        candidate_id="channel_太阳",
        six_channels=["太阳"],
        independent_score=0.8,
        supporting_findings=["发热"],
        evidence_completeness=0.9,
        status="active",
    )


async def _formulas(knowledge: KnowledgeService, names: list[str]) -> list[FormulaCandidate]:
    """只绑定真实来源和来源六经，不构造虚构引用。"""
    result = []
    for name in names:
        source = await knowledge.get_formula_by_name(name)
        assert source is not None
        result.append(
            FormulaCandidate(
                formula_name=name,
                association_scope="classic_clause_based",
                canonical_six_channels=source.metadata["canonical_six_channels"],
                evidence_ids=[source.evidence_id],
                rank_score=0.5,
            )
        )
    return result


async def _path(knowledge: KnowledgeService) -> ReasoningPath:
    """生成具有真实方证核验的摘要测试路径。"""
    clinical = parse_clinical_text("发热，无汗，恶寒，喘，咳嗽")
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [_channel()],
        [],
        await _formulas(knowledge, ["小青龙汤"]),
        [],
        knowledge,
    )
    return paths[0]


@pytest.mark.asyncio
async def test_completeness_counts_known_not_matches_and_deduplicates(
    knowledge: KnowledgeService,
) -> None:
    """反证是已知；重复来源不增完整度，缺失不能被无关明确事实填满。"""
    source = await knowledge.get_clause_by_id("35")
    assert source is not None
    candidate = _channel()
    candidate.classic_evidence_ids = [source.evidence_id, source.evidence_id]
    clinical = parse_clinical_text("发热，汗出")
    paths, catalog = await build_topk_paths(
        clinical, analyze_eight_principles(clinical), [candidate], [], [], [], knowledge
    )
    path = paths[0]
    detail = path.completeness_detail
    assert detail is not None
    assert detail.supported_count == 1 and detail.contradicted_count == 1
    assert (
        detail.total_count
        == detail.supported_count + detail.contradicted_count + detail.unknown_count
    )
    assert path.evidence_completeness == pytest.approx(2 / detail.total_count)
    repeated = path.model_copy(deep=True)
    repeated.evidence_assessments *= 2
    assert summarize_completeness(repeated, catalog, [], []).model_dump() == detail.model_dump()
    assert "无汗" not in path.supporting_findings
    assert detail.unknown_count > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["无汗不详", "昨日无汗", "汗出，无汗"])
async def test_unknown_historical_conflicting_conditions_stay_unknown(
    knowledge: KnowledgeService, text: str
) -> None:
    """历史、冲突和 unknown 不提供当前单向支持或反证，不提升来源覆盖率。"""
    source = await knowledge.get_clause_by_id("35")
    assert source is not None
    candidate = _channel()
    candidate.classic_evidence_ids = [source.evidence_id]
    clinical = parse_clinical_text(text)
    paths, _ = await build_topk_paths(
        clinical, analyze_eight_principles(clinical), [candidate], [], [], [], knowledge
    )
    assert paths[0].evidence_completeness == 0
    assert paths[0].completeness_detail.unknown_count == paths[0].completeness_detail.total_count


@pytest.mark.asyncio
async def test_unavailable_preserves_denominator_and_old_snapshot_defaults(
    knowledge: KnowledgeService,
) -> None:
    """无法获取仍是 Unknown；历史响应缺字段可加载，零来源不冒充完整。"""
    path = await _path(knowledge)
    sources = await knowledge.get_evidence_by_ids(path.formula_evidence)
    before = summarize_completeness(path, sources, [], ["起病先后未定"])
    keys = next(
        condition.alternatives[0]
        for assessment in path.evidence_assessments
        for condition in assessment.condition_assessments
        if condition.alternatives
    )
    gap = EvidenceGap(
        question="核对汗",
        dimension="汗",
        priority=0.5,
        candidate_coverage=2,
        finding_keys=list(keys),
        status="unavailable",
    )
    # 用缺汗的真实核验重新形成未决条件。
    path.evidence_assessments = await knowledge.assess_candidate_evidence(
        path.formula_evidence,
        ClinicalEvidence(),
        channels=path.six_channels,
        formula_name=path.formula_pattern,
    )
    after = summarize_completeness(path, sources, [gap], ["起病先后未定"])
    assert after.total_count == before.total_count
    assert any(item.unavailable for item in after.conditions)
    assert after.missing_timeline == ["起病先后未定"]
    payload = path.model_dump()
    for field in [
        "completeness_detail",
        "comparison_with_next",
        "treatment_summary",
        "summary_version",
    ]:
        payload.pop(field)
    old = ReasoningPath.model_validate(payload)
    assert old.completeness_detail is None and old.treatment_summary is None
    old.evidence_assessments = []
    assert summarize_completeness(old, [], [], []).total_count == 0


@pytest.mark.asyncio
async def test_next_comparison_uses_true_rule_and_keeps_unreturned_next(
    knowledge: KnowledgeService,
) -> None:
    """Top1 仍与未返回次选比较；汗状态重新核验，不使用伪造的上游组级结论。"""
    clinical = parse_clinical_text("发热，无汗，身痛，喘，脉浮紧")
    formulas = await _formulas(knowledge, ["麻黄汤", "桂枝汤"])
    differential = await compare_formula_candidates(clinical, formulas, knowledge)
    paths, _ = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [_channel()],
        [],
        formulas,
        differential,
        knowledge,
        top_k=2,
    )
    selected, catalog = await build_topk_paths(
        clinical,
        analyze_eight_principles(clinical),
        [_channel()],
        [],
        formulas,
        differential,
        knowledge,
        top_k=1,
    )
    assert selected[0].path_id == paths[0].path_id
    comparison = selected[0].comparison_with_next
    assert comparison is not None and comparison.next_path_id == paths[1].path_id
    sweat = next(item for item in comparison.differences if item.dimension == "汗")
    assert (
        sweat.conclusion == "favors_current"
        and sweat.current_status == "supports"
        and sweat.next_status == "contradicts"
    )
    assert sweat.evidence_id in {item.evidence_id for item in catalog}
    assert all(item.dimension not in {"病机", "追问重点"} for item in comparison.differences)
    # 即使上游把 observed/status 改成支持，摘要仍按医生未知事实重新核验。
    for result in differential:
        for dimension in result.dimensions:
            dimension.status = "supports"
            dimension.observed_value = "伪造观察"
    unknown, _ = await compare_next_path(
        paths[0], paths[1], ClinicalEvidence(), differential, knowledge
    )
    assert all(item.conclusion == "unresolved" for item in unknown.differences)


@pytest.mark.asyncio
async def test_comparison_without_pair_source_does_not_invent_difference(
    knowledge: KnowledgeService,
) -> None:
    """分差、同方不同经和无方不能替代真实鉴别来源。"""
    path = await _path(knowledge)
    second = path.model_copy(update={"path_id": "other"})
    same, _ = await compare_next_path(path, second, ClinicalEvidence(), [], knowledge)
    assert same.status == "same_formula" and same.differences == []
    second.formula_pattern = "麻黄汤"
    missing, _ = await compare_next_path(path, second, ClinicalEvidence(), [], knowledge)
    assert missing.status == "no_source_rule" and not missing.differences
    second.formula_pattern = None
    assert (await compare_next_path(path, second, ClinicalEvidence(), [], knowledge))[
        0
    ].status == "no_formula"


@pytest.mark.asyncio
async def test_treatment_unique_short_source_with_patient_support(
    knowledge: KnowledgeService,
) -> None:
    """唯一短句可摘录；不改变排序，不把来源治法当患者处方。"""
    path = await _path(knowledge)
    assert path.treatment_summary is not None
    assert path.treatment_summary.status == "available"
    assert path.treatment_principle == "辛温解表，温化水饮。"
    assert path.treatment_summary.evidence_ids == path.formula_evidence
    sources = await knowledge.get_evidence_by_ids(path.formula_evidence)
    assert path.treatment_principle in sources[0].content
    path.evidence_assessments = [
        EvidenceAssessment(
            evidence_id=sources[0].evidence_id, grade="background", rationale="无患者支持"
        )
    ]
    assert summarize_treatment(path, sources).status == "insufficient_support"
    path.candidate_status = "excluded"
    assert summarize_treatment(path, sources).text is None


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["桂枝汤", "麻黄汤", "小柴胡汤"])
async def test_aggregated_treatments_remain_ambiguous(
    knowledge: KnowledgeService, name: str
) -> None:
    """当前库中混入其他方剂治法，不按第一条或相似词猜测归属。"""
    path = await _path(knowledge)
    source = await knowledge.get_formula_by_name(name)
    assert source is not None
    path.formula_pattern, path.formula_evidence = name, [source.evidence_id]
    summary = summarize_treatment(path, [source])
    assert summary.status == "ambiguous" and summary.text is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["每服三两", "温服10克每日三次", "先温里，后解表。", "不存在于来源的治法"]
)
async def test_treatment_rejects_dose_branch_or_ungrounded_text(
    knowledge: KnowledgeService, text: str
) -> None:
    """剂量、治疗次序和未对齐元数据不进入治法摘要，测试仅复制来源。"""
    path = await _path(knowledge)
    source = (await knowledge.get_evidence_by_ids(path.formula_evidence))[0]
    changed = source.model_copy(
        update={"metadata": {**source.metadata, "treatment_method": [text]}}
    )
    assert summarize_treatment(path, [changed]).status == "ambiguous"
    path.association_scope = "case_derived_historical"
    assert summarize_treatment(path, [source]).status == "historical"
