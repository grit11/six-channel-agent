"""审查问题的需求回归：使用真实来源，验证事实语义与候选特异鉴别，不作临床金标准。"""

from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.channel_candidate import generate_channel_candidates
from backend.app.nodes.clinical_parser import match_source_phrase, parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.formula_candidate import generate_formula_candidates
from backend.app.schemas.clinical import ClinicalEvidence, Finding
from backend.app.schemas.reasoning import ChannelCandidate, FormulaCandidate


@pytest.fixture
async def knowledge(tmp_path: Path) -> KnowledgeService:
    """离线读取实际知识快照，不调用在线模型。"""
    settings = Settings(_env_file=None, retrieval_mode="bm25", reranker_enabled=False)
    return await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")


def test_compound_pulse_preserves_components_and_original() -> None:
    """C01：浮紧有两项明确脉象，不凭浮脉补成缓脉或弱脉。"""
    evidence = parse_clinical_text("脉浮紧")
    known = {finding.key: finding for finding in evidence.positive_findings}
    assert {"pulse_floating", "pulse_tight"}.issubset(known)
    assert known["pulse_tight"].source_text == "脉浮紧"
    assert "pulse_slow" not in known


def test_negation_is_local_and_temporal_occurrences_are_retained() -> None:
    """C02：昨日与今日分别绑定否定和时间，不覆盖历史阳性，不制造同时间冲突。"""
    evidence = parse_clinical_text("昨日发热，今日无发热")
    fever = [item for item in evidence.findings if item.key == "fever"]
    assert [(item.polarity, item.time_scope) for item in fever] == [
        ("positive", "昨日"),
        ("negative", "今日"),
    ]
    assert fever[1].source_text == "无发热"
    assert evidence.contradictions == []


def test_repeated_current_fact_conflicts_do_not_support_an_axis() -> None:
    """同时间肯定与否定都保留，但不能在冲突未解决时提供单向支持。"""
    evidence = parse_clinical_text("发热，无发热")
    assert len(evidence.contradictions) == 1
    state = analyze_eight_principles(evidence)
    assert state.cold_heat.score_right == 0


def test_eight_principles_yang_label_and_insufficient_single_fact() -> None:
    """C03：阴阳轴左阴右阳；单一症状仍为未定，摘要显示医学轴名。"""
    state = analyze_eight_principles(parse_clinical_text("发热，脉浮"))
    assert set(state.yin_yang.supporting_findings) == {"阳:fever", "阳:pulse_floating"}
    assert state.yin_yang.lean == "right"
    assert "阴阳:阳倾向" in state.summary
    assert analyze_eight_principles(parse_clinical_text("发热")).yin_yang.lean == "uncertain"


@pytest.mark.asyncio
async def test_formula_does_not_invent_compound_pulse_or_penalize_required_negative(
    knowledge: KnowledgeService,
) -> None:
    """C04：只有浮脉不能命中浮紧；麻黄汤来源的无汗要求不是禁忌。"""
    evidence = parse_clinical_text("发热，无汗，脉浮")
    channel = ChannelCandidate(
        candidate_id="solar",
        six_channels=["太阳"],
        independent_score=0.8,
        evidence_completeness=0.5,
        status="active",
    )
    formulas = await generate_formula_candidates(evidence, [channel], knowledge, top_k=5)
    assert formulas
    assert all(
        term not in item.supporting_findings
        for item in formulas
        for term in ("脉浮紧", "脉浮缓", "脉浮弱")
    )
    mahuang = next(item for item in formulas if item.formula_name == "麻黄汤")
    assert "无汗" in mahuang.supporting_findings
    assert "无汗" not in mahuang.contradicting_findings
    assert "无汗而喘者" not in mahuang.contradicting_findings


@pytest.mark.asyncio
async def test_formula_normalizes_grounded_negative_conditions_without_treating_them_as_contra(
    knowledge: KnowledgeService,
) -> None:
    """来源阴性字段中的完整条件按当前极性核对，复杂说明保留缺口。"""
    formulas = await generate_formula_candidates(
        parse_clinical_text("发热，无汗，脉浮"),
        [
            ChannelCandidate(
                candidate_id="solar",
                six_channels=["太阳"],
                independent_score=0.8,
                evidence_completeness=0.5,
                status="active",
            )
        ],
        knowledge,
        top_k=5,
    )
    mahuang = next(item for item in formulas if item.formula_name == "麻黄汤")
    assert "无汗" in mahuang.supporting_findings
    assert all("禁忌" not in item for item in mahuang.contradicting_findings)


@pytest.mark.asyncio
async def test_differential_splits_candidate_values_and_records_counterevidence(
    knowledge: KnowledgeService,
) -> None:
    """C05：精确桂枝/麻黄规则的汗与脉必须按候选拆分，当前无汗有明确反证。"""
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=["unused"],
            rank_score=0.5,
        )
        for name in ("桂枝汤", "麻黄汤")
    ]
    results = await compare_formula_candidates(
        parse_clinical_text("无汗，脉浮"), formulas, knowledge
    )
    assert len(results) == 1
    sweat = next(item for item in results[0].dimensions if item.dimension == "汗")
    assert sweat.candidate_values == {"桂枝汤证": "汗出", "麻黄汤证": "无汗"}
    assert sweat.status == "supports"
    assert sweat.observed_polarity == "negative"
    assert any("麻黄汤" in item for item in results[0].decisive_support)
    assert any("桂枝汤" in item for item in results[0].decisive_contradictions)
    pulse = next(item for item in results[0].dimensions if item.dimension == "脉")
    assert pulse.status == "unknown"


@pytest.mark.asyncio
async def test_full_formula_pool_reaches_late_guizhi_mahuang_differential(
    knowledge: KnowledgeService,
) -> None:
    """完整 S3 方证池不能截断 S4；汗出、浮缓应触发桂枝/麻黄鉴别。"""
    evidence = parse_clinical_text("起病两天，发热恶风，微微汗出，头痛，脉浮而缓")
    channel = ChannelCandidate(
        candidate_id="solar",
        six_channels=["太阳"],
        independent_score=0.8,
        supporting_findings=["发热", "aversion_to_wind", "sweating"],
        evidence_completeness=0.5,
        status="active",
    )
    formulas = await generate_formula_candidates(evidence, [channel], knowledge, top_k=12)
    assert "麻黄汤" in {item.formula_name for item in formulas}
    results = await compare_formula_candidates(evidence, formulas, knowledge)
    pair = next(item for item in results if {"桂枝汤", "麻黄汤"}.issubset(item.candidate_ids))
    sweat = next(item for item in pair.dimensions if item.dimension == "汗")
    assert sweat.status == "supports"
    assert any("桂枝汤" in item for item in pair.decisive_support)
    assert any("麻黄汤" in item for item in pair.decisive_contradictions)


@pytest.mark.asyncio
async def test_differential_preserves_observed_facts_in_complex_conditions(
    knowledge: KnowledgeService,
) -> None:
    """长说明中的已记录词项可展示，但整体条件未完整规范化时仍保持未定。"""
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=["unused"],
            rank_score=0.5,
        )
        for name in ("葛根黄芩黄连汤", "麻黄杏仁甘草石膏汤")
    ]
    results = await compare_formula_candidates(parse_clinical_text("下利，喘"), formulas, knowledge)
    assert results
    dimension = next(item for item in results[0].dimensions if item.dimension == "主症主次")
    assert dimension.observed_value is not None
    assert "下利" in dimension.observed_value
    assert dimension.status == "unknown"


@pytest.mark.asyncio
async def test_differential_does_not_duplicate_existing_formula_suffix(
    knowledge: KnowledgeService,
) -> None:
    """已带“证”的方名仍能精确命中鉴别规则。"""
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=["unused"],
            rank_score=0.5,
        )
        for name in ("桂枝汤证", "麻黄汤")
    ]
    results = await compare_formula_candidates(parse_clinical_text("无汗"), formulas, knowledge)
    assert results


@pytest.mark.asyncio
async def test_unrelated_rules_cannot_be_applied_to_candidate_pair(
    knowledge: KnowledgeService,
) -> None:
    """没有包含当前双方候选的规则时，相关召回不能冒充当前候选对鉴别。"""
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=["unused"],
            rank_score=0.5,
        )
        for name in ("桂枝汤", "乌梅丸")
    ]
    assert await compare_formula_candidates(parse_clinical_text("无汗"), formulas, knowledge) == []


@pytest.mark.asyncio
async def test_empty_current_evidence_receives_no_retrieval_bonus(
    knowledge: KnowledgeService,
) -> None:
    """C06：历史和推测事实不支持当前候选；文档召回数量不能制造支持分。"""
    evidence = ClinicalEvidence(
        findings=[
            Finding(key="fever", value=True, polarity="positive", source="inference"),
            Finding(
                key="chills", value=True, polarity="positive", source="record", time_scope="昨日"
            ),
        ]
    )
    candidates = await generate_channel_candidates(
        evidence, analyze_eight_principles(evidence), knowledge, top_k=3
    )
    assert all(item.independent_score == 0 for item in candidates)
    assert all(not item.supporting_findings for item in candidates)


def test_structured_snapshot_recomputes_conflicts_after_edit() -> None:
    """C07：反序列化以当前 findings 重建冲突，不信任过时的客户端列表。"""
    evidence = parse_clinical_text("汗出，无汗")
    payload = evidence.model_dump()
    for finding in payload["findings"]:
        if finding["key"] == "sweating":
            finding["polarity"] = "unknown"
            finding["value"] = None
    assert ClinicalEvidence.model_validate(payload).contradictions == []


def test_repeated_positive_occurrences_do_not_duplicate_axis_scores() -> None:
    """反复记录同一症状不能凭出现次数强化轴向或绕过单症状未定约束。"""
    state = analyze_eight_principles(parse_clinical_text("发热，发热，发热"))
    assert state.yin_yang.lean == "uncertain"
    assert state.yin_yang.score_right == 0.5


def test_negative_compound_pulse_does_not_deny_each_component() -> None:
    """否认浮紧不等于否认浮脉及紧脉，未解释复合事实应保持可核对的独立键。"""
    evidence = parse_clinical_text("脉浮，但不是脉浮紧")
    assert not any(item.key == "pulse_floating" for item in evidence.negative_findings)


def test_partial_negative_requirement_needs_all_required_components() -> None:
    """无汗而喘需两项共同支持，只有无汗不能支持整段来源要求。"""
    assert match_source_phrase("无汗而喘者", parse_clinical_text("无汗")) == "unknown"
    assert match_source_phrase("无汗而喘者", parse_clinical_text("无汗，喘")) == "supports"
