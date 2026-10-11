"""RP-11 工程回归：排序复算、来源忠实性、三值语义与受控停止，不替代临床评测。"""

from pathlib import Path

import pytest

from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.evidence_gap import build_evidence_gaps, decide_evidence_gap_stop
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import DifferentialRule
from backend.app.schemas.reasoning import (
    DifferentialResult,
    EvidenceGapContext,
    EvidenceGapPolicy,
    FormulaCandidate,
)
from backend.app.services.reasoning_service import ReasoningService

KNOWLEDGE_DIRECTORY = Path(__file__).resolve().parents[3] / "knowledge/v1.2"


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """读取真实来源，不修改主知识库。"""
    return await KnowledgeService.from_directory(KNOWLEDGE_DIRECTORY, "1.2")


async def _compare(
    knowledge: KnowledgeService, text: str = ""
) -> tuple[ClinicalEvidence, list[DifferentialResult]]:
    """用真实候选对触发来源鉴别，避免方证召回排序影响节点测试。"""
    evidence = parse_clinical_text(text) if text else ClinicalEvidence()
    formulas = [
        FormulaCandidate(
            formula_name=name,
            association_scope="classic_clause_based",
            evidence_ids=["1.2:test"],
            rank_score=0.5,
        )
        for name in ("桂枝汤", "麻黄汤")
    ]
    return evidence, await compare_formula_candidates(evidence, formulas, knowledge)


@pytest.mark.asyncio
async def test_priority_recomputes_and_repetition_reduces_it(knowledge: KnowledgeService) -> None:
    """来源复核可靠性与六因子可复算；历史提问只降低排序，不改事实。"""
    evidence, results = await _compare(knowledge)
    gaps = await build_evidence_gaps(results, evidence, knowledge)
    pending = [gap for gap in gaps if gap.status == "pending"]
    assert {gap.dimension for gap in pending} == {"汗", "脉"}
    sweat = next(gap for gap in pending if gap.dimension == "汗")
    factors = sweat.priority_factors
    assert factors["source_reliability"] == 0.8
    assert sweat.priority == pytest.approx(
        factors["discriminative_power"]
        * factors["clinical_answerability"]
        * factors["source_reliability"]
        * factors["candidate_coverage"]
        - factors["question_cost"]
    )
    repeated = await build_evidence_gaps(
        results,
        evidence,
        knowledge,
        context=EvidenceGapContext(asked_counts={sweat.question_id: 2}),
    )
    assert next(
        gap for gap in repeated if gap.question_id == sweat.question_id
    ).priority == pytest.approx(sweat.priority - 0.2)
    assert evidence.findings == []
    assert all(
        gap.status == "deferred"
        for gap in gaps
        if gap.dimension in {"病机", "共同表现", "追问重点"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["昨日汗出", "汗出，无汗", "汗出或无汗"])
async def test_unusable_fact_does_not_close_question(
    knowledge: KnowledgeService, text: str
) -> None:
    """历史、冲突、未决分支不能支持当前汗出条件，也不能消除补证。"""
    evidence, results = await _compare(knowledge, text)
    gaps = await build_evidence_gaps(results, evidence, knowledge)
    assert next(gap for gap in gaps if gap.dimension == "汗").status == "pending"


@pytest.mark.asyncio
async def test_known_fact_unavailable_and_missing_source(knowledge: KnowledgeService) -> None:
    """已明确汗出不重复提问；无法获取不转阴性，伪造来源不进入主动补证。"""
    evidence, results = await _compare(knowledge, "无汗")
    gaps = await build_evidence_gaps(results, evidence, knowledge)
    assert "汗" not in {gap.dimension for gap in gaps}
    pulse = next(gap for gap in gaps if gap.status == "pending")
    unavailable = await build_evidence_gaps(
        results,
        evidence,
        knowledge,
        context=EvidenceGapContext(unavailable_question_ids=[pulse.question_id]),
    )
    assert (
        next(gap for gap in unavailable if gap.question_id == pulse.question_id).status
        == "unavailable"
    )
    decision = decide_evidence_gap_stop(
        unavailable, results, [], supplementation_round=1, max_rounds=3
    )
    assert decision.stop and decision.reason == "all_unavailable"
    bad = results[0].model_copy(deep=True)
    for dimension in bad.dimensions:
        dimension.source_rule_id = "missing-rule"
    rejected = await build_evidence_gaps([bad], evidence, knowledge)
    assert all(
        gap.status == "deferred" and gap.priority_factors["source_reliability"] == 0
        for gap in rejected
    )


@pytest.mark.asyncio
async def test_partial_or_is_not_lost_when_dimension_has_support(
    knowledge: KnowledgeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一方已支持而另一方 OR 仍未知时继续问，排除已反证分支的无用事实键。"""
    rule = await knowledge.get_differential_by_id("diff_v12_001")
    assert rule is not None
    fixture_rule = rule.model_copy(
        update={"key_dimensions": {"汗": "桂枝汤证汗出；麻黄汤证无汗或喘。"}}
    )

    async def get_rule(rule_id: str) -> DifferentialRule:
        """提供测试副本来源，不替换正式知识。"""
        return fixture_rule

    monkeypatch.setattr(knowledge, "get_differential_by_id", get_rule)
    evidence, results = await _compare(knowledge, "汗出")
    dimension = results[0].dimensions[1]
    dimension.candidate_values = {"桂枝汤证": "汗出", "麻黄汤证": "无汗或喘"}
    dimension.status = "supports"
    results[0].dimensions = [dimension]
    gaps = await build_evidence_gaps(results[:1], evidence, knowledge)
    assert gaps[0].status == "pending" and gaps[0].finding_keys == ["dyspnea"]


@pytest.mark.asyncio
async def test_stop_guards_separation_and_reports_round_limit(knowledge: KnowledgeService) -> None:
    """大分差不能跳过未决高反证；单候选及缺来源保持保守，医生选择明确记录。"""
    evidence, results = await _compare(knowledge)
    gaps = await build_evidence_gaps(results, evidence, knowledge)
    response = await ReasoningService(knowledge).analyze_text("发热，无汗，恶寒，身痛，喘，脉浮紧")
    paths = response.reasoning_paths[:2]
    paths[0].ranking_score, paths[1].ranking_score = 0.9, 0.1
    decision = decide_evidence_gap_stop(gaps, results, paths, supplementation_round=0, max_rounds=3)
    assert not decision.stop and not decision.high_weight_counterevidence_checked
    assert (
        decide_evidence_gap_stop(
            gaps, results, paths[:1], supplementation_round=0, max_rounds=3
        ).separation
        is None
    )
    assert (
        decide_evidence_gap_stop(gaps, results, paths, supplementation_round=3, max_rounds=3).reason
        == "max_rounds"
    )
    assert (
        decide_evidence_gap_stop(
            gaps,
            results,
            paths,
            supplementation_round=0,
            max_rounds=3,
            doctor_current_information=True,
        ).reason
        == "doctor_current_information"
    )
    assert EvidenceGapPolicy(minimum_priority=0.2).minimum_priority == 0.2


@pytest.mark.asyncio
async def test_score_stop_requires_completed_counterevidence(knowledge: KnowledgeService) -> None:
    """仅完整条件夹具允许按分差停止；不可解析或单候选来源不冒充检查完成。"""
    evidence, results = await _compare(knowledge, "无汗，脉浮紧，脉不缓")
    results[0].dimensions = [
        item for item in results[0].dimensions if item.dimension in {"汗", "脉"}
    ]
    # 使用可完整核对的真实汗维度，排除脉词项互斥尚未覆盖的来源条件。
    results[0].dimensions = [item for item in results[0].dimensions if item.dimension == "汗"]
    response = await ReasoningService(knowledge).analyze_text("发热，无汗，恶寒，身痛，喘，脉浮紧")
    paths = [path.model_copy(deep=True) for path in response.reasoning_paths[:2]]
    for path in paths:
        path.evidence_assessments = [
            assessment.model_copy(update={"unresolved_requirements": []})
            for assessment in path.evidence_assessments
        ]
    paths[0].ranking_score, paths[1].ranking_score = 0.9, 0.1
    decision = decide_evidence_gap_stop([], results, paths, supplementation_round=0, max_rounds=3)
    assert decision.reason == "score_separation" and decision.high_weight_counterevidence_checked
    paths[0].evidence_assessments[0].unresolved_requirements = ["未核对条件"]
    blocked = decide_evidence_gap_stop([], results, paths, supplementation_round=0, max_rounds=3)
    assert (
        blocked.reason == "no_actionable_gaps" and not blocked.high_weight_counterevidence_checked
    )


@pytest.mark.asyncio
async def test_common_symptom_cannot_claim_discrimination(
    knowledge: KnowledgeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """双方同症状时区分度为零，不能凭候选覆盖率生成高价值问题。"""
    rule = await knowledge.get_differential_by_id("diff_v12_001")
    assert rule is not None
    fixture_rule = rule.model_copy(
        update={"key_dimensions": {"汗": "桂枝汤证汗出；麻黄汤证汗出。"}}
    )

    async def get_rule(rule_id: str) -> DifferentialRule:
        """返回测试副本。"""
        return fixture_rule

    monkeypatch.setattr(knowledge, "get_differential_by_id", get_rule)
    evidence, results = await _compare(knowledge)
    dimension = results[0].dimensions[1]
    dimension.candidate_values = {"桂枝汤证": "汗出", "麻黄汤证": "汗出"}
    results[0].dimensions = [dimension]
    gap = (await build_evidence_gaps(results[:1], evidence, knowledge))[0]
    assert gap.status == "deferred" and gap.priority_factors["discriminative_power"] == 0


@pytest.mark.asyncio
async def test_question_identity_survives_partial_answer_and_duplicates(
    knowledge: KnowledgeService,
) -> None:
    """部分回答不重置问题身份和重复计数，重复来源不提高优先级。"""
    evidence, results = await _compare(knowledge)
    gaps = await build_evidence_gaps(results, evidence, knowledge)
    duplicate = await build_evidence_gaps(results + results, evidence, knowledge)
    assert [gap.model_dump() for gap in gaps] == [gap.model_dump() for gap in duplicate]
    old = next(gap for gap in gaps if gap.dimension == "脉")
    partial, results = await _compare(knowledge, "脉浮")
    revised = await build_evidence_gaps(
        results, partial, knowledge, context=EvidenceGapContext(asked_counts={old.question_id: 1})
    )
    pulse = next(gap for gap in revised if gap.dimension == "脉")
    assert pulse.question_id == old.question_id
    assert "pulse_floating" not in pulse.finding_keys
    assert pulse.priority_factors["repetition_penalty"] == 0.1
