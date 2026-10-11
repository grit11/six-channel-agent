"""S2 关系全类别回归：使用真实框架来源、医生核对与时间事实，缺失不等于否定。"""

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.relationship_rules import evaluate_relationships
from backend.app.schemas.clinical import (
    ClinicalEvidence,
    Finding,
    RelationshipHistory,
    TimelineEvent,
    TreatmentEvent,
)
from backend.app.schemas.evidence import Evidence
from backend.tests.unit.test_syndrome_relation import candidate, evidence_with_timeline


@pytest.fixture
async def sources() -> list[Evidence]:
    """精确读取原知识，不编造临床定义。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    return await knowledge.get_evidence_by_ids(
        [f"1.2:core_{index:04}" for index in (1, 2, 10, 11)] + ["1.2:clause_16", "1.2:clause_267"]
    )


def one_current(channel: str = "少阴") -> ClinicalEvidence:
    """构造医生确认的单组当前事实，不由候选数量推断单证。"""
    return ClinicalEvidence(
        findings=[Finding(key="limb_cold", polarity="positive", source="doctor_input")],
        timeline=[
            TimelineEvent(
                six_channel=channel,
                finding_keys=["limb_cold"],
                onset_order=1,
                active_now=True,
                source_text="医生确认当前肢冷症状组",
            )
        ],
        relationship_history=RelationshipHistory(
            source_text="医生已核对病史", primary_six_channel=channel, history_complete=True
        ),
    )


def types(evidence: ClinicalEvidence, sources: list[Evidence]) -> set[str]:
    """只取真实当前组的候选，历史经不能产生伪造当前身份。"""
    channels = {event.six_channel for event in evidence.timeline if event.time_scope is None}
    return {
        item.relationship_type
        for item in evaluate_relationships(
            evidence, [candidate(channel) for channel in channels], sources
        )
    }


@pytest.mark.asyncio
async def test_single_requires_doctor_exclusion_not_top_one(sources: list[Evidence]) -> None:
    """只有确认排查其他组才能产生 single；未问及保持未定。"""
    evidence = one_current()
    assert types(evidence, sources) == {"uncertain"}
    evidence.relationship_history.no_other_current_syndrome = True
    assert "single" in types(evidence, sources)


@pytest.mark.asyncio
async def test_direct_invasion_requires_explicit_negative_yang_history(
    sources: list[Evidence],
) -> None:
    """未知既往三阳不当作未经过，明确有三阳则不得确认直中。"""
    evidence = one_current()
    assert "direct_invasion" not in types(evidence, sources)
    evidence.relationship_history.prior_yang_phase = False
    assert "direct_invasion" in types(evidence, sources)
    evidence.relationship_history.prior_yang_phase = True
    assert "direct_invasion" not in types(evidence, sources)


@pytest.mark.asyncio
async def test_twofold_and_concurrent_require_distinct_history_conditions(
    sources: list[Evidence],
) -> None:
    """同时发病不单独证明两感；主经、表里配对及同时感邪由医生核对。"""
    evidence = evidence_with_timeline(0)
    evidence.timeline[1].six_channel = "少阴"
    evidence.relationship_history = RelationshipHistory(
        primary_six_channel="太阳", source_text="医生核对本兼关系", simultaneous_exposure=True
    )
    assert types(evidence, sources) == {"combined_disease", "concurrent"}
    evidence.relationship_history.exterior_interior_pair = True
    assert "twofold_attack" in types(evidence, sources)
    evidence.timeline[1].onset_order = 1
    assert "twofold_attack" not in types(evidence, sources)
    assert "sequential_concurrent_disease" in types(evidence, sources)


@pytest.mark.asyncio
async def test_transmission_and_changes_need_historical_facts_and_not_days(
    sources: list[Evidence],
) -> None:
    """历史经无当前候选仍可审计；按日数或没有脉证变化不能传经。"""
    evidence = one_current()
    evidence.course = "病程七日"
    evidence.relationship_history.syndrome_changed = True
    assert "transmission" not in types(evidence, sources)
    evidence.findings.append(
        Finding(key="chills", polarity="positive", source="doctor_input", time_scope="昨日")
    )
    evidence.timeline.append(
        TimelineEvent(
            six_channel="太阳",
            finding_keys=["chills"],
            time_scope="昨日",
            onset_order=0,
            active_now=False,
            source_text="昨日太阳组已罢",
        )
    )
    result = evaluate_relationships(evidence, [candidate("少阴")], sources)
    assert {item.relationship_type for item in result} == {"transmission", "changed_syndrome"}
    assert all(
        item.primary_candidate_id == "channel_少阴" and not item.secondary_candidate_ids
        for item in result
    )
    assert result[0].previous_six_channels == ["太阳"]
    evidence.timeline[1].active_now = True
    assert "transmission" not in types(evidence, sources)


@pytest.mark.asyncio
async def test_spoiled_needs_doctor_causality_and_complete_treatment_bindings(
    sources: list[Evidence],
) -> None:
    """治疗相邻不推出坏病，三项病史确认和治疗前后引用齐全才给候选。"""
    evidence = one_current()
    evidence.findings.append(
        Finding(key="chills", polarity="positive", source="doctor_input", time_scope="昨日")
    )
    evidence.timeline.append(
        TimelineEvent(
            six_channel="太阳",
            finding_keys=["chills"],
            time_scope="昨日",
            onset_order=0,
            active_now=False,
            source_text="昨日症状组",
        )
    )
    evidence.treatment_events = [
        TreatmentEvent(
            event_id="t1",
            description="医生记录已用治疗",
            event_order=0,
            before_finding_keys=["chills"],
            before_time_scope="昨日",
            after_finding_keys=["limb_cold"],
            source_text="医生确认治疗前后观察",
        )
    ]
    evidence.relationship_history.syndrome_changed = True
    evidence.relationship_history.original_pattern_applicable = False
    evidence.relationship_history.treatment_event_ids = ["t1"]
    assert "spoiled_disease" not in types(evidence, sources)
    evidence.relationship_history.treatment_error_confirmed = True
    assert "spoiled_disease" in types(evidence, sources)
    evidence.treatment_events[0].after_finding_keys = ["chills"]
    assert "spoiled_disease" not in types(evidence, sources)
    evidence.treatment_events[0].after_finding_keys = ["limb_cold"]
    evidence.treatment_events[0].before_time_scope = None
    assert "spoiled_disease" not in types(evidence, sources)


@pytest.mark.asyncio
async def test_sources_missing_inferred_history_and_unknown_group_block_confirmation(
    sources: list[Evidence],
) -> None:
    """伪造类别、推测病史或未知当前组不能暗自选择一条有效记录确认关系。"""
    evidence = one_current()
    evidence.relationship_history.no_other_current_syndrome = True
    assert types(evidence, []) == {"uncertain"}
    evidence.relationship_history.source = "inference"
    assert types(evidence, sources) == {"uncertain"}
    evidence.relationship_history.source = "doctor_input"
    evidence.timeline.append(
        evidence.timeline[0].model_copy(update={"six_channel": "太阳", "active_now": None})
    )
    assert types(evidence, sources) == {"uncertain"}


@pytest.mark.asyncio
async def test_incomplete_order_is_gap_and_unsupported_current_channel_is_not_ignored(
    sources: list[Evidence],
) -> None:
    """未问及次序不产生临床反证；候选外当前组也不能静默过滤为单证。"""
    evidence = evidence_with_timeline(1)
    evidence.timeline[1].onset_order = None
    relation = evaluate_relationships(evidence, [candidate("太阳"), candidate("阳明")], sources)[0]
    combined = next(
        item for item in relation.assessments if item.relationship_type == "combined_disease"
    )
    assert combined.status == "insufficient" and not combined.contradicting_conditions
    evidence = one_current()
    evidence.relationship_history.no_other_current_syndrome = True
    evidence.timeline.append(evidence.timeline[0].model_copy(update={"six_channel": "太阳"}))
    result = evaluate_relationships(evidence, [candidate("少阴")], sources)
    assert all(item.relationship_type == "uncertain" for item in result)
