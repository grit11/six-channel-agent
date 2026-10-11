"""病程核验回归：时间绑定、三值、重复身份和治疗因果边界；不使用模型或伪造知识。"""

import pytest

from backend.app.nodes.syndrome_relation import judge_syndrome_relationship
from backend.app.nodes.temporal_evidence import assess_temporal_evidence
from backend.app.schemas.clinical import ClinicalEvidence, Finding, TreatmentEvent
from backend.tests.unit.test_syndrome_relation import candidate, evidence_with_timeline


def treatment_evidence() -> ClinicalEvidence:
    """构造同一事实在治疗前后明确不同极性的医生记录。"""
    return ClinicalEvidence(
        findings=[
            Finding(key="fever", polarity="positive", source="doctor_input", time_scope="昨日"),
            Finding(key="fever", polarity="negative", source="doctor_input", time_scope="今日"),
        ],
        treatment_events=[
            TreatmentEvent(
                event_id="t1",
                description="医生记录既往治疗",
                event_order=0,
                before_finding_keys=["fever"],
                before_time_scope="昨日",
                after_finding_keys=["fever"],
                after_time_scope="今日",
                source_text="核对昨日发热与今日不发热记录",
            )
        ],
    )


def test_treatment_binds_frames_without_causal_inference() -> None:
    """阴性可以证明治疗后观察，不会变成当前阳性或治疗因果。"""
    evidence = treatment_evidence()
    audit = assess_temporal_evidence(evidence)[0]
    assert audit.status == "usable"
    assert audit.supporting_findings == ["fever"]
    assert "不认定治疗因果" in audit.reason
    assert evidence.current_findings()[0].polarity == "negative"
    assert (
        ClinicalEvidence.model_validate({"prior_treatment": [{"old": True}]}).treatment_events == []
    )


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("before_time_scope", None, "incomplete"),
        ("before_time_scope", "今日", "incomplete"),
        ("event_order", None, "incomplete"),
        ("after_finding_keys", ["missing"], "incomplete"),
        ("source", "inference", "unsupported"),
    ],
)
def test_treatment_missing_binding_is_not_fabricated(
    field: str, value: object, status: str
) -> None:
    """缺少时间、次序或事实时返回可审计的未完成状态。"""
    evidence = treatment_evidence()
    raw = evidence.treatment_events[0].model_dump()
    raw[field] = value
    evidence.treatment_events[0] = TreatmentEvent.model_validate(raw)
    assert assess_temporal_evidence(evidence)[0].status == status


def test_duplicate_treatment_and_conflicting_facts_remain_visible() -> None:
    """重复身份及同时间正反事实不能通过择一得到 usable。"""
    evidence = treatment_evidence()
    evidence.treatment_events.append(evidence.treatment_events[0].model_copy())
    assert all(item.status == "conflicting" for item in assess_temporal_evidence(evidence))
    evidence.treatment_events.pop()
    evidence.findings.append(Finding(key="fever", polarity="positive", source="record"))
    assert assess_temporal_evidence(evidence)[0].status == "conflicting"


@pytest.mark.parametrize("change", ["historical", "zero", "conflict", "negative", "inference"])
def test_timeline_cannot_promote_untrusted_current_support(change: str) -> None:
    """历史、零确定性、冲突、阴性和推测均不能确认当前两经共存。"""
    evidence = evidence_with_timeline(0)
    fact = evidence.findings[1]
    if change == "historical":
        fact.time_scope = "昨日"
        evidence.timeline[1].time_scope = "昨日"
    elif change == "zero":
        fact.certainty = 0
    elif change == "conflict":
        evidence.findings.append(Finding(key="thirst", polarity="negative", source="record"))
    elif change == "negative":
        fact.polarity = "negative"
    else:
        fact.source = "inference"
    relation = judge_syndrome_relationship(
        evidence, [candidate("太阳"), candidate("阳明")], ["1.2:core_0010"]
    )[0]
    assert relation.relationship_type == "uncertain"
    if change != "historical":
        assert "thirst" not in relation.temporal_assessments[1].supporting_findings


def test_same_channel_candidates_need_explicit_attribution() -> None:
    """不能用候选遍历次序暗选同经归属。"""
    duplicate = candidate("太阳")
    duplicate.candidate_id = "second_solar"
    relation = judge_syndrome_relationship(
        evidence_with_timeline(0),
        [candidate("太阳"), duplicate, candidate("阳明")],
        ["1.2:core_0010"],
    )[0]
    assert relation.relationship_type == "uncertain"
    assert "同一经的多个候选需明确关系归属" in relation.missing_timeline_evidence


def test_four_channel_combination_is_outside_verified_definition() -> None:
    """来源只定义两/三经同时发病，不把任意多经组合升级为合病。"""
    evidence = evidence_with_timeline(0)
    for channel in ("少阳", "太阴"):
        evidence.timeline.append(evidence.timeline[0].model_copy(update={"six_channel": channel}))
    result = judge_syndrome_relationship(
        evidence,
        [candidate(channel) for channel in ("太阳", "阳明", "少阳", "太阴")],
        ["1.2:core_0010"],
    )[0]
    assert result.relationship_type == "uncertain"
