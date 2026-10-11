"""病证关系回归：时间线必须绑定明确事实，候选数量、单一病程和推测不能替代关系证据。"""

import pytest

from backend.app.nodes.syndrome_relation import judge_syndrome_relationship
from backend.app.schemas.clinical import ClinicalEvidence, Finding, TimelineEvent
from backend.app.schemas.reasoning import ChannelCandidate


def candidate(channel: str) -> ChannelCandidate:
    """构造无医学知识内容的候选状态，用于验证关系门槛。"""
    return ChannelCandidate(
        candidate_id=f"channel_{channel}",
        six_channels=[channel],
        independent_score=0.5,
        evidence_completeness=0.5,
        status="active",
    )


def evidence_with_timeline(second_order: int) -> ClinicalEvidence:
    """构造医生确认两个症状组当前共存的结构化时间线。"""
    return ClinicalEvidence(
        findings=[
            Finding(key="chills", polarity="positive", value=True, source="doctor_input"),
            Finding(key="thirst", polarity="positive", value=True, source="doctor_input"),
        ],
        timeline=[
            TimelineEvent(
                six_channel="太阳",
                finding_keys=["chills"],
                onset_order=0,
                active_now=True,
                source_text="医生确认太阳相关恶寒先起且未罢",
            ),
            TimelineEvent(
                six_channel="阳明",
                finding_keys=["thirst"],
                onset_order=second_order,
                active_now=True,
                source_text="医生确认阳明相关口渴起病次序",
            ),
        ],
    )


@pytest.mark.parametrize("top_k", [1, 2])
def test_candidates_and_course_do_not_prove_relationship(top_k: int) -> None:
    """截断到单候选或填写普通病程均不能证明单经本证或并病。"""
    evidence = ClinicalEvidence(
        onset="三日前", course="病程三日", prior_treatment=[{"change": "汗后不适"}]
    )
    result = judge_syndrome_relationship(
        evidence, [candidate("太阳"), candidate("阳明")][:top_k], ["1.2:core_0010"]
    )[0]
    assert result.relationship_type == "uncertain"
    assert result.missing_timeline_evidence


@pytest.mark.parametrize(
    "order,expected", [(0, "combined_disease"), (1, "sequential_concurrent_disease")]
)
def test_confirmed_timeline_classifies_coexistence(order: int, expected: str) -> None:
    """只有明确关联、当前共存和起病次序同时齐全时，才区分合病或并病候选。"""
    result = judge_syndrome_relationship(
        evidence_with_timeline(order), [candidate("太阳"), candidate("阳明")], ["1.2:core_0010"]
    )[0]
    assert result.relationship_type == expected
    assert len(result.temporal_evidence) == 2
    assert result.supporting_evidence == ["1.2:core_0010"]


@pytest.mark.parametrize(
    "field,value",
    [("active_now", None), ("active_now", False), ("source", "inference"), ("onset_order", None)],
)
def test_incomplete_or_inferred_timeline_stays_uncertain(field: str, value: object) -> None:
    """未知、前证已罢、推测来源或不明次序均不能被标为并病。"""
    evidence = evidence_with_timeline(1)
    raw = evidence.timeline[1].model_dump()
    raw[field] = value
    evidence.timeline[1] = TimelineEvent.model_validate(raw)
    result = judge_syndrome_relationship(
        evidence, [candidate("太阳"), candidate("阳明")], ["1.2:core_0010"]
    )[0]
    assert result.relationship_type == "uncertain"


def test_unknown_fact_cannot_ground_timeline() -> None:
    """时间线关联 unknown 事实时不产生确认的病证关系。"""
    evidence = evidence_with_timeline(1)
    evidence.findings[1].polarity = "unknown"
    result = judge_syndrome_relationship(
        evidence, [candidate("太阳"), candidate("阳明")], ["1.2:core_0010"]
    )[0]
    assert result.relationship_type == "uncertain"


def test_negative_fact_cannot_confirm_current_syndrome() -> None:
    """明确否认的症状也不能用来确认该症状组仍存在。"""
    evidence = evidence_with_timeline(1)
    evidence.findings[1].polarity = "negative"
    evidence.findings[1].value = False
    result = judge_syndrome_relationship(
        evidence, [candidate("太阳"), candidate("阳明")], ["1.2:core_0010"]
    )[0]
    assert result.relationship_type == "uncertain"


def test_duplicate_timeline_requires_doctor_resolution() -> None:
    """重复时间线不静默择一，仍保留冲突提示给医生核对。"""
    evidence = evidence_with_timeline(1)
    evidence.timeline.append(evidence.timeline[1].model_copy())
    result = judge_syndrome_relationship(
        evidence, [candidate("太阳"), candidate("阳明")], ["1.2:core_0010"]
    )[0]
    assert result.relationship_type == "uncertain"
    assert result.contradicting_evidence
