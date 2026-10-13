"""S3 查询计划测试：只组装已核验事实，保持正负事实分离。"""

from backend.app.clinical_v2.retrieval import build_query_plan
from backend.app.clinical_v2.schemas import ChannelLocalizationV2, ClinicalEvidenceV2


def test_query_plan_preserves_negative_and_does_not_invent_symptoms() -> None:
    """显式阴性进入独立字段，病例未出现的症状不会出现在查询。"""
    evidence = ClinicalEvidenceV2(
        raw_text="发热，无汗",
        findings=[
            {
                "fact_id": "a",
                "concept": "发热",
                "polarity": "positive",
                "time_scope": "current",
                "source_span": {"start": 0, "end": 2, "text": "发热"},
                "review_status": "verified_source",
            },
            {
                "fact_id": "b",
                "concept": "汗出",
                "polarity": "negative",
                "time_scope": "current",
                "source_span": {"start": 3, "end": 5, "text": "无汗"},
                "review_status": "verified_source",
            },
        ],
    )
    localization = ChannelLocalizationV2(
        status="resolved",
        candidates=[
            {
                "candidate_id": "c",
                "six_channels": ["太阳"],
                "support_fact_ids": ["a", "b"],
                "retrieval_focus": ["表证"],
                "short_basis": "有据",
            }
        ],
        proceed_to_retrieval=True,
        clarification_questions=[],
        note="",
    )
    plan = build_query_plan(evidence, localization)
    assert plan.explicit_negative_facts == ["汗出"]
    assert "发热" in plan.primary_query and "表证" in plan.primary_query
    assert "口渴" not in plan.primary_query
    assert plan.filter_mode == "soft_channel"
