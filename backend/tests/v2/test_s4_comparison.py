"""S4 测试：排序来自 LLM，程序不重排且不允许候选池外方剂。"""

from unittest.mock import AsyncMock

import pytest

from backend.app.clinical_v2.comparison import compare_formula_candidates
from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ComparisonV2,
    FormulaEvidenceCandidateV2,
    RetrievalBundleV2,
    RetrievalQueryPlanV2,
)
from backend.app.clinical_v2.verification import verify_comparison_evidence


class ComparisonLLM:
    """返回指定顺序的结构化比较结果。"""

    def __init__(self, result: ComparisonV2) -> None:
        self.result = result

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """S4 必须调用结构化接口。"""
        raise AssertionError("S4 不调用自由文本端口")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
        """返回模型相对顺序。"""
        assert schema is ComparisonV2
        return self.result


def setup() -> tuple[ClinicalEvidenceV2, ChannelLocalizationV2, RetrievalBundleV2]:
    """构造同一候选池和真实来源 ID。"""
    evidence = ClinicalEvidenceV2(
        raw_text="发热",
        findings=[
            {
                "fact_id": "fact-a",
                "concept": "发热",
                "polarity": "positive",
                "time_scope": "current",
                "source_span": {"start": 0, "end": 2, "text": "发热"},
                "review_status": "verified_source",
            }
        ],
    )
    localization = ChannelLocalizationV2(
        status="resolved",
        candidates=[
            {
                "candidate_id": "ch",
                "six_channels": ["太阳"],
                "support_fact_ids": ["fact-a"],
                "retrieval_focus": ["表证"],
                "short_basis": "有据",
            }
        ],
        proceed_to_retrieval=True,
        clarification_questions=[],
        note="",
    )
    refs = []
    candidates = []
    for ident, name in [("ca", "桂枝汤"), ("cb", "麻黄汤")]:
        refs.append(f"1.2:{ident}")
        candidates.append(
            FormulaEvidenceCandidateV2(
                candidate_id=ident,
                formula_name=name,
                source_evidence=[
                    {
                        "evidence_id": f"1.2:{ident}",
                        "knowledge_type": "formula_pattern",
                        "knowledge_version": "1.2",
                        "excerpt": name,
                        "review_status": "ok",
                    }
                ],
                provenance="classic",
            )
        )
    plan = RetrievalQueryPlanV2(
        channel_hypotheses=["太阳"],
        positive_facts=["发热"],
        explicit_negative_facts=[],
        primary_query="发热 太阳",
        filter_mode="soft_channel",
    )
    return (
        evidence,
        localization,
        RetrievalBundleV2(
            query_plan=plan,
            formula_candidates=candidates,
            retrieved_evidence_ids=refs,
            retrieval_status="ok",
        ),
    )


@pytest.mark.asyncio
async def test_s4_preserves_llm_order_without_score_fields() -> None:
    """模型给出的麻黄汤优先顺序原样保留，结果没有数值评分字段。"""
    evidence, localization, retrieval = setup()
    result = ComparisonV2(
        status="ranked",
        ordered_paths=[
            {
                "path_id": "p-b",
                "candidate_id": "cb",
                "six_channels": ["太阳"],
                "formula_name": "麻黄汤",
                "rank": 1,
                "support_fact_ids": ["fact-a"],
                "counter_fact_ids": [],
                "missing_dimensions": [],
                "support_evidence_ids": ["1.2:cb"],
                "counter_evidence_ids": [],
                "brief_rationale": "模型顺序",
                "differential_summary": "对照",
                "verification_status": "pending",
            }
        ],
        next_questions=[],
    )
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    checked = await compare_formula_candidates(
        evidence, localization, retrieval, knowledge, ComparisonLLM(result)
    )
    assert checked.ordered_paths[0].candidate_id == "cb"
    assert "score" not in checked.ordered_paths[0].model_dump()
    assert checked.ordered_paths[0].verification_status == "grounded"
    verified = verify_comparison_evidence(checked, evidence, retrieval)
    assert verified.ordered_paths[0].candidate_id == "cb"


@pytest.mark.asyncio
async def test_s4_rejects_candidate_outside_retrieval() -> None:
    """模型不能从记忆中增加未召回的方剂。"""
    evidence, localization, retrieval = setup()
    result = ComparisonV2(
        status="ranked",
        ordered_paths=[
            {
                "path_id": "p-x",
                "candidate_id": "missing",
                "six_channels": ["太阳"],
                "formula_name": "小柴胡汤",
                "rank": 1,
                "support_fact_ids": ["fact-a"],
                "counter_fact_ids": [],
                "missing_dimensions": [],
                "support_evidence_ids": [],
                "counter_evidence_ids": [],
                "brief_rationale": "越界",
                "differential_summary": "越界",
                "verification_status": "pending",
            }
        ],
        next_questions=[],
    )
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    with pytest.raises(Exception) as caught:
        await compare_formula_candidates(
            evidence, localization, retrieval, knowledge, ComparisonLLM(result)
        )
    assert getattr(caught.value, "code", None) == "COMPARISON_UNGROUNDED"
