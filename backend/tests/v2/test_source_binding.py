"""S4 来源绑定测试：事实和 Evidence ID 必须来自本轮上下文。"""

from unittest.mock import AsyncMock

import pytest

from backend.app.clinical_v2.comparison import ComparisonUngroundedError, compare_formula_candidates
from backend.app.clinical_v2.schemas import ComparisonDraftV2, ComparisonV2
from backend.tests.v2.test_s4_comparison import ComparisonLLM, setup


def test_comparison_schema_has_no_numeric_medical_score() -> None:
    """DTO 字段不包含 confidence、score_components 或 ranking_score。"""
    names = set(ComparisonV2.model_fields)
    assert not {"confidence", "score", "ranking_score", "score_components"} & names


def comparison() -> ComparisonV2:
    """构造引用当前方证真实来源的合法模型返回。"""
    return ComparisonV2(
        status="ranked",
        ordered_paths=[
            {
                "path_id": "p1",
                "candidate_id": "ca",
                "formula_name": "桂枝汤",
                "six_channels": ["太阳"],
                "rank": 1,
                "support_fact_ids": ["fact-a"],
                "counter_fact_ids": [],
                "missing_dimensions": [],
                "support_evidence_ids": ["1.2:ca"],
                "counter_evidence_ids": [],
                "brief_rationale": "来源支持",
                "differential_summary": "比较",
                "verification_status": "pending",
            }
        ],
    )


def draft_fields_are_model_owned() -> set[str]:
    """草稿协议不允许模型生成工程身份、方名或来源签名。"""
    return set(ComparisonDraftV2.model_fields)


@pytest.mark.parametrize(
    "failure", ["unknown_fact", "unseen_source", "wrong_rank", "other_candidate_source"]
)
async def test_source_binding_rejects_invalid_path(failure: str) -> None:
    """未知事实、模型未见的来源、错序和其他候选来源不能签发 grounded。"""
    evidence, localization, retrieval = setup()
    result = comparison()
    if failure == "unknown_fact":
        evidence.findings[0].polarity = "unknown"
    elif failure == "unseen_source":
        retrieval.retrieved_evidence_ids.append("1.2:unseen")
        result.ordered_paths[0].support_evidence_ids = ["1.2:unseen"]
    elif failure == "wrong_rank":
        result.ordered_paths[0].rank = 2
    else:
        result.ordered_paths[0].support_evidence_ids = ["1.2:cb"]
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    with pytest.raises(ComparisonUngroundedError):
        await compare_formula_candidates(
            evidence, localization, retrieval, knowledge, ComparisonLLM(result)
        )


async def test_source_payload_is_signed_by_program() -> None:
    """公开来源摘录由程序从本轮知识包签发，模型不能修改原文。"""
    evidence, localization, retrieval = setup()
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    checked = await compare_formula_candidates(
        evidence, localization, retrieval, knowledge, ComparisonLLM(comparison())
    )
    assert {ref.evidence_id for ref in checked.source_evidence} == {"1.2:ca", "1.2:cb"}
    assert (
        checked.source_evidence[0].excerpt
        == retrieval.formula_candidates[0].source_evidence[0].excerpt
    )


def test_draft_excludes_program_signed_fields() -> None:
    """草稿 schema 不暴露程序签发字段。"""
    assert not {
        "path_id",
        "formula_name",
        "verification_status",
        "source_evidence",
    } & draft_fields_are_model_owned()
