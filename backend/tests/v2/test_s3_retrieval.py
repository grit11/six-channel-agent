"""S3 候选归并测试：同方多条来源只占一个方证位置。"""

from unittest.mock import AsyncMock

import pytest

from backend.app.clinical_v2.retrieval import retrieve_formula_candidates
from backend.app.clinical_v2.schemas import RetrievalQueryPlanV2
from backend.app.schemas.evidence import Evidence


def evidence(name: str, index: int) -> Evidence:
    """构造具有真实格式的来源替身。"""
    return Evidence(
        evidence_id=f"1.2:f{index}",
        knowledge_id=f"f{index}",
        knowledge_type="formula_pattern",
        knowledge_version="1.2",
        source="经典",
        source_level="L3",
        content=f"{name}原文{index}",
        retrieval_score=1.0 / (index + 1),
        metadata={"formula_name": name, "review_status": "ok", "canonical_six_channels": ["太阳"]},
        retrieval_method="hybrid",
    )


@pytest.mark.asyncio
async def test_retrieval_groups_formula_sources_and_caps_at_five() -> None:
    """不会拿同一方的多条条文凑 Top-5，也不制造不存在的方剂。"""
    knowledge = AsyncMock()
    knowledge.search_formula.return_value = [evidence("桂枝汤", 1), evidence("桂枝汤", 2)] + [
        evidence(name, index + 3)
        for index, name in enumerate(["麻黄汤", "小柴胡汤", "理中丸", "四逆汤", "乌梅丸", "葛根汤"])
    ]
    plan = RetrievalQueryPlanV2(
        channel_hypotheses=["太阳"],
        positive_facts=["发热"],
        explicit_negative_facts=[],
        primary_query="发热 太阳",
        differential_query=None,
        rescue_query="发热 方证",
        filter_mode="soft_channel",
    )
    bundle = await retrieve_formula_candidates(knowledge, plan)
    assert len(bundle.formula_candidates) == 5
    assert [item.formula_name for item in bundle.formula_candidates].count("桂枝汤") == 1
    gui = next(item for item in bundle.formula_candidates if item.formula_name == "桂枝汤")
    assert len(gui.source_evidence) == 2
