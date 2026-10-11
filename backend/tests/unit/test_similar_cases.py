"""真实医案核验回归：初诊极性、历史/未知隔离和经典排序不变，不调用在线模型。"""

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.similar_cases import assess_similar_case
from backend.app.services.reasoning_service import ReasoningService


@pytest.mark.asyncio
async def test_real_case_audits_polarity_and_keeps_canonical_paths() -> None:
    """case_0001 初诊无汗不能被标签汗出误导，引用只出现在历史类别。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    service = ReasoningService(knowledge)
    result = await service.analyze_text("发热，无汗，恶寒，脉浮紧")
    source = (await knowledge.get_evidence_by_ids(["1.2:case_0001"]))[0]
    audit = assess_similar_case(source, result.clinical_evidence, result.reasoning_paths)
    assert audit.status == "reference"
    assert "无汗" in audit.matching_conditions
    conflicting = assess_similar_case(source, parse_clinical_text("汗出"), result.reasoning_paths)
    assert conflicting.status == "conflicting"
    assert "无汗" in conflicting.conflicting_conditions
    unknown = assess_similar_case(
        source, parse_clinical_text("昨日发热，昨日无汗，昨日脉浮紧"), result.reasoning_paths
    )
    assert unknown.status == "insufficient"
    assert unknown.similarity_score == 0
    assert result.similar_cases
    assert all(path.score_components["case_factor"] == 0 for path in result.reasoning_paths)
    assert all(
        not ref.startswith("1.2:case_")
        for path in result.reasoning_paths
        for ref in path.classic_evidence
    )
    assert [stage.stage for stage in result.stage_snapshots] == result.completed_stages
    assert result.stage_snapshots[-1].stage == "similar_cases"
    assert result.stage_snapshots[-2].payload["reasoning_paths"][0]["case_evidence"] == []
