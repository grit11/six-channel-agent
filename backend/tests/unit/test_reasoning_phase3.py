"""Phase 3 八纲和六经候选单测，使用真实知识快照但不调用外部模型。"""

from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.channel_candidate import generate_channel_candidates
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.formula_candidate import generate_formula_candidates


@pytest.fixture
async def knowledge(tmp_path: Path) -> KnowledgeService:
    """以 BM25 知识服务替身验证候选编排，不要求远程模型或 Milvus。"""
    return await KnowledgeService.from_directory(Settings().knowledge_directory, "1.2")


def test_eight_principles_keeps_unknown_out_of_scores() -> None:
    """未提供舌脉只能进入 unknown 列表，不能制造寒热或虚实分数。"""
    evidence = parse_clinical_text("发热，无汗，恶寒，脉浮")
    state = analyze_eight_principles(evidence)
    assert state.cold_heat.score_left > 0
    assert state.cold_heat.unknown_key_findings
    assert state.deficiency_excess.score_left == 0


@pytest.mark.asyncio
async def test_channel_candidates_keep_multiple_independent_results(
    knowledge: KnowledgeService,
) -> None:
    """候选按独立排序分返回，证据 ID 必须来自五类知识真实来源。"""
    evidence = parse_clinical_text("发热，无汗，恶寒，脉浮")
    principles = analyze_eight_principles(evidence)
    candidates = await generate_channel_candidates(evidence, principles, knowledge, top_k=3)
    assert len(candidates) == 3
    assert all(item.status in {"active", "weak"} for item in candidates)
    assert all(item.classic_evidence_ids for item in candidates)
    assert candidates[0].independent_score >= candidates[-1].independent_score
    assert all("%" not in item.provisional_syndrome for item in candidates)


@pytest.mark.asyncio
async def test_formula_candidates_preserve_scope_and_risk(knowledge: KnowledgeService) -> None:
    """方证候选只读真实来源，canonical 与历史派生范围不被合并改写。"""
    evidence = parse_clinical_text("发热，无汗，恶寒，脉浮")
    principles = analyze_eight_principles(evidence)
    channels = await generate_channel_candidates(evidence, principles, knowledge, top_k=3)
    formulas = await generate_formula_candidates(evidence, channels, knowledge, top_k=5)
    assert formulas
    assert all(item.evidence_ids for item in formulas)
    assert all(
        item.association_scope in {"classic_clause_based", "case_derived_historical"}
        for item in formulas
    )
    assert all(not hasattr(item, "dosage") for item in formulas)
