"""六阶段知识分支和缓存回归：使用真实来源与并发作用域，不访问在线服务。"""

import asyncio

import pytest

from backend.app.core.config import Settings
from backend.app.knowledge.branch_index import validate_branches
from backend.app.knowledge.run_cache import cached, remember, run_cache_scope
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import (
    CompactClinicalExtraction,
    compact_to_clinical,
    parse_clinical_text,
)
from backend.app.rag.grader import grade_candidate_evidence


@pytest.mark.asyncio
async def test_branch_index_rejects_forged_parent_quote_and_version() -> None:
    """合法派生索引仍逐项关联五类主库；伪造条件或换版本必须拒绝。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    records = list(knowledge._records.values())
    branches = knowledge._branches
    assert len(branches) >= 100
    assert all(item.review_status == "source_verified_pending_expert" for item in branches)
    for update in (
        {"formula_evidence_id": "1.2:invented"},
        {"knowledge_version": "9.9"},
        {"positive_conditions": ["不存在的来源条件"]},
        {"source_quote": "伪造条文"},
    ):
        with pytest.raises(ValueError):
            validate_branches([branches[0].model_copy(update=update)], records, "1.2")


@pytest.mark.asyncio
async def test_formula_branches_do_not_merge_contradictions_from_other_clause() -> None:
    """取最佳可解释分支，不把另条来源阴性当本分支必备或重复支持。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    source = await knowledge.get_formula_by_name("桂枝汤")
    assert source is not None
    branches = [
        item
        for item in knowledge.formula_branches(source.evidence_id)
        if item.logical_scope == "single_clause"
    ]
    assert len(branches) > 1
    facts = parse_clinical_text("发热，汗出，恶风，脉浮而缓")
    result = grade_candidate_evidence(
        source, facts, channels=["太阳"], formula_name="桂枝汤", branches=branches
    )
    repeated = grade_candidate_evidence(
        source, facts, channels=["太阳"], formula_name="桂枝汤", branches=branches * 2
    )
    assert result == repeated
    assert result.grade != "direct"
    assert result.supporting_requirements
    assert all(
        term not in result.contradicting_requirements for term in result.supporting_requirements
    )


@pytest.mark.asyncio
async def test_run_cache_isolation_and_defensive_copy() -> None:
    """并发病例相同键不能共享值；结束后清理，返回值不能污染缓存。"""

    async def case(value: str) -> str:
        """模拟独立患者的异步处理。"""
        with run_cache_scope() as cache:
            remember(("source", "snapshot", "id"), [value])
            await asyncio.sleep(0)
            item = cached(("source", "snapshot", "id"), list)
            assert item == [value]
            item.append("mutation")
            assert cached(("source", "snapshot", "id"), list) == [value]
            assert cache.hits == 2
        assert cache.values == {}
        return value

    assert await asyncio.gather(case("first"), case("second")) == ["first", "second"]
    assert cached(("source", "snapshot", "id"), list) is None


@pytest.mark.asyncio
async def test_assessment_cache_separates_facts_and_formula_context() -> None:
    """同轮复用只适用于完全相同事实；改汗证必须重新核验并保留相反结果。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    with run_cache_scope() as cache:
        positive = await knowledge.assess_candidate_evidence(
            ["1.2:clause_13"], parse_clinical_text("汗出"), channels=["太阳"]
        )
        same = await knowledge.assess_candidate_evidence(
            ["1.2:clause_13"], parse_clinical_text("汗出"), channels=["太阳"]
        )
        negative = await knowledge.assess_candidate_evidence(
            ["1.2:clause_13"], parse_clinical_text("无汗"), channels=["太阳"]
        )
        assert positive == same and negative != positive and cache.hits >= 2


def test_compact_parser_pulse_negation_unknown_and_history() -> None:
    """复合脉象不漏缓，未问汗不等于无汗；历史与当前冲突分别保留。"""
    facts = parse_clinical_text("昨日发热，今日无发热；脉浮而缓，未问及汗出")
    current = {item.key: item.polarity for item in facts.current_findings()}
    assert current["pulse_floating"] == current["pulse_relaxed"] == "positive"
    assert current["sweating"] == "unknown" and current["fever"] == "negative"
    with pytest.raises(ValueError):
        compact_to_clinical(
            CompactClinicalExtraction.model_validate(
                {
                    "findings": [
                        {"key": "sweating", "polarity": "unknown", "source_text": "不存在的原文"}
                    ]
                }
            ),
            "发热",
            "doctor_input",
        )
