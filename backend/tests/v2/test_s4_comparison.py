"""S4 上下文与来源回归测试，依赖注入模型和知识端口，不调用真实模型或用户数据库。

验证紧凑投影保留事实语义与鉴别条件，完整来源仍供审核展开；不测试医学诊断正确性。
"""

import json
import hashlib
from unittest.mock import AsyncMock

import pytest

from backend.app.clinical_v2.comparison import (
    ComparisonUngroundedError,
    _clip_text,
    compare_formula_candidates,
)
from backend.app.clinical_v2.schemas import (
    CandidateComparisonV2,
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ComparisonDraftV2,
    ComparisonV2,
    FormulaEvidenceCandidateV2,
    RetrievalBundleV2,
    RetrievalQueryPlanV2,
)
from backend.app.clinical_v2.verification import verify_comparison_evidence
from backend.app.schemas.evidence import Evidence


class ComparisonLLM:
    """返回指定顺序的结构化比较结果。"""

    def __init__(self, result: ComparisonV2 | ComparisonDraftV2) -> None:
        """保存返回对象与待核对的真实节点输入。"""
        if isinstance(result, ComparisonV2):
            payload = result.model_dump(mode="python")
            payload.pop("source_evidence", None)
            payload.pop("provisional_candidates", None)
            for item in payload["ordered_paths"]:
                item.pop("path_id", None)
                item.pop("formula_name", None)
                item.pop("verification_status", None)
            self.result = ComparisonDraftV2.model_validate(payload)
        else:
            self.result = result
        self.messages: list[dict[str, str]] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """S4 必须调用结构化接口。"""
        raise AssertionError("S4 不调用自由文本端口")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
        """返回模型相对顺序。"""
        assert schema is ComparisonDraftV2
        self.messages = messages
        return self.result


class RepairingLLM(ComparisonLLM):
    """第一次返回来源不完整草稿，第二次返回可签发草稿。"""

    def __init__(self, first: ComparisonDraftV2, second: ComparisonDraftV2) -> None:
        super().__init__(second)
        self.results = [first, second]
        self.calls = 0

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
        assert schema is ComparisonDraftV2
        self.messages = messages
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


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
        evidence,
        localization,
        retrieval,
        knowledge,
        ComparisonLLM(result),
        consultation_id="consult-1",
        revision=2,
    )
    assert checked.ordered_paths[0].candidate_id == "cb"
    assert "score" not in checked.ordered_paths[0].model_dump()
    assert checked.ordered_paths[0].verification_status == "grounded"
    assert checked.ordered_paths[0].path_id == "path-" + hashlib.sha256(
        b"consult-1:2:cb"
    ).hexdigest()[:20]
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


@pytest.mark.asyncio
async def test_s4_sends_compact_evidence_context() -> None:
    """S4 请求保留可审计来源，但不再发送完整 DTO、检索轨迹和原始元数据。"""
    evidence, localization, retrieval = setup()
    result = ComparisonV2(status="insufficient", ordered_paths=[])
    llm = ComparisonLLM(result)
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []

    await compare_formula_candidates(evidence, localization, retrieval, knowledge, llm)

    payload = json.loads(llm.messages[-1]["content"])
    assert payload["context_format"] == "compact_evidence_context_v1"
    assert "raw_text" not in payload["clinical_evidence"]
    assert "retrieval_trace" not in payload["retrieval"]
    assert "retrieved_evidence_ids" not in payload["retrieval"]
    assert "query_plan" not in payload["retrieval"]
    assert payload["clinical_evidence"]["findings"][0]["source_text"] == "发热"
    assert payload["retrieval"]["formula_candidates"][0]["candidate_id"] == "ca"


async def test_compact_differential_preserves_dimensions_and_full_review_sources() -> None:
    """去重不能丢失反证、安全约束、来源版本或医生可展开的完整资料。"""
    evidence, localization, retrieval = setup()
    dimensions = {f"维度{i}": "有条件甲；若条件乙不存在则不能据此判断。" for i in range(12)}
    source = Evidence(
        evidence_id="1.2:diff-a",
        knowledge_id="diff-a",
        knowledge_type="differential",
        knowledge_version="1.2",
        source="测试来源",
        source_level="teaching",
        content="完整鉴别来源。" * 200,
        retrieval_score=0.8,
        retrieval_method="hybrid",
        metadata={
            "title": "候选对照",
            "candidates": ["桂枝汤证", "麻黄汤证"],
            "key_dimensions": dimensions,
            "review_status": "needs_expert_review",
            "red_flags": ["风险提示"],
            "clinical_use_policy": "只辅助医生",
            "decision_rules": list(dimensions.values()),
            "text_for_embedding": "重复文本" * 1000,
        },
    )
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = [source]
    llm = ComparisonLLM(ComparisonV2(status="insufficient", ordered_paths=[]))
    original = source.model_dump(mode="json")
    result = await compare_formula_candidates(evidence, localization, retrieval, knowledge, llm)
    compact = json.loads(llm.messages[-1]["content"])["differential_sources"][0]
    assert compact["key_dimensions"] == dimensions
    assert compact["red_flags"] == ["风险提示"]
    assert compact["clinical_use_policy"] == "只辅助医生"
    assert compact["review_status"] == "needs_expert_review"
    assert compact["knowledge_version"] == "1.2"
    assert compact["excerpt"] == ""
    assert "metadata" not in compact and "retrieval_score" not in compact
    assert source.model_dump(mode="json") == original
    assert result.source_evidence[-1].excerpt == source.content[:1200]
    knowledge.search_differential.assert_awaited_once_with(
        ["桂枝汤", "麻黄汤"], query=None, top_k=10
    )


async def test_compact_context_preserves_unknown_negative_time_and_unresolved_text() -> None:
    """事实投影保留未知、阴性、时点、冲突和未解析原文，不能把缺失变为否定。"""
    _, localization, retrieval = setup()
    text = "不渴；汗不明；服药前发热；难辨。"
    findings = []
    for ident, concept, source_text, polarity, time_scope in [
        ("fact-a", "口渴", "不渴", "negative", "current"),
        ("fact-b", "汗", "汗不明", "unknown", "unspecified"),
        ("fact-c", "发热", "服药前发热", "positive", "before_treatment"),
    ]:
        start = text.index(source_text)
        findings.append(
            {
                "fact_id": ident,
                "concept": concept,
                "polarity": polarity,
                "time_scope": time_scope,
                "source_span": {
                    "start": start,
                    "end": start + len(source_text),
                    "text": source_text,
                },
                "review_status": "needs_clarification",
            }
        )
    start = text.index("难辨")
    evidence = ClinicalEvidenceV2.model_validate(
        {
            "raw_text": text,
            "findings": findings,
            "conflicts": [["fact-a", "fact-b"]],
            "unresolved_phrases": [{"start": start, "end": start + 2, "text": "难辨"}],
        }
    )
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = [
        Evidence(
            evidence_id="1.2:plain",
            knowledge_id="plain",
            knowledge_type="differential",
            knowledge_version="1.2",
            source="测试",
            source_level="teaching",
            content="没有关键维度时保留完整原文。" * 100,
            retrieval_score=1,
            retrieval_method="bm25",
        )
    ]
    llm = ComparisonLLM(ComparisonV2(status="insufficient", ordered_paths=[]))
    await compare_formula_candidates(evidence, localization, retrieval, knowledge, llm)
    payload = json.loads(llm.messages[-1]["content"])
    compact = payload["clinical_evidence"]
    assert [x["polarity"] for x in compact["findings"]] == ["negative", "unknown", "positive"]
    assert compact["findings"][2]["time_scope"] == "before_treatment"
    assert compact["unresolved_phrases"] == ["难辨"]
    assert compact["conflicts"] == [["fact-a", "fact-b"]]
    assert payload["differential_sources"][0]["excerpt"] == (
        knowledge.search_differential.return_value[0].content
    )


def test_excerpt_does_not_cut_incomplete_conditions() -> None:
    """来源过长只按完整句子摘录并标记省略；无完整句界时宁可保留全文。"""
    assert _clip_text("若甲成立且乙不成立才适用", 5) == "若甲成立且乙不成立才适用"
    excerpt = _clip_text("第一句。若甲成立且乙不成立才适用。", 8)
    assert excerpt.startswith("第一句。")
    assert "未展示内容不等于不存在" in excerpt
    assert "若甲" not in excerpt


def ranked_result(ranks: list[int | None], status: str = "ranked") -> ComparisonV2:
    """用真实候选和来源构造排名草稿，不包含模型外的医学评分。"""
    _, _, retrieval = setup()
    paths = [
        CandidateComparisonV2(
            path_id=f"p-{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            six_channels=["太阳"],
            formula_name=candidate.formula_name,
            rank=rank,
            support_fact_ids=["fact-a"],
            counter_fact_ids=[],
            missing_dimensions=[],
            support_evidence_ids=[candidate.source_evidence[0].evidence_id],
            counter_evidence_ids=[],
            brief_rationale="模型给出的优先级",
            differential_summary="测试结构一致性",
            verification_status="pending",
        )
        for candidate, rank in zip(retrieval.formula_candidates, ranks, strict=False)
    ]
    return ComparisonV2.model_validate({"status": status, "ordered_paths": paths})


async def test_s4_orders_serialized_paths_by_model_rank_without_mutating_draft() -> None:
    """数组按候选顺序而 rank 相反时，忠实采用模型 rank，且不改写原始草稿。"""
    evidence, localization, retrieval = setup()
    draft = ranked_result([2, 1])
    original = draft.model_dump()
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    result = await compare_formula_candidates(
        evidence, localization, retrieval, knowledge, ComparisonLLM(draft)
    )
    assert [path.candidate_id for path in result.ordered_paths] == ["cb", "ca"]
    assert [path.rank for path in result.ordered_paths] == [1, 2]
    assert all(path.verification_status == "grounded" for path in result.ordered_paths)
    assert draft.model_dump() == original
    verified = verify_comparison_evidence(result, evidence, retrieval)
    assert verified == result


@pytest.mark.parametrize("ranks", [[], [2], [None], [1, 1], [1, 3], [None, 1]])
async def test_s4_does_not_invent_missing_or_duplicate_rank(ranks: list[int | None]) -> None:
    """空排名、缺失、并列、跳号均不能靠数组位置补齐为确定排名。"""
    evidence, localization, retrieval = setup()
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    with pytest.raises(ComparisonUngroundedError, match="唯一连续"):
        await compare_formula_candidates(
            evidence, localization, retrieval, knowledge, ComparisonLLM(ranked_result(ranks))
        )


async def test_s4_rank_normalization_still_rejects_forged_evidence() -> None:
    """整理 rank 不能绕过来源校验，即使排名合法也拒绝伪造 Evidence ID。"""
    evidence, localization, retrieval = setup()
    draft = ranked_result([2, 1])
    draft.ordered_paths[0].support_evidence_ids = ["forged-source"]
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    with pytest.raises(ComparisonUngroundedError, match="不存在的证据"):
        await compare_formula_candidates(
            evidence, localization, retrieval, knowledge, ComparisonLLM(draft)
        )


@pytest.mark.parametrize("status", ["insufficient", "needs_clarification"])
async def test_s4_does_not_rank_undecided_result(status: str) -> None:
    """未决状态没有可靠 rank 时保留原顺序，不能由程序产生医学优先级。"""
    evidence, localization, retrieval = setup()
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    draft = ranked_result([None, None], status)
    result = await compare_formula_candidates(
        evidence, localization, retrieval, knowledge, ComparisonLLM(draft)
    )
    assert result.ordered_paths == []
    assert [p.candidate_id for p in result.provisional_candidates] == ["ca", "cb"]
    assert all(p.rank is None for p in result.provisional_candidates)


async def test_s4_rejects_formal_rank_in_undecided_draft() -> None:
    """未决状态即使带 rank 也不能发布为正式排名。"""
    evidence, localization, retrieval = setup()
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    draft = ranked_result([1], "needs_clarification")
    with pytest.raises(ComparisonUngroundedError, match="未决状态"):
        await compare_formula_candidates(
            evidence, localization, retrieval, knowledge, ComparisonLLM(draft)
        )


async def test_s4_allows_one_restricted_structural_repair() -> None:
    """结构/来源校验失败只触发一次受限修复，修复不改变候选池。"""
    evidence, localization, retrieval = setup()
    first = ComparisonDraftV2.model_validate(
        {
            "status": "ranked",
            "ordered_paths": [
                {
                    "candidate_id": "cb",
                    "six_channels": ["太阳"],
                    "rank": 1,
                    "support_fact_ids": ["fact-a"],
                    "counter_fact_ids": [],
                    "missing_dimensions": [],
                    "support_evidence_ids": ["1.2:ca"],
                    "counter_evidence_ids": [],
                    "brief_rationale": "来源待纠正",
                    "differential_summary": "测试",
                }
            ],
        }
    )
    second = first.model_copy(
        update={
            "ordered_paths": [
                first.ordered_paths[0].model_copy(update={"support_evidence_ids": ["1.2:cb"]})
            ]
        }
    )
    llm = RepairingLLM(first, second)
    knowledge = AsyncMock()
    knowledge.search_differential.return_value = []
    result = await compare_formula_candidates(
        evidence, localization, retrieval, knowledge, llm
    )
    assert llm.calls == 2
    assert result.ordered_paths[0].candidate_id == "cb"
