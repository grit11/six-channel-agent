"""紧凑候选回归：模型调用数、事实来源、三值与来源去重，不调用在线服务。"""

import json
from unittest.mock import patch

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import StructuredResult
from backend.app.graph.six_channel_graph import STAGES, build_six_channel_graph
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.channel_candidate import generate_channel_candidates
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.reasoning_llm import PassACandidateDraft
from backend.app.schemas.evidence import Evidence


class ReasoningLLM:
    """候选草稿替身，记录请求以核对载荷与次数。"""

    def __init__(self, invalid: str | None = None) -> None:
        """指定需拒绝的越界字段，不预先写模型排序分。"""
        self.invalid = invalid
        self.calls: list[dict[str, object]] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """禁止绕过 Schema。"""
        raise AssertionError("unexpected text call")

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """返回最小候选并注入单一可审查错误。"""
        self.calls.append(json.loads(messages[-1]["content"]))
        draft: dict[str, object] = {"six_channels": ["太阳"], "supporting_keys": ["fever"]}
        if self.invalid in {"unknown_support", "fact", "history"}:
            draft["supporting_keys"] = [
                "tongue" if self.invalid == "unknown_support" else "invented"
            ]
        elif self.invalid == "channel":
            draft["six_channels"] = ["未知经"]
        elif self.invalid == "exclude":
            draft["status"] = "excluded"
        elif self.invalid == "premature_source":
            draft["classic_evidence_ids"] = ["1.2:clause_1"]
        elif self.invalid == "source":
            draft["framework_evidence_ids"] = ["1.2:invented"]
        elif self.invalid == "score":
            draft["pass_a_score"] = 0.9
        return schema.model_validate({"candidates": [draft]})


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """加载五类真实知识，不在夹具伪造医学来源。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


def test_pass_a_schema_excludes_server_only_fields() -> None:
    """模型无法填写分数、来源 ID、最终审核与校正结果。"""
    fields = PassACandidateDraft.model_json_schema()["properties"]
    assert (
        not {"pass_a_score", "pass_b_score", "candidate_id", "framework_evidence_ids", "status"}
        & fields.keys()
    )
    with pytest.raises(ValueError):
        PassACandidateDraft.model_validate({"six_channels": ["太阳"], "pass_a_score": 0.7})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "unknown_support",
        "source",
        "fact",
        "history",
        "channel",
        "exclude",
        "premature_source",
        "score",
    ],
)
async def test_reasoning_rejects_untrusted_fields(
    knowledge: KnowledgeService, invalid: str
) -> None:
    """合法 JSON 也不能升级未知事实、伪造来源或代替医生排除。"""
    facts = parse_clinical_text("发热")
    with pytest.raises((ProviderUnavailableError, ValueError)):
        await generate_channel_candidates(
            facts, analyze_eight_principles(facts), knowledge, llm=ReasoningLLM(invalid)
        )


@pytest.mark.asyncio
async def test_graph_uses_one_candidate_model_and_deterministic_axes(
    knowledge: KnowledgeService,
) -> None:
    """S2 只调用一次模型，八纲规则与草稿独立，医案不自动检索。"""
    model = ReasoningLLM()
    facts = parse_clinical_text("发热，脉浮")
    with patch.object(knowledge, "search_case", side_effect=AssertionError("不能自动医案")):
        graph = build_six_channel_graph(knowledge, reasoning_llm=model)
        result = await graph.ainvoke({"clinical_evidence": facts, "top_k": 3})
    assert result["eight_principles"] == analyze_eight_principles(facts)
    assert result["channel_candidates"][0].pass_a_score == 0.5
    assert len(model.calls) == 1
    assert "clinical_evidence" not in model.calls[0]
    assert len(model.calls[0]["framework"]) <= 4
    assert result["completed_stages"] == STAGES and result["similar_cases"] == []


@pytest.mark.asyncio
async def test_rules_mode_does_not_call_model(knowledge: KnowledgeService) -> None:
    """规则模式保留全部六经弱候选，不自动软最大化成单分类。"""
    facts = parse_clinical_text("发热")
    candidates = await generate_channel_candidates(
        facts, analyze_eight_principles(facts), knowledge, top_k=6
    )
    assert len(candidates) == 6 and any(item.status == "weak" for item in candidates)


@pytest.mark.asyncio
async def test_pass_b_duplicate_sources_do_not_change_correction(
    knowledge: KnowledgeService,
) -> None:
    """重复来源不加分；来源正反条件仍使用可复算校正。"""
    evidence = parse_clinical_text("发热，无汗，脉浮紧")
    principles = analyze_eight_principles(evidence)
    original = await generate_channel_candidates(
        evidence, principles, knowledge, llm=ReasoningLLM()
    )
    classic = knowledge.search_classic
    formula = knowledge.search_formula

    async def repeat_classic(
        query: str,
        semantic_channels: list[str] | None = None,
        clause_nos: list[str] | None = None,
        top_k: int = 8,
    ) -> list[Evidence]:
        """只重复真实来源。"""
        return (
            await classic(
                query, semantic_channels=semantic_channels, clause_nos=clause_nos, top_k=top_k
            )
        ) * 2

    async def repeat_formula(
        query: str,
        channels: list[str] | None = None,
        association_scope: list[str] | None = None,
        top_k: int = 8,
    ) -> list[Evidence]:
        """复制同一方证，不新造引用。"""
        return (
            await formula(
                query, channels=channels, association_scope=association_scope, top_k=top_k
            )
        ) * 2

    with (
        patch.object(knowledge, "search_classic", side_effect=repeat_classic),
        patch.object(knowledge, "search_formula", side_effect=repeat_formula),
    ):
        repeated = await generate_channel_candidates(
            evidence, principles, knowledge, llm=ReasoningLLM()
        )
    assert original[0].pass_b_score == repeated[0].pass_b_score
    item = original[0]
    support = {
        term for audit in item.evidence_assessments for term in audit.supporting_requirements
    }
    contradict = {
        term for audit in item.evidence_assessments for term in audit.contradicting_requirements
    }
    assert item.evidence_adjustment == pytest.approx(
        0.25 * len(support) / max(1, len(support | contradict))
        - 0.35 * len(contradict) / max(1, len(support | contradict))
    )
