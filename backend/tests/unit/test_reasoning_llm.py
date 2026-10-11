"""S1.2 结构化八纲与六经两步校正测试。

使用模型端口替身验证专用 Prompt、真实来源 ID 和当前事实边界；不调用在线模型，
也不把模型输出当临床金标准。
"""

from unittest.mock import patch

import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import StructuredResult
from backend.app.graph.six_channel_graph import build_six_channel_graph
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.channel_candidate import generate_channel_candidates
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.eight_principles import analyze_eight_principles_node
from backend.app.nodes.reasoning_llm import analyze_eight_principles_with_llm
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import ChannelCandidate, EightPrinciplesState


class ReasoningLLM:
    """按目标 Schema 返回固定结构化结果，并检查节点使用了专用 Prompt。"""

    def __init__(self, *, invalid: str | None = None) -> None:
        """配置需要拒绝的字段，模拟模型越界输出。"""
        self.invalid = invalid
        self.calls: list[str] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """推理节点不得退回普通文本模式。"""
        raise AssertionError("unexpected text call")

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """只返回测试用结构，隐藏推理不进入结果。"""
        self.calls.append(messages[0]["content"])
        if schema is EightPrinciplesState:
            value = {
                "yin_yang": {
                    "lean": "right",
                    "score_left": 0.0,
                    "score_right": 0.8,
                    "supporting_findings": ["阳:fever"],
                    "contradicting_findings": [],
                    "unknown_key_findings": ["tongue"],
                },
                "exterior_interior": {
                    "lean": "left",
                    "score_left": 0.8,
                    "score_right": 0.0,
                    "supporting_findings": ["表:fever"],
                    "contradicting_findings": [],
                    "unknown_key_findings": [],
                },
                "cold_heat": {
                    "lean": "right",
                    "score_left": 0.4,
                    "score_right": 0.4,
                    "supporting_findings": ["阳:fever"],
                    "contradicting_findings": [],
                    "unknown_key_findings": [],
                },
                "deficiency_excess": {
                    "lean": "uncertain",
                    "score_left": 0.0,
                    "score_right": 0.0,
                    "supporting_findings": [],
                    "contradicting_findings": [],
                    "unknown_key_findings": ["pulse"],
                },
                "summary": "测试八纲倾向",
                "framework_evidence_ids": [],
            }
            if self.invalid == "fact":
                value["yin_yang"]["supporting_findings"] = ["阳:未记录事实"]
            if self.invalid == "unknown_support":
                value["yin_yang"]["supporting_findings"] = ["阳:tongue"]
            if self.invalid == "source":
                value["framework_evidence_ids"] = ["1.2:invented"]
            return schema.model_validate(value)
        candidate = ChannelCandidate(
            candidate_id="channel_太阳",
            six_channels=["太阳"],
            provisional_syndrome="太阳候选",
            independent_score=0.7,
            pass_a_score=0.7,
            supporting_findings=["fever"],
            contradicting_findings=[],
            missing_key_findings=["sweating"],
            framework_evidence_ids=[],
            evidence_completeness=0.5,
            status="active",
        )
        if self.invalid == "channel":
            candidate.six_channels = ["未知经"]
        if self.invalid == "exclude":
            candidate.status = "excluded"
        if self.invalid == "premature_source":
            candidate.classic_evidence_ids = ["1.2:clause_35"]
        return schema.model_validate({"candidates": [candidate.model_dump(mode="json")]})


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """加载真实五类知识来源，框架 ID 不在测试中伪造。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


@pytest.mark.asyncio
async def test_reasoning_llm_uses_separate_prompt_and_preserves_unknown(
    knowledge: KnowledgeService,
) -> None:
    """结构化八纲保留 Unknown，且使用专用 Prompt。"""
    model = ReasoningLLM()
    result = await analyze_eight_principles_with_llm(
        parse_clinical_text("发热，脉浮"), knowledge, model
    )
    assert result.summary == "测试八纲倾向"
    assert "tongue" in result.yin_yang.unknown_key_findings
    assert "不得判断六经" in model.calls[0]


@pytest.mark.asyncio
async def test_reasoning_mode_rules_does_not_call_model(knowledge: KnowledgeService) -> None:
    """规则模式是显式基线，不因模型替身存在而暗中调用。"""
    model = ReasoningLLM()
    result = await analyze_eight_principles_node(parse_clinical_text("发热"), knowledge, None)
    assert result.cold_heat.score_right > 0
    assert model.calls == []


@pytest.mark.asyncio
async def test_reasoning_llm_rejects_invented_fact_and_channel(knowledge: KnowledgeService) -> None:
    """模型输出即使 Schema 合法，事实和六经越界也必须失败。"""
    with pytest.raises(ProviderUnavailableError):
        await analyze_eight_principles_with_llm(
            parse_clinical_text("发热"), knowledge, ReasoningLLM(invalid="fact")
        )
    with pytest.raises(ProviderUnavailableError):
        await generate_channel_candidates(
            parse_clinical_text("发热"),
            await analyze_eight_principles_with_llm(
                parse_clinical_text("发热"), knowledge, ReasoningLLM()
            ),
            knowledge,
            top_k=3,
            llm=ReasoningLLM(invalid="channel"),
        )


@pytest.mark.asyncio
async def test_channel_pass_a_and_pass_b_are_separate(knowledge: KnowledgeService) -> None:
    """候选保存 Pass A、来源校正和 Pass B，不以召回数量增加分数。"""
    evidence = parse_clinical_text("发热")
    principles = await analyze_eight_principles_with_llm(evidence, knowledge, ReasoningLLM())
    candidates = await generate_channel_candidates(
        evidence, principles, knowledge, top_k=3, llm=ReasoningLLM()
    )
    assert candidates
    assert all(item.pass_a_score is not None for item in candidates)
    assert all(item.pass_b_score is not None for item in candidates)
    assert all(-1 <= item.evidence_adjustment <= 1 for item in candidates)
    for item in candidates:
        support = {
            term
            for assessment in item.evidence_assessments
            for term in assessment.supporting_requirements
        }
        contradict = {
            term
            for assessment in item.evidence_assessments
            for term in assessment.contradicting_requirements
        }
        denominator = max(1, len(support | contradict))
        assert item.evidence_adjustment == pytest.approx(
            0.25 * len(support) / denominator - 0.35 * len(contradict) / denominator
        )
        assert item.pass_b_score == pytest.approx(
            max(0, min(1, item.pass_a_score + item.evidence_adjustment))
        )


@pytest.mark.asyncio
async def test_pass_b_duplicate_sources_do_not_change_correction(
    knowledge: KnowledgeService,
) -> None:
    """重复召回经典/方证不增加支持或惩罚，Pass B 按唯一条件核对。"""
    evidence = parse_clinical_text("发热，无汗，脉浮紧")
    model = ReasoningLLM()
    principles = await analyze_eight_principles_with_llm(evidence, knowledge, model)
    original = await generate_channel_candidates(evidence, principles, knowledge, llm=model)
    classic = knowledge.search_classic
    formula = knowledge.search_formula

    async def repeat_classic(
        query: str,
        semantic_channels: list[str] | None = None,
        clause_nos: list[str] | None = None,
        top_k: int = 8,
    ) -> list[Evidence]:
        """复制原服务来源作为重复召回替身。"""
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
        """复制方证来源但不创建新 Evidence ID。"""
        return (
            await formula(
                query, channels=channels, association_scope=association_scope, top_k=top_k
            )
        ) * 2

    with (
        patch.object(knowledge, "search_classic", side_effect=repeat_classic),
        patch.object(knowledge, "search_formula", side_effect=repeat_formula),
    ):
        repeated = await generate_channel_candidates(evidence, principles, knowledge, llm=model)
    assert repeated[0].evidence_adjustment == original[0].evidence_adjustment
    assert repeated[0].pass_b_score == original[0].pass_b_score


@pytest.mark.asyncio
async def test_graph_passes_reasoning_model_to_both_s1_2_nodes(
    knowledge: KnowledgeService,
) -> None:
    """真实子图使用注入模型完成八纲和六经两步，模型失败不会切回规则节点。"""
    model = ReasoningLLM()
    graph = build_six_channel_graph(knowledge, reasoning_llm=model)
    result = await graph.ainvoke(
        {"clinical_evidence": parse_clinical_text("发热，脉浮"), "top_k": 3}
    )
    assert result["eight_principles"].summary == "测试八纲倾向"
    assert result["channel_candidates"][0].pass_a_score == 0.7
    assert len(model.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unknown_support", "source", "fact"])
async def test_axis_rejects_unknown_as_support_and_forged_source(
    knowledge: KnowledgeService,
    invalid: str,
) -> None:
    """Schema 合法也不能把未知或来源外事实当当前支持。"""
    with pytest.raises(ProviderUnavailableError):
        await analyze_eight_principles_with_llm(
            parse_clinical_text("发热"), knowledge, ReasoningLLM(invalid=invalid)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["exclude", "premature_source"])
async def test_channel_pass_a_rejects_model_exclusion_and_later_stage_sources(
    knowledge: KnowledgeService,
    invalid: str,
) -> None:
    """模型宽召回不能越级引用或自行排除。"""
    evidence = parse_clinical_text("发热")
    with pytest.raises(ProviderUnavailableError):
        await generate_channel_candidates(
            evidence,
            await analyze_eight_principles_with_llm(evidence, knowledge, ReasoningLLM()),
            knowledge,
            llm=ReasoningLLM(invalid=invalid),
        )
