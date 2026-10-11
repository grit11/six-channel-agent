"""独立 QA 单元回归：六类意图、精确工具边界、来源层级、风险与模型约束。

读取真实知识来源；模型用替身，不访问在线模型、会诊或密钥。
"""

from typing import TypeVar
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from backend.app.agents.knowledge_qa_agent import (
    ExcerptSelection,
    KnowledgeQAAgent,
    QueryClassification,
)
from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    InvalidBusinessRequestError,
    KnowledgeRetrievalError,
    ProviderUnavailableError,
)
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.knowledge import KnowledgeQARequest

T = TypeVar("T", bound=BaseModel)


@pytest.fixture
async def knowledge() -> KnowledgeService:
    """真实五类知识，显式关闭 .env 配置读取。"""
    return await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )


@pytest.mark.parametrize(
    "query,category,tool",
    [
        ("第35条原文是什么？", "exact_clause", "get_clause_by_id"),
        ("真武汤证的核心表现有哪些？", "formula", "get_formula_by_name"),
        ("桂枝汤证与麻黄汤证怎么鉴别？", "differential", "search_differential"),
        ("六经理论中合病与并病", "framework", "search_core_framework"),
        ("范中林有没有无汗脉浮紧的类似医案？", "case", "search_case"),
        ("无汗脉浮紧的条文、方证及医案", "mixed", "retrieve_evidence_bundle"),
    ],
)
async def test_six_routes(
    knowledge: KnowledgeService, query: str, category: str, tool: str
) -> None:
    """RP-21 六类自然语言例子触发真实知识工具，不生成患者候选。"""
    answer = await KnowledgeQAAgent(knowledge).answer(KnowledgeQARequest(query=query, top_k=1))
    assert answer.plan.query_type == category
    assert tool in answer.tools_used
    assert answer.sections
    assert all(
        item.evidence_id in {e.evidence_id for e in answer.evidence_catalog}
        for section in answer.sections
        for item in section.items
    )
    assert "reasoning_paths" not in answer.model_dump()


async def test_exact_targets_never_search(
    knowledge: KnowledgeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """精确未命中不得召回相近来源，条文多目标保留缺失，长方名不误取子串。"""
    search = AsyncMock(side_effect=AssertionError("不允许模糊检索"))
    monkeypatch.setattr(knowledge, "_search", search)
    agent = KnowledgeQAAgent(knowledge)
    answer = await agent.answer(KnowledgeQARequest(query="第35条与第9999条原文"))
    assert answer.status == "partial"
    assert answer.missing_evidence == ["未找到条文：9999"]
    assert answer.sections[0].items[0].text == answer.evidence_catalog[0].content
    assert answer.evidence_catalog[0].retrieval_method == "exact"
    missing = await agent.answer(KnowledgeQARequest(query="查方证", formula_name="不存在方剂"))
    assert missing.status == "no_evidence" and missing.evidence_catalog == []
    names, _ = await knowledge.identify_query_targets("桂枝加桂汤的主症")
    assert names == ["桂枝加桂汤"]
    search.assert_not_called()


async def test_historical_layers_and_risk(knowledge: KnowledgeService) -> None:
    """五类来源及历史范围分别展示，高风险字段保留，正文不复制组成剂量。"""
    agent = KnowledgeQAAgent(knowledge)
    answer = await agent.answer(KnowledgeQARequest(query="恶寒脉浮", query_type="mixed"))
    for section in answer.sections:
        for item in section.items:
            source = next(e for e in answer.evidence_catalog if e.evidence_id == item.evidence_id)
            assert source.source_level == section.source_level
            assert source.knowledge_type == section.knowledge_type
            assert item.source_field in source.metadata
            assert item.text in (
                [source.metadata[item.source_field]]
                if isinstance(source.metadata[item.source_field], str)
                else source.metadata[item.source_field]
            )
    risk = await agent.answer(KnowledgeQARequest(query="三物小白散的主症"))
    assert risk.evidence_catalog[0].metadata["risk_level"] == "high"
    assert "巴豆" in risk.evidence_catalog[0].metadata["risk_flags"]
    assert all(
        item.source_field != "composition_classic"
        for section in risk.sections
        for item in section.items
    )
    historical = await agent.answer(KnowledgeQARequest(query="桂枝汤医案", query_type="case"))
    assert all(section.source_level == "L5" for section in historical.sections)
    assert all(
        item.source_field not in {"initial_visit", "followup_and_discussion"}
        for section in historical.sections
        for item in section.items
    )


async def test_missing_and_context(knowledge: KnowledgeService) -> None:
    """空召回不编造来源，主动背景只作为检索文本，不构造临床事实。"""
    answer = await KnowledgeQAAgent(knowledge).answer(KnowledgeQARequest(query="unmatched_z9x8q7"))
    assert answer.status == "no_evidence"
    assert answer.sections == [] and answer.evidence_catalog == []
    contextual = await KnowledgeQAAgent(knowledge).answer(
        KnowledgeQARequest(query="相似医案", context_text="无汗脉浮紧")
    )
    assert contextual.context_used
    assert contextual.plan.query_type == "case"
    with pytest.raises(ValidationError):
        KnowledgeQARequest(query="  ")
    with pytest.raises(ValidationError):
        KnowledgeQARequest(query="问题", consultation_id="clinical")
    with pytest.raises(InvalidBusinessRequestError):
        await KnowledgeQAAgent(knowledge).answer(
            KnowledgeQARequest(query="条文", query_type="exact_clause")
        )


async def test_explicit_exact_overrides_question_text(knowledge: KnowledgeService) -> None:
    """显式精确目标优先于问题中的背景意图，不被医案或理论词改路由。"""
    agent = KnowledgeQAAgent(knowledge)
    clause = await agent.answer(KnowledgeQARequest(query="与其他医案比较", clause_id="35"))
    assert clause.plan.query_type == "exact_clause"
    assert clause.tools_used == ["get_clause_by_id"]
    formula = await agent.answer(KnowledgeQARequest(query="相关六经理论", formula_name="麻黄汤"))
    assert formula.plan.query_type == "formula"
    assert formula.tools_used == ["get_formula_by_name"]
    mixed = await agent.answer(
        KnowledgeQARequest(query="第9999条与医案", query_type="mixed", top_k=1)
    )
    assert "未找到条文：9999" in mixed.missing_evidence
    assert not any(section.knowledge_type == "classic_clause" for section in mixed.sections)


class SelectionLLM:
    """结构化模型替身，只分类/选择摘录，记录调用以检验精确目标不被改写。"""

    def __init__(self, indices: list[int]) -> None:
        """固定选择序号，测试越界或空相关性。"""
        self.indices = indices
        self.calls: list[type[BaseModel]] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """QA 不应调用自由文本生成接口。"""
        raise AssertionError("不允许自由文本回答")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """返回受限六类意图或摘录序号，不生成医学内容。"""
        self.calls.append(schema)
        return schema.model_validate(
            {"query_type": "mixed"}
            if schema is QueryClassification
            else {"item_indices": self.indices}
        )


async def test_model_only_selects_verified_excerpts(knowledge: KnowledgeService) -> None:
    """模型不能覆盖精确目标；空选择为证据不足，伪造序号明确失败。"""
    llm = SelectionLLM([0])
    answer = await KnowledgeQAAgent(knowledge, llm).answer(
        KnowledgeQARequest(query="第35条原文是什么")
    )
    assert answer.plan.query_type == "exact_clause"
    assert llm.calls == [ExcerptSelection]
    assert answer.sections[0].items[0].text == answer.evidence_catalog[0].content
    assert answer.answer_mode == "llm"
    empty = await KnowledgeQAAgent(knowledge, SelectionLLM([])).answer(
        KnowledgeQARequest(query="第35条")
    )
    assert empty.status == "no_evidence" and empty.sections == []
    for invalid in [[-1], [9999], [0, 0]]:
        with pytest.raises(KnowledgeRetrievalError):
            await KnowledgeQAAgent(knowledge, SelectionLLM(invalid)).answer(
                KnowledgeQARequest(query="第35条")
            )


async def test_model_failure_is_not_silent_fallback(knowledge: KnowledgeService) -> None:
    """模型不可用时抛出已有业务错误，不伪装成功来源回答。"""
    llm = SelectionLLM([0])
    llm.ainvoke_structured = AsyncMock(side_effect=ProviderUnavailableError("测试模型不可用"))
    with pytest.raises(ProviderUnavailableError):
        await KnowledgeQAAgent(knowledge, llm).answer(KnowledgeQARequest(query="第35条"))


def test_qa_llm_settings_require_configuration() -> None:
    """QA 单独启用模型也必须在启动阶段校验配置。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, knowledge_qa_mode="llm")
