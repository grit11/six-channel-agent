"""S1 离线契约测试：验证模型事实可追溯且不依赖旧词表或医学规则。"""

import pytest

from backend.app.clinical_v2.extraction import extract_clinical_facts
from backend.app.clinical_v2.schemas import S1ExtractionDraft


class FakeLLM:
    """只返回预先构造的模型草稿，测试 Provider 端口而非规则路径。"""

    def __init__(self, draft: S1ExtractionDraft) -> None:
        self.draft = draft
        self.messages: list[dict[str, str]] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """S1 不应调用文本端口。"""
        raise AssertionError("S1 必须使用结构化端口")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
        """返回结构化草稿。"""
        assert schema is S1ExtractionDraft
        self.messages = messages
        return self.draft


@pytest.mark.asyncio
async def test_s1_keeps_unmapped_findings_and_three_value_polarity() -> None:
    """口语和词表外概念保留，明确未知不被转换为阴性。"""
    text = "汗始终出不来，便中夹脓血，暂无明确里热口渴表现"
    draft = S1ExtractionDraft(
        findings=[
            {
                "local_id": "a",
                "concept": "汗未出",
                "polarity": "negative",
                "time_scope": "current",
                "source_text": "汗始终出不来",
            },
            {
                "local_id": "b",
                "concept": "便中脓血",
                "polarity": "positive",
                "time_scope": "current",
                "source_text": "便中夹脓血",
            },
            {
                "local_id": "c",
                "concept": "口渴表现",
                "polarity": "unknown",
                "time_scope": "current",
                "source_text": "暂无明确里热口渴表现",
                "needs_clarification": True,
            },
        ],
        unresolved_phrases=[],
    )
    llm = FakeLLM(draft)
    result = await extract_clinical_facts(text, llm)
    assert [item.polarity for item in result.findings] == ["negative", "positive", "unknown"]
    assert result.findings[1].normalized_key is None
    assert result.findings[0].source_span.text == "汗始终出不来"
    assert "V2.1 S1 Few-shot" in llm.messages[0]["content"]


@pytest.mark.asyncio
async def test_s1_keeps_temporal_distinction() -> None:
    """同一概念跨时点不合并，供后续定位阶段识别病程变化。"""
    text = "昨日发热，今日已退热但恶寒"
    draft = S1ExtractionDraft(
        findings=[
            {
                "local_id": "past-fever",
                "concept": "发热",
                "polarity": "positive",
                "time_scope": "past",
                "source_text": "昨日发热",
            },
            {
                "local_id": "current-fever",
                "concept": "发热",
                "polarity": "negative",
                "time_scope": "current",
                "source_text": "今日已退热",
            },
            {
                "local_id": "current-chills",
                "concept": "恶寒",
                "polarity": "positive",
                "time_scope": "current",
                "source_text": "恶寒",
            },
        ]
    )
    result = await extract_clinical_facts(text, FakeLLM(draft))
    assert [(item.concept, item.time_scope) for item in result.findings] == [
        ("发热", "past"),
        ("发热", "current"),
        ("恶寒", "current"),
    ]


async def test_source_repair_is_bounded_and_model_based() -> None:
    """首次来源改写只允许一次模型修复，程序不把医学词表替换成患者原文。"""
    from backend.app.clinical_v2.verification import ExtractionInvalidError

    class RepairLLM(FakeLLM):
        """依次返回无效和合法来源，记录真实端口次数。"""

        def __init__(self, repair: bool) -> None:
            """控制第二次是否能修复，避免无限重试。"""
            self.calls = 0
            self.repair = repair

        async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
            """第二次请求必须携带软件错误反馈，不调用规则。"""
            self.calls += 1
            if self.calls == 2:
                assert "软件来源校验未通过" in messages[-1]["content"]
            return S1ExtractionDraft(
                findings=[
                    {
                        "local_id": "a",
                        "concept": "汗出",
                        "polarity": "negative",
                        "time_scope": "current",
                        "source_text": (
                            "汗始终出不来" if self.repair and self.calls == 2 else "无汗"
                        ),
                    }
                ]
            )

    valid = RepairLLM(True)
    result = await extract_clinical_facts("汗始终出不来", valid)
    assert valid.calls == 2 and result.findings[0].polarity == "negative"
    invalid = RepairLLM(False)
    with pytest.raises(ExtractionInvalidError):
        await extract_clinical_facts("汗始终出不来", invalid)
    assert invalid.calls == 2
