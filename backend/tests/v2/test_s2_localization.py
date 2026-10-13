"""S2 定位与状态守卫测试：unknown/conflicting 不得进入检索。"""

from backend.app.clinical_v2.channel_localization import localize_channels
from backend.app.clinical_v2.schemas import ChannelLocalizationV2, ClinicalEvidenceV2


class LocalizeLLM:
    """为 S2 返回固定结构化候选，验证端口契约。"""

    def __init__(self, result: ChannelLocalizationV2) -> None:
        self.result = result

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """S2 不应调用文本接口。"""
        raise AssertionError("S2 必须使用结构化接口")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
        """返回预构造的定位结果。"""
        assert schema is ChannelLocalizationV2
        return self.result


def evidence() -> ClinicalEvidenceV2:
    """构造来源已验证的最小事实。"""
    return ClinicalEvidenceV2(
        raw_text="发热恶寒，脉浮",
        findings=[
            {
                "fact_id": "fact-a",
                "concept": "发热",
                "polarity": "positive",
                "time_scope": "current",
                "source_span": {"start": 0, "end": 2, "text": "发热"},
                "review_status": "verified_source",
            },
            {
                "fact_id": "fact-b",
                "concept": "恶寒",
                "polarity": "positive",
                "time_scope": "current",
                "source_span": {"start": 2, "end": 4, "text": "恶寒"},
                "review_status": "verified_source",
            },
            {
                "fact_id": "fact-c",
                "concept": "脉浮",
                "polarity": "positive",
                "time_scope": "current",
                "source_span": {"start": 6, "end": 7, "text": "浮"},
                "review_status": "verified_source",
            },
        ],
    )


async def test_unknown_localization_is_valid_and_not_retrievable() -> None:
    """只有不确定状态时允许空候选，并保持 S3 禁止标志。"""
    result = ChannelLocalizationV2(
        status="unknown",
        candidates=[],
        proceed_to_retrieval=False,
        clarification_questions=[],
        note="依据不足",
    )
    localized = await localize_channels(evidence(), LocalizeLLM(result))
    assert localized.status == "unknown"
    assert localized.proceed_to_retrieval is False


async def test_resolved_candidate_must_cite_existing_fact() -> None:
    """模型引用不存在的事实时拒绝，而不是静默删除引用。"""
    result = ChannelLocalizationV2(
        status="resolved",
        candidates=[
            {
                "candidate_id": "c1",
                "six_channels": ["太阳"],
                "support_fact_ids": ["missing"],
                "missing_dimensions": [],
                "retrieval_focus": ["表证"],
                "short_basis": "测试",
            }
        ],
        proceed_to_retrieval=True,
        clarification_questions=[],
        note="测试",
    )
    try:
        await localize_channels(evidence(), LocalizeLLM(result))
    except Exception as exc:
        assert getattr(exc, "code", None) == "LOCALIZATION_INVALID"
    else:
        raise AssertionError("应拒绝不存在的事实引用")
