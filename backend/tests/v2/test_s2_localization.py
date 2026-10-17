"""S2 定位、引用守卫与有限修正测试；合成事实不代表医学金标准。

使用注入模型检查预算、引用身份与三值边界，不访问真实 Provider 或 RAG。
"""

import asyncio
import json

import pytest

from backend.app.clinical_v2.channel_localization import (
    LocalizationInvalidError,
    localize_channels,
)
from backend.app.clinical_v2.extraction import ClinicalLLMUnavailableError
from backend.app.clinical_v2.schemas import ChannelLocalizationV2, ClinicalEvidenceV2
from backend.app.core.llm import ClinicalStageStreamError


class LocalizeLLM:
    """为 S2 返回固定结构化候选，验证端口契约。"""

    def __init__(self, result: ChannelLocalizationV2) -> None:
        """注入合成结果，不对患者事实进行分类。"""
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


def valid_localization() -> ChannelLocalizationV2:
    """构造有真实引用且两侧互斥的合成定位。"""
    return ChannelLocalizationV2(
        status="resolved",
        candidates=[
            {
                "candidate_id": "candidate-1",
                "six_channels": ["太阳"],
                "support_fact_ids": ["fact-a"],
                "counter_fact_ids": ["fact-b"],
                "short_basis": "仅供引用协议测试",
            }
        ],
        proceed_to_retrieval=True,
        clarification_questions=[],
        note="合成测试",
    )


def overlapping_localization() -> ChannelLocalizationV2:
    """构造 Schema 合法但事实引用互相矛盾的草稿。"""
    result = valid_localization()
    result.candidates[0].counter_fact_ids = ["fact-a"]
    return result


class RepairLLM(LocalizeLLM):
    """可观察模型修正过程，强制两次调用共享同一阶段截止时间。"""

    def __init__(self, outputs: list[ChannelLocalizationV2]) -> None:
        """保存输出序列和请求副本，不存真实病例。"""
        super().__init__(outputs[0])
        self.outputs = outputs
        self.messages: list[list[dict[str, str]]] = []
        self.deadlines: list[float | None] = []
        self.deadline_calls = 0

    def clinical_deadline(self, stage: str) -> float:
        """创建一次阶段预算；测试验证修正没有重置预算。"""
        assert stage == "S2"
        self.deadline_calls += 1
        return asyncio.get_running_loop().time() + 2

    async def ainvoke_clinical_structured_stream(
        self,
        messages: list[dict[str, str]],
        schema: type,
        *,
        stage: str,
        deadline: float | None = None,
    ) -> ChannelLocalizationV2:
        """返回下一个草稿，记录模型是否收到完整修正上下文。"""
        assert stage == "S2" and schema is ChannelLocalizationV2
        self.messages.append([dict(message) for message in messages])
        self.deadlines.append(deadline)
        return self.outputs[min(len(self.messages) - 1, len(self.outputs) - 1)]


async def test_overlap_is_corrected_by_model_once_without_editing_facts() -> None:
    """模型修正引用后才能发布，程序不删反证、不改原始事实或无效草稿。"""
    facts = evidence()
    before = facts.model_dump_json()
    bad, good = overlapping_localization(), valid_localization()
    llm = RepairLLM([bad, good])
    result = await localize_channels(facts, llm)
    assert result == good
    assert facts.model_dump_json() == before
    assert bad.candidates[0].counter_fact_ids == ["fact-a"]
    assert len(llm.messages) == 2
    assert llm.messages[0][1]["content"] == before
    assert llm.messages[1][-2] == {"role": "assistant", "content": bad.model_dump_json()}
    assert "support_counter_overlap" in llm.messages[1][-1]["content"]
    assert llm.deadline_calls == 1
    assert llm.deadlines[0] == llm.deadlines[1]


async def test_persistent_overlap_is_rejected_with_bounded_safe_details() -> None:
    """两次都违反引用约束时明确失败，安全详情不能包含患者文本或模型内容。"""
    llm = RepairLLM([overlapping_localization()])
    with pytest.raises(LocalizationInvalidError) as caught:
        await localize_channels(evidence(), llm)
    assert len(llm.messages) == 2
    assert caught.value.details == {
        "validation_rule": "support_counter_overlap",
        "candidate_index": 0,
        "conflicting_reference_count": 1,
        "failure_code": "LOCALIZATION_INVALID",
        "error_category": "localization_reference_validation",
        "stage": "s2_localize",
        "repair_attempts": 1,
    }
    assert "发热" not in json.dumps(caught.value.details, ensure_ascii=False)
    assert "fact-a" not in str(caught.value.details)


async def test_fact_may_support_one_candidate_and_counter_another() -> None:
    """互斥约束只适用于同一候选，不能误伤候选之间的鉴别。"""
    result = valid_localization()
    result.status = "ambiguous"
    result.candidates.append(
        result.candidates[0].model_copy(
            update={
                "candidate_id": "candidate-2",
                "six_channels": ["少阳"],
                "support_fact_ids": ["fact-b"],
                "counter_fact_ids": ["fact-a"],
            }
        )
    )
    assert await localize_channels(evidence(), LocalizeLLM(result)) == result


async def test_past_and_current_facts_remain_distinct_during_repair() -> None:
    """过去和当前事实即使概念相同仍按身份引用，不用病例关键词决定六经。"""
    facts = evidence()
    facts.findings[0].time_scope = "past"
    llm = RepairLLM([overlapping_localization(), valid_localization()])
    result = await localize_channels(facts, llm)
    original = json.loads(llm.messages[1][1]["content"])
    assert [f["time_scope"] for f in original["findings"]] == ["past", "current", "current"]
    assert result.candidates[0].support_fact_ids == ["fact-a"]
    assert result.candidates[0].counter_fact_ids == ["fact-b"]


async def test_unknown_fact_still_cannot_be_used_as_counter_evidence() -> None:
    """修正机制不能把 unknown 当反证或自动删除引用使其通过。"""
    facts = evidence()
    facts.findings[1].polarity = "unknown"
    llm = RepairLLM([valid_localization()])
    with pytest.raises(LocalizationInvalidError) as caught:
        await localize_channels(facts, llm)
    assert caught.value.details["validation_rule"] == "unknown_as_counter"
    assert len(llm.messages) == 2


async def test_provider_failure_does_not_trigger_reference_repair() -> None:
    """超时或断流不能借引用修正机制重新请求。"""

    class UnavailableLLM(LocalizeLLM):
        """模拟流式入口在结构化结果形成前失败。"""

        calls = 0

        async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type) -> object:
            """首次调用就模拟传输失败。"""
            self.calls += 1
            raise ClinicalStageStreamError("S2", "STREAM_INTERRUPTED")

    llm = UnavailableLLM(valid_localization())
    with pytest.raises(ClinicalLLMUnavailableError):
        await localize_channels(evidence(), llm)
    assert llm.calls == 1


async def test_repair_exhausts_original_budget_without_third_call() -> None:
    """修正也受原总预算约束，不能再获得一轮完整 S2 时限。"""

    class SlowRepairLLM(RepairLLM):
        """第二次生成超过剩余预算的合成模型。"""

        def clinical_deadline(self, stage: str) -> float:
            """设置极短预算验证外层计时器而不等待真实模型。"""
            return asyncio.get_running_loop().time() + 0.03

        async def ainvoke_clinical_structured_stream(
            self,
            messages: list[dict[str, str]],
            schema: type,
            *,
            stage: str,
            deadline: float | None = None,
        ) -> ChannelLocalizationV2:
            """只在第二次调用等待，触发原阶段的总超时。"""
            result = await super().ainvoke_clinical_structured_stream(
                messages, schema, stage=stage, deadline=deadline
            )
            if len(self.messages) == 2:
                await asyncio.sleep(1)
            return result

    llm = SlowRepairLLM([overlapping_localization(), valid_localization()])
    with pytest.raises(ClinicalLLMUnavailableError) as caught:
        await localize_channels(evidence(), llm)
    assert caught.value.details["failure_code"] == "S2_STAGE_TIMEOUT"
    assert len(llm.messages) == 2
