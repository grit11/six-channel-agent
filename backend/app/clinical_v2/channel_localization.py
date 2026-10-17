"""S2 六经定位：LLM 读取已核验事实和本地指导卡，不调用在线 RAG。

只校验事实引用和状态路由；引用失败可由模型在同一阶段预算内修正一次，程序不代改
支持或反证。指导卡的医学内容仍待专家审核，开发结果不得伪称已审。
"""

import asyncio
import json
from pathlib import Path
from typing import cast

from backend.app.clinical_v2.extraction import ClinicalLLMUnavailableError
from backend.app.clinical_v2.schemas import ChannelLocalizationV2, ClinicalEvidenceV2
from backend.app.core.exceptions import ApplicationError, ProviderUnavailableError
from backend.app.core.ports import ClinicalStreamingLLM, StructuredLLM


class LocalizationInvalidError(ApplicationError):
    """六经模型捏造事实引用或误用冲突事实。"""

    code = "LOCALIZATION_INVALID"


def _validate_localization(evidence: ClinicalEvidenceV2, result: ChannelLocalizationV2) -> None:
    """核验候选身份及逐候选引用；同一事实可跨候选扮演不同角色。"""
    known = {item.fact_id: item for item in evidence.findings}
    conflicts = {item for group in evidence.conflicts for item in group}
    ids = [item.candidate_id for item in result.candidates]
    if len(ids) != len(set(ids)):
        raise LocalizationInvalidError(
            "六经候选身份重复",
            details={"validation_rule": "duplicate_candidate"},
        )
    for index, candidate in enumerate(result.candidates):
        cited = set(candidate.support_fact_ids + candidate.counter_fact_ids)
        if not cited.issubset(known) or (result.proceed_to_retrieval and cited & conflicts):
            raise LocalizationInvalidError(
                "定位引用不存在或尚有冲突的事实",
                details={"validation_rule": "invalid_fact_reference", "candidate_index": index},
            )
        overlap = set(candidate.support_fact_ids) & set(candidate.counter_fact_ids)
        if overlap:
            raise LocalizationInvalidError(
                "同一事实不能同时支持和反对同一候选",
                details={
                    "validation_rule": "support_counter_overlap",
                    "candidate_index": index,
                    "conflicting_reference_count": len(overlap),
                },
            )
        if any(known[item].polarity == "unknown" for item in candidate.counter_fact_ids):
            raise LocalizationInvalidError(
                "未知事实不能当作反证",
                details={"validation_rule": "unknown_as_counter", "candidate_index": index},
            )
    if any(not set(q.source_fact_ids).issubset(known) for q in result.clarification_questions):
        raise LocalizationInvalidError(
            "补问引用了不存在的事实", details={"validation_rule": "invalid_question_reference"}
        )
    # 指导卡状态仅作为工程元数据，由发布门禁检查，模型不能自我签发审核通过。
    json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


async def localize_channels(
    evidence: ClinicalEvidenceV2, llm: StructuredLLM
) -> ChannelLocalizationV2:
    """模型形成候选，引用失败只修正一次且共享 S2 预算，不做医学规则兜底。"""
    directory = Path(__file__).parent / "prompts"
    prompt, guidance = await asyncio.gather(
        asyncio.to_thread((directory / "s2_channel_localize.md").read_text, encoding="utf-8"),
        asyncio.to_thread((directory / "six_channel_reference.md").read_text, encoding="utf-8"),
    )
    messages = [
        {"role": "system", "content": prompt + "\n" + guidance},
        {"role": "user", "content": evidence.model_dump_json()},
    ]
    deadline_factory = getattr(llm, "clinical_deadline", None)
    deadline = deadline_factory("S2") if deadline_factory else None
    # 与 S1 来源修复同样有限：完整草稿只回送给本次模型，不进入日志或业务快照。
    for attempt in range(2):
        try:
            async with asyncio.timeout_at(deadline):
                if hasattr(llm, "ainvoke_clinical_structured_stream"):
                    result = await cast(
                        ClinicalStreamingLLM, llm
                    ).ainvoke_clinical_structured_stream(
                        messages, ChannelLocalizationV2, stage="S2", deadline=deadline
                    )
                else:
                    result = await llm.ainvoke_structured(messages, ChannelLocalizationV2)
        except TimeoutError as exc:
            raise ClinicalLLMUnavailableError(
                "S2 阶段总预算耗尽", details={"failure_code": "S2_STAGE_TIMEOUT"}
            ) from exc
        except ProviderUnavailableError as exc:
            raise ClinicalLLMUnavailableError("S2 临床模型不可用", details=exc.details) from exc
        try:
            _validate_localization(evidence, result)
            return result
        except LocalizationInvalidError as exc:
            if attempt:
                exc.details.update(
                    failure_code="LOCALIZATION_INVALID",
                    error_category="localization_reference_validation",
                    stage="s2_localize",
                    repair_attempts=attempt,
                )
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": result.model_dump_json()},
                    {
                        "role": "user",
                        "content": "软件引用校验未通过："
                        + exc.message
                        + "；校验位置："
                        + json.dumps(exc.details, ensure_ascii=False)
                        + "。请依据原始已核验事实重新检查并返回完整 ChannelLocalizationV2。"
                        "仅修正候选身份和事实引用，不新增事实、不改变原事实极性或时点。"
                        "同一候选的 support_fact_ids 与 counter_fact_ids 必须互斥；"
                        "跨候选可分别引用同一事实。过去与当前表现须分别按已有 fact_id 引用，"
                        "不能把症状减轻当作明确消失；无法确定证据方向时用缺失维度或澄清问题表达，"
                        "不要为了过校验随意删去反证或强行得出六经结论。",
                    },
                ]
            )
    raise RuntimeError("不可达的 S2 引用修复状态")
