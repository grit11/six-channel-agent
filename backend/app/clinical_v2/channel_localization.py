"""S2 六经定位：LLM 读取已核验事实和本地指导卡，不调用在线 RAG。

只校验事实引用和状态路由；指导卡的医学内容仍待专家审核，开发结果不得伪称已审。
"""

import asyncio
import json
from pathlib import Path

from backend.app.clinical_v2.extraction import ClinicalLLMUnavailableError
from backend.app.clinical_v2.schemas import ChannelLocalizationV2, ClinicalEvidenceV2
from backend.app.core.exceptions import ApplicationError, ProviderUnavailableError
from backend.app.core.ports import StructuredLLM


class LocalizationInvalidError(ApplicationError):
    """六经模型捏造事实引用或误用冲突事实。"""

    code = "LOCALIZATION_INVALID"


async def localize_channels(
    evidence: ClinicalEvidenceV2, llm: StructuredLLM
) -> ChannelLocalizationV2:
    """模型形成多候选或未决，程序只核对真实事实身份，不做医学打分。"""
    directory = Path(__file__).parent / "prompts"
    prompt, guidance = await asyncio.gather(
        asyncio.to_thread((directory / "s2_channel_localize.md").read_text, encoding="utf-8"),
        asyncio.to_thread((directory / "six_channel_reference.md").read_text, encoding="utf-8"),
    )
    try:
        result = await llm.ainvoke_structured(
            [
                {"role": "system", "content": prompt + "\n" + guidance},
                {"role": "user", "content": evidence.model_dump_json()},
            ],
            ChannelLocalizationV2,
        )
    except ProviderUnavailableError as exc:
        raise ClinicalLLMUnavailableError("S2 临床模型不可用") from exc
    known = {item.fact_id: item for item in evidence.findings}
    conflicts = {item for group in evidence.conflicts for item in group}
    ids = [item.candidate_id for item in result.candidates]
    if len(ids) != len(set(ids)):
        raise LocalizationInvalidError("六经候选身份重复")
    for candidate in result.candidates:
        cited = set(candidate.support_fact_ids + candidate.counter_fact_ids)
        if not cited.issubset(known) or (result.proceed_to_retrieval and cited & conflicts):
            raise LocalizationInvalidError("定位引用不存在或尚有冲突的事实")
        if set(candidate.support_fact_ids) & set(candidate.counter_fact_ids):
            raise LocalizationInvalidError("同一事实不能同时支持和反对同一候选")
        if any(known[item].polarity == "unknown" for item in candidate.counter_fact_ids):
            raise LocalizationInvalidError("未知事实不能当作反证")
    if any(not set(q.source_fact_ids).issubset(known) for q in result.clarification_questions):
        raise LocalizationInvalidError("补问引用了不存在的事实")
    # 指导卡状态仅作为工程元数据，由发布门禁检查，模型不能自我签发审核通过。
    json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    return result
