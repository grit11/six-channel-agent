"""S1 LLM 语义抽取：通过通用端口调用模型并核验原文，不诊断、不规则兜底。

Prompt 在异步线程读入；Schema 只由 Adapter 注入一次，禁止暴露隐藏推理。
"""

import asyncio
import json
from pathlib import Path
from typing import cast

from backend.app.clinical_v2.schemas import ClinicalEvidenceV2, S1ExtractionDraft
from backend.app.clinical_v2.verification import ExtractionInvalidError, validate_extraction
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import ClinicalStreamingLLM, StructuredLLM


class ClinicalLLMUnavailableError(ProviderUnavailableError):
    """临床模型不可用，不切回旧规则系统。"""

    code = "LLM_UNAVAILABLE"


async def extract_clinical_facts(
    text: str,
    llm: StructuredLLM,
    *,
    revision: int = 0,
    terminology: dict[str, str] | None = None,
) -> ClinicalEvidenceV2:
    """抽取且对齐原文事实，未提供的术语映射保持空值。"""
    if not text.strip() or len(text) > 20000:
        raise ValueError("病例原文须在 1 至 20000 字符之间")
    prompt = await asyncio.to_thread(
        (Path(__file__).parent / "prompts/s1_clinical_extract.md").read_text,
        encoding="utf-8",
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"clinical_text": text, "approved_terminology": terminology or {}},
                ensure_ascii=False,
            ),
        },
    ]
    deadline_factory = getattr(llm, "clinical_deadline", None)
    deadline = deadline_factory("S1") if deadline_factory else None
    # 来源对齐失败只允许一次结构修复。仍由模型理解语义，程序不改极性或编造出处。
    for attempt in range(2):
        try:
            if hasattr(llm, "ainvoke_clinical_structured_stream"):
                draft = await cast(ClinicalStreamingLLM, llm).ainvoke_clinical_structured_stream(
                    messages, S1ExtractionDraft, stage="S1", deadline=deadline
                )
            else:
                draft = await llm.ainvoke_structured(messages, S1ExtractionDraft)
        except ProviderUnavailableError as exc:
            raise ClinicalLLMUnavailableError("S1 临床模型不可用", details=exc.details) from exc
        try:
            return validate_extraction(text, draft, revision=revision, terminology=terminology)
        except ExtractionInvalidError as exc:
            if attempt:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": draft.model_dump_json()},
                    {
                        "role": "user",
                        "content": "软件来源校验未通过："
                        + exc.message
                        + "。请仅修复原文引用、local_id/冲突引用及重复身份，所有 source_text、"
                        + "chief_complaint_text 和 unresolved_phrases 必须为连续原样子串；"
                        + "无法确定则保留真实未决短语，不增加诊断或原文没有的事实。",
                    },
                ]
            )
    raise RuntimeError("不可达的来源修复状态")
