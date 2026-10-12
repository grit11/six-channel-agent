"""六经宽候选模型端口：依赖注入模型和知识边界，只提出有事实来源的草稿。

八纲由 Python 计算；本模块不赋分、不生成 Evidence ID，不管理审核或最终诊断。
"""

import asyncio
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.reasoning import EightPrinciplesState

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PassACandidateDraft(BaseModel):
    """最小候选草稿；全部事实键须属于本轮当前事实，禁止模型排序和签署结论。"""

    model_config = ConfigDict(extra="forbid")
    six_channels: list[Literal["太阳", "阳明", "少阳", "太阴", "少阴", "厥阴"]] = Field(
        min_length=1, max_length=6
    )
    provisional_syndrome: str | None = Field(default=None, max_length=150)
    supporting_keys: list[str] = Field(default_factory=list, max_length=30)
    contradicting_keys: list[str] = Field(default_factory=list, max_length=30)
    missing_discriminators: list[str] = Field(default_factory=list, max_length=10)


class ChannelCandidateList(BaseModel):
    """保留一至六个宽候选；不要求单标签分类或归一化概率。"""

    model_config = ConfigDict(extra="forbid")
    candidates: list[PassACandidateDraft] = Field(min_length=1, max_length=6)


async def generate_channel_with_llm(
    evidence: ClinicalEvidence,
    principles: EightPrinciplesState,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
    *,
    top_k: int = 6,
) -> list[PassACandidateDraft]:
    """一次调用提出宽候选，先校验全部草稿再交给 Python 核验和赋分。"""
    framework = await knowledge.search_core_framework(principles.summary, top_k=4)
    path = _PROMPT_DIR / "channel_candidate.md"
    prompt = await asyncio.to_thread(
        path.read_text,
        encoding="utf-8",
    )
    facts = evidence.current_findings()
    known = {item.key for item in facts if item.polarity != "unknown"}
    unknown = {item.key for item in facts if item.polarity == "unknown"}
    raw = await llm.ainvoke_structured(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "facts": [
                            item.model_dump(
                                include={"key", "polarity", "source_text", "time_scope"}
                            )
                            for item in facts
                            if item.source_text
                        ],
                        "known_finding_keys": sorted(known),
                        "unknown_finding_keys": sorted(unknown),
                        "eight_principles": {
                            name: getattr(principles, name).lean
                            for name in (
                                "yin_yang",
                                "exterior_interior",
                                "cold_heat",
                                "deficiency_excess",
                            )
                        },
                        "framework": [item.content[:600] for item in framework],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        ChannelCandidateList,
    )
    identities: set[tuple[str, ...]] = set()
    for draft in raw.candidates:
        identity = tuple(sorted(draft.six_channels))
        if len(set(identity)) != len(identity) or identity in identities:
            raise ProviderUnavailableError("六经模型输出重复候选或六经")
        identities.add(identity)
        if not set(draft.supporting_keys + draft.contradicting_keys).issubset(known):
            raise ProviderUnavailableError("六经模型把未知、历史或来源外事实作为支持或反证")
        if set(draft.supporting_keys) & set(draft.contradicting_keys):
            raise ProviderUnavailableError("同一草稿事实不能同时支持和反驳")
        # 未提及事实也可以是缺口，但只能使用原解析器的受控词项。
        from backend.app.nodes.clinical_parser import _UNKNOWN_KEYS

        if not set(draft.missing_discriminators).issubset(set(_UNKNOWN_KEYS) - known):
            raise ProviderUnavailableError("模型缺口不属于未确认的规范事实")
    return raw.candidates[:top_k]
