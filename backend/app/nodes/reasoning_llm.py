"""结构化八纲与六经候选模型节点。

职责：调用注入的 StructuredLLM，使用专用 Prompt 对规则基线进行结构化校正，
并验证模型只能引用当前事实和真实来源。
边界：不创建 Provider、不访问 FastAPI/数据库、不执行向量检索；模型失败或来源校验失败
明确抛错，绝不静默回退规则结果。
"""

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.reasoning import ChannelCandidate, EightPrinciplesState

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


class ChannelCandidateList(BaseModel):
    """结构化模型返回的宽候选列表，不允许额外隐藏字段。"""

    model_config = ConfigDict(extra="forbid")
    candidates: list[ChannelCandidate] = Field(min_length=1, max_length=6)


async def _prompt(name: str) -> str:
    """读取节点专用 Prompt；不存在时明确失败，不使用万能 Prompt。"""
    path = _PROMPT_DIR / name
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


def _json_message(value: object) -> str:
    """将结构化输入序列化为模型上下文，不包含客户端或隐藏推理。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _finding_tokens(evidence: ClinicalEvidence) -> set[str]:
    """允许模型引用的事实键和原文片段集合。"""
    return {
        token
        for finding in evidence.current_findings()
        for token in (finding.key, finding.source_text)
        if token
    }


def _validate_axis(
    result: EightPrinciplesState,
    evidence: ClinicalEvidence,
    framework_ids: set[str],
) -> EightPrinciplesState:
    """拒绝模型补出的事实和伪造框架 ID，保持三值与来源边界。"""
    tokens = _finding_tokens(evidence)
    for axis in (
        result.yin_yang,
        result.exterior_interior,
        result.cold_heat,
        result.deficiency_excess,
    ):
        for item in axis.supporting_findings + axis.contradicting_findings:
            key = item.split(":", 1)[-1]
            if item not in tokens and key not in tokens:
                raise ProviderUnavailableError("八纲模型引用了当前事实之外的内容")
            if not any(
                finding.polarity != "unknown"
                and (item in {finding.key, finding.source_text} or key == finding.key)
                for finding in evidence.current_findings()
            ):
                raise ProviderUnavailableError("八纲模型把未知事实当成支持或反证")
        unknown_keys = {
            finding.key for finding in evidence.current_findings() if finding.polarity == "unknown"
        }
        if not set(axis.unknown_key_findings).issubset(unknown_keys):
            raise ProviderUnavailableError("八纲模型的未知列表引用了已知事实")
    if not set(result.framework_evidence_ids).issubset(framework_ids):
        raise ProviderUnavailableError("八纲模型引用了不存在的框架来源")
    return result


async def analyze_eight_principles_with_llm(
    evidence: ClinicalEvidence,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
) -> EightPrinciplesState:
    """结构化调用八纲模型，并核对事实与框架来源。"""
    framework = await knowledge.search_core_framework(
        " ".join(item.source_text or item.key for item in evidence.current_findings()), top_k=5
    )
    prompt = await _prompt("eight_principles.md")
    result = await llm.ainvoke_structured(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _json_message(
                    {
                        "clinical_evidence": evidence.model_dump(mode="json"),
                        "framework": [item.model_dump(mode="json") for item in framework],
                    }
                ),
            },
        ],
        EightPrinciplesState,
    )
    return _validate_axis(result, evidence, {item.evidence_id for item in framework})


async def generate_channel_with_llm(
    evidence: ClinicalEvidence,
    principles: EightPrinciplesState,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
    *,
    top_k: int = 5,
) -> list[ChannelCandidate]:
    """结构化生成宽候选，要求候选和来源 ID 均来自当前输入。"""
    framework = await knowledge.search_core_framework(principles.summary, top_k=8)
    prompt = await _prompt("channel_candidate.md")
    raw = await llm.ainvoke_structured(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _json_message(
                    {
                        "clinical_evidence": evidence.model_dump(mode="json"),
                        "eight_principles": principles.model_dump(mode="json"),
                        "framework": [item.model_dump(mode="json") for item in framework],
                    }
                ),
            },
        ],
        ChannelCandidateList,
    )
    allowed_findings = _finding_tokens(evidence)
    allowed_ids = {item.evidence_id for item in framework}
    candidates = raw.candidates
    if not candidates:
        raise ProviderUnavailableError("六经模型未返回可审核候选")
    allowed_channels = {"太阳", "阳明", "少阳", "太阴", "少阴", "厥阴"}
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ProviderUnavailableError("六经模型输出了重复候选身份")
    for candidate in candidates:
        if candidate.status == "excluded":
            raise ProviderUnavailableError("模型不能自行硬排除六经候选")
        if candidate.classic_evidence_ids or candidate.formula_evidence_ids:
            raise ProviderUnavailableError("Pass A 不得提前引用经典或方证来源")
        if candidate.evidence_assessments or candidate.pass_b_score is not None:
            raise ProviderUnavailableError("Pass A 不得生成后续来源校正结果")
        if not set(candidate.six_channels).issubset(allowed_channels):
            raise ProviderUnavailableError("六经模型输出了未知六经")
        for item in (
            candidate.supporting_findings
            + candidate.contradicting_findings
            + candidate.missing_key_findings
        ):
            if item not in allowed_findings and item.split(":", 1)[-1] not in allowed_findings:
                raise ProviderUnavailableError("六经模型引用了当前事实之外的内容")
        for item in candidate.supporting_findings + candidate.contradicting_findings:
            if not any(
                finding.polarity != "unknown"
                and (
                    item in {finding.key, finding.source_text}
                    or item.split(":", 1)[-1] == finding.key
                )
                for finding in evidence.current_findings()
            ):
                raise ProviderUnavailableError("六经模型把未知事实当成支持或反证")
        if not set(candidate.framework_evidence_ids).issubset(allowed_ids):
            raise ProviderUnavailableError("六经模型引用了不存在的框架来源")
        if candidate.pass_a_score is None:
            raise ProviderUnavailableError("六经模型缺少 Pass A 排序分")
    # 校验全部输出后按独立分排序；提供方返回顺序不应决定哪些候选进入 Pass B。
    candidates.sort(key=lambda candidate: (-(candidate.pass_a_score or 0), candidate.candidate_id))
    return candidates[:top_k]
