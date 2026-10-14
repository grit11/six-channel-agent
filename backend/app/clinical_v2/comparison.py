"""S4 批量 LLM 方证比较：模型负责相对判断，程序只核验候选、事实和来源身份。

本模块只负责 S4 的证据上下文裁剪、结构化模型调用和来源绑定；完整知识记录仍由
KnowledgeService 按 Evidence ID 提供展开，不在模型请求中重复发送检索 DTO、轨迹和冗余元数据。
不执行旧 Python 规则、不计算医学分数、不自动改写模型顺序。
"""

import asyncio
import json
from pathlib import Path

from backend.app.clinical_v2.extraction import ClinicalLLMUnavailableError
from backend.app.clinical_v2.retrieval import _evidence_ref
from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ComparisonV2,
    RetrievalBundleV2,
)
from backend.app.core.exceptions import ApplicationError, ProviderUnavailableError
from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.evidence import Evidence


class ComparisonUngroundedError(ApplicationError):
    """S4 引用不存在或模型生成候选时拒绝发布。"""

    code = "COMPARISON_UNGROUNDED"


def _clip_text(value: str, limit: int) -> str:
    """限制模型上下文中的摘录长度，同时保留来源可回查的 Evidence ID。"""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    # 不截断半句后把残片当完整条件；找不到完整句界时保留全文。
    end = max(text.rfind(mark, 0, limit) for mark in ("。", "！", "？", "\n"))
    return text[: end + 1] + "…（摘录未完整，未展示内容不等于不存在）" if end >= 0 else text


def _compact_clinical_evidence(evidence: ClinicalEvidenceV2) -> dict[str, object]:
    """只保留 S4 比较所需的事实身份、原文表现、极性和时点。"""
    return {
        "findings": [
            {
                "fact_id": item.fact_id,
                "concept": item.concept,
                "polarity": item.polarity,
                "time_scope": item.time_scope,
                "source_text": item.source_span.text,
                "review_status": item.review_status,
            }
            for item in evidence.findings
        ],
        "conflicts": evidence.conflicts,
        "unresolved_phrases": [item.text for item in evidence.unresolved_phrases],
        "doctor_revision": evidence.doctor_revision,
    }


def _compact_localization(localization: ChannelLocalizationV2) -> dict[str, object]:
    """只保留六经候选及其事实依据，避免把完整 S2 DTO重复送入 S4。"""
    return {
        "status": localization.status,
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "six_channels": item.six_channels,
                "support_fact_ids": item.support_fact_ids,
                "counter_fact_ids": item.counter_fact_ids,
                "missing_dimensions": item.missing_dimensions,
                "retrieval_focus": item.retrieval_focus,
                "short_basis": item.short_basis,
            }
            for item in localization.candidates
        ],
        "note": localization.note,
    }


def _compact_retrieval(retrieval: RetrievalBundleV2) -> dict[str, object]:
    """保留候选名称、来源 ID 和短摘录，排除检索轨迹及原始分数元数据。"""
    return {
        "formula_candidates": [
            {
                "candidate_id": item.candidate_id,
                "formula_name": item.formula_name,
                "six_channel_tags": item.six_channel_tags,
                "provenance": item.provenance,
                "coverage_notes": item.coverage_notes,
                "source_evidence": [
                    {
                        "evidence_id": ref.evidence_id,
                        "knowledge_type": ref.knowledge_type,
                        "knowledge_version": ref.knowledge_version,
                        "clause_no": ref.clause_no,
                        "excerpt": _clip_text(ref.excerpt, 500),
                        "review_status": ref.review_status,
                    }
                    for ref in item.source_evidence
                ],
            }
            for item in retrieval.formula_candidates
        ],
        "retrieval_status": retrieval.retrieval_status,
    }


def _compact_differential_sources(differential: list[Evidence]) -> list[dict[str, object]]:
    """从鉴别 Evidence 提取短摘录和关键维度，不发送完整 metadata。"""
    compact: list[dict[str, object]] = []
    for item in differential:
        metadata = item.metadata
        dimensions = metadata.get("key_dimensions", {})
        if isinstance(dimensions, dict) and dimensions:
            # 关键鉴别条件整体保留，不按患者词项筛选或只留下支持条件。
            key_dimensions = {str(key): value for key, value in dimensions.items()}
        else:
            key_dimensions = {}
        compact.append(
            {
                "evidence_id": item.evidence_id,
                "knowledge_type": item.knowledge_type,
                "knowledge_version": item.knowledge_version,
                "source_level": item.source_level,
                "review_status": metadata.get("review_status", "unknown"),
                "clinical_use_policy": metadata.get("clinical_use_policy", ""),
                "red_flags": metadata.get("red_flags", []),
                "title": metadata.get("title", ""),
                "candidates": metadata.get("candidates", []),
                "key_dimensions": key_dimensions,
                # 结构化鉴别维度与 content/decision_rules/text_for_embedding 重复，
                # 只发送一份；其他知识版本没有关键维度时保留来源正文，避免丢失依据。
                "excerpt": "" if key_dimensions else item.content,
            }
        )
    return compact


async def compare_formula_candidates(
    evidence: ClinicalEvidenceV2,
    localization: ChannelLocalizationV2,
    retrieval: RetrievalBundleV2,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
    *,
    differential_top_k: int = 10,
) -> ComparisonV2:
    """一次结构化调用比较 S3 候选；程序签发 grounded 状态。"""
    guide_dir = Path(__file__).parent / "prompts"
    prompt, guide = await asyncio.gather(
        asyncio.to_thread((guide_dir / "s4_formula_comparison.md").read_text, encoding="utf-8"),
        asyncio.to_thread(
            (guide_dir / "formula_discrimination_guide.md").read_text, encoding="utf-8"
        ),
    )
    candidates = {item.candidate_id: item for item in retrieval.formula_candidates}
    if not candidates:
        return ComparisonV2(
            status="insufficient",
            ordered_paths=[],
            unresolved_reason="没有真实方证候选",
        )
    differential = await knowledge.search_differential(
        list(item.formula_name for item in candidates.values()),
        query=retrieval.query_plan.differential_query,
        top_k=differential_top_k,
    )
    # S4 只接收紧凑证据上下文；完整 DTO、检索轨迹和来源 metadata 通过 Evidence ID
    # 保留可回查性，但不重复占用模型上下文窗口。
    payload = {
        "context_format": "compact_evidence_context_v1",
        "clinical_evidence": _compact_clinical_evidence(evidence),
        "localization": _compact_localization(localization),
        "retrieval": _compact_retrieval(retrieval),
        "differential_sources": _compact_differential_sources(differential),
    }
    try:
        draft = await llm.ainvoke_structured(
            [
                {"role": "system", "content": prompt + "\n" + guide},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            ComparisonV2,
        )
    except ProviderUnavailableError as exc:
        raise ClinicalLLMUnavailableError("S4 模型不可用，无法生成证据比较") from exc
    fact_ids = {item.fact_id for item in evidence.findings}
    known_ids = {item.fact_id for item in evidence.findings if item.polarity != "unknown"}
    evidence_ids = {
        ref.evidence_id for candidate in candidates.values() for ref in candidate.source_evidence
    } | {item.evidence_id for item in differential}
    seen_candidates: set[str] = set()
    paths = []
    for position, path in enumerate(draft.ordered_paths, 1):
        if path.candidate_id not in candidates or path.candidate_id in seen_candidates:
            raise ComparisonUngroundedError("比较结果引用了不存在或重复的方证候选")
        seen_candidates.add(path.candidate_id)
        if not set(path.support_fact_ids + path.counter_fact_ids).issubset(known_ids):
            raise ComparisonUngroundedError("比较结果引用了不存在或未知的事实")
        if not set(path.support_evidence_ids + path.counter_evidence_ids).issubset(evidence_ids):
            raise ComparisonUngroundedError("比较结果引用了不存在的证据")
        expected_name = candidates[path.candidate_id].formula_name
        if path.formula_name != expected_name:
            raise ComparisonUngroundedError("比较结果方名与候选池不一致")
        candidate_sources = {
            ref.evidence_id for ref in candidates[path.candidate_id].source_evidence
        }
        if not set(path.support_evidence_ids) & candidate_sources:
            raise ComparisonUngroundedError("路径缺少当前候选自身的来源支持")
        if draft.status == "ranked" and path.rank != position:
            raise ComparisonUngroundedError("路径顺序与 rank 不一致")
        paths.append(path.model_copy(update={"verification_status": "grounded"}))
    if draft.status == "ranked" and not paths:
        raise ComparisonUngroundedError("ranked 结果必须包含至少一条路径")
    if len({path.path_id for path in paths}) != len(paths):
        raise ComparisonUngroundedError("比较路径身份重复")
    if any(not set(q.source_fact_ids).issubset(fact_ids) for q in draft.next_questions):
        raise ComparisonUngroundedError("补问引用了不存在的事实")
    # 来源由程序附加，不能相信模型自行生成的摘录或校验标志。
    refs = {
        ref.evidence_id: ref
        for candidate in candidates.values()
        for ref in candidate.source_evidence
    }
    refs.update({item.evidence_id: _evidence_ref(item) for item in differential})
    return draft.model_copy(update={"ordered_paths": paths, "source_evidence": list(refs.values())})
