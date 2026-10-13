"""V2 软件完整性验证：核对原文与身份，不使用医学词表重判 LLM。

依赖 V2 DTO；只证明来源存在，不能证明模型医学解释正确。
"""

import hashlib

from backend.app.clinical_v2.schemas import (
    ClinicalEvidenceV2,
    ComparisonV2,
    FindingV2,
    RetrievalBundleV2,
    S1ExtractionDraft,
    SourceSpanV2,
)
from backend.app.core.exceptions import ApplicationError


class ExtractionInvalidError(ApplicationError):
    """S1 原文或身份校验失败，拒绝伪造出处。"""

    code = "EXTRACTION_INVALID"


def verify_comparison_evidence(
    comparison: ComparisonV2,
    evidence: ClinicalEvidenceV2,
    retrieval: RetrievalBundleV2,
) -> ComparisonV2:
    """核对比较引用的事实和来源身份，不按医学权重修改模型顺序。"""
    facts = {item.fact_id for item in evidence.findings}
    sources = set(retrieval.retrieved_evidence_ids) | {
        ref.evidence_id for ref in comparison.source_evidence
    }
    candidates = {item.candidate_id: item for item in retrieval.formula_candidates}
    checked = []
    for path in comparison.ordered_paths:
        if path.candidate_id not in candidates:
            raise ExtractionInvalidError("比较引用了未召回候选")
        cited_facts = set(path.support_fact_ids + path.counter_fact_ids)
        cited_sources = set(path.support_evidence_ids + path.counter_evidence_ids)
        if not cited_facts.issubset(facts) or not cited_sources.issubset(sources):
            raise ExtractionInvalidError("比较引用超出本轮事实或来源包")
        if path.formula_name != candidates[path.candidate_id].formula_name:
            raise ExtractionInvalidError("比较方名与候选身份不一致")
        checked.append(path.model_copy(update={"verification_status": "grounded"}))
    return comparison.model_copy(update={"ordered_paths": checked})


def align_source(raw_text: str, source_text: str, occurrence: int = 0) -> SourceSpanV2:
    """匹配连续且未改写的原文；重复片段必须使用明确出现序号。"""
    start = -1
    for _ in range(occurrence + 1):
        start = raw_text.find(source_text, start + 1)
        if start < 0:
            raise ExtractionInvalidError("模型来源片段无法在原文中对齐")
    return SourceSpanV2(start=start, end=start + len(source_text), text=source_text)


def validate_extraction(
    raw_text: str,
    draft: S1ExtractionDraft,
    *,
    revision: int = 0,
    terminology: dict[str, str] | None = None,
) -> ClinicalEvidenceV2:
    """程序签发事实身份；受控映射由调用方提供，未映射概念保持原样。"""
    ids: dict[str, str] = {}
    findings: list[FindingV2] = []
    for item in draft.findings:
        if item.local_id in ids:
            raise ExtractionInvalidError("模型局部事实身份重复")
        span = align_source(raw_text, item.source_text, item.occurrence)
        identity = f"{revision}:{span.start}:{span.end}:{item.concept}:{item.time_scope}"
        fact_id = "fact-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        if fact_id in ids.values():
            raise ExtractionInvalidError("模型重复报告同一事实，应先核对冲突")
        ids[item.local_id] = fact_id
        findings.append(
            FindingV2(
                fact_id=fact_id,
                concept=item.concept,
                normalized_key=(terminology or {}).get(item.concept),
                polarity=item.polarity,
                time_scope=item.time_scope,
                source_span=span,
                review_status=(
                    "needs_clarification"
                    if item.needs_clarification or item.polarity == "unknown"
                    else "verified_source"
                ),
            )
        )
    if any(not set(group).issubset(ids) or len(group) < 2 for group in draft.conflicts):
        raise ExtractionInvalidError("模型冲突引用不存在的事实")
    return ClinicalEvidenceV2(
        raw_text=raw_text,
        findings=findings,
        conflicts=[[ids[item] for item in group] for group in draft.conflicts],
        unresolved_phrases=[align_source(raw_text, text) for text in draft.unresolved_phrases],
        chief_complaint_span=(
            align_source(raw_text, draft.chief_complaint_text)
            if draft.chief_complaint_text
            else None
        ),
        doctor_revision=revision,
    )
