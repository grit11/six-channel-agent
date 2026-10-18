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

    def __init__(
        self,
        message: str,
        *,
        validation_rule: str,
        field_path: str | None = None,
        item_index: int | None = None,
    ) -> None:
        """只保留可定位的规则与索引，不把原文写入异常详情。"""
        details: dict[str, object] = {"validation_rule": validation_rule}
        if field_path is not None:
            details["field_path"] = field_path
        if item_index is not None:
            details["item_index"] = item_index
        super().__init__(message, details=details)


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
            raise ExtractionInvalidError(
                "比较引用了未召回候选", validation_rule="candidate_not_in_pool"
            )
        cited_facts = set(path.support_fact_ids + path.counter_fact_ids)
        cited_sources = set(path.support_evidence_ids + path.counter_evidence_ids)
        if not cited_facts.issubset(facts) or not cited_sources.issubset(sources):
            raise ExtractionInvalidError(
                "比较引用超出本轮事实或来源包",
                validation_rule="comparison_reference_not_allowed",
            )
        if path.formula_name != candidates[path.candidate_id].formula_name:
            raise ExtractionInvalidError(
                "比较方名与候选身份不一致", validation_rule="formula_name_mismatch"
            )
        checked.append(path.model_copy(update={"verification_status": "grounded"}))
    return comparison.model_copy(update={"ordered_paths": checked})


def align_source(raw_text: str, source_text: str, occurrence: int = 0) -> SourceSpanV2:
    """匹配连续且未改写的原文；重复片段必须使用明确出现序号。"""
    start = -1
    for _ in range(occurrence + 1):
        start = raw_text.find(source_text, start + 1)
        if start < 0:
            raise ExtractionInvalidError(
                "模型来源片段无法在原文中对齐",
                validation_rule="source_span_not_found",
            )
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
    # 同一复合原文（如“脉浮而紧”）可以承载多个不同概念；只拒绝同一
    # 原文范围被同一概念/时点重复声明，避免把合法拆解误判为重复事实。
    span_owners: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for index, item in enumerate(draft.findings):
        if item.local_id in ids:
            raise ExtractionInvalidError(
                "模型局部事实身份重复",
                validation_rule="duplicate_local_id",
                field_path="findings.local_id",
                item_index=index,
            )
        try:
            span = align_source(raw_text, item.source_text, item.occurrence)
        except ExtractionInvalidError as exc:
            exc.details.update(field_path="findings.source_text", item_index=index)
            raise
        span_key = (span.start, span.end)
        owner = (item.concept, item.time_scope)
        owners = span_owners.setdefault(span_key, set())
        if owner in owners:
            raise ExtractionInvalidError(
                "多个事实引用了同一原文范围",
                validation_rule="duplicate_source_span",
                field_path="findings.source_text",
                item_index=index,
            )
        owners.add(owner)
        identity = f"{revision}:{span.start}:{span.end}:{item.concept}:{item.time_scope}"
        fact_id = "fact-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        if fact_id in ids.values():
            raise ExtractionInvalidError(
                "模型重复报告同一事实，应先核对冲突",
                validation_rule="duplicate_fact_identity",
                field_path="findings",
                item_index=index,
            )
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
    for index, group in enumerate(draft.conflicts):
        if len(group) < 2 or not set(group).issubset(ids):
            raise ExtractionInvalidError(
                "模型冲突引用不存在的事实",
                validation_rule="invalid_conflict_ref",
                field_path="conflicts",
                item_index=index,
            )
    unresolved: list[SourceSpanV2] = []
    for index, text in enumerate(draft.unresolved_phrases):
        try:
            unresolved.append(align_source(raw_text, text))
        except ExtractionInvalidError as exc:
            exc.details.update(
                validation_rule="invalid_unresolved_span",
                field_path="unresolved_phrases",
                item_index=index,
            )
            raise
    chief_span = None
    if draft.chief_complaint_text:
        try:
            chief_span = align_source(raw_text, draft.chief_complaint_text)
        except ExtractionInvalidError as exc:
            exc.details.update(
                validation_rule="chief_complaint_span_mismatch",
                field_path="chief_complaint_text",
            )
            raise
    return ClinicalEvidenceV2(
        raw_text=raw_text,
        findings=findings,
        conflicts=[[ids[item] for item in group] for group in draft.conflicts],
        unresolved_phrases=unresolved,
        chief_complaint_span=chief_span,
        doctor_revision=revision,
    )
