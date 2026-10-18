"""S3 六经引导 Hybrid Retrieval：只召回和归并真实来源，不做医学评分。

KnowledgeService 是唯一知识访问边界；本模块不直接调用向量数据库，也不把检索分数转成诊断分。
"""

import hashlib
from collections import defaultdict
from typing import Literal, cast

from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    EvidenceRefV2,
    FormulaEvidenceCandidateV2,
    RetrievalBundleV2,
    RetrievalQueryPlanV2,
)
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.evidence import Evidence


def build_query_plan(
    evidence: ClinicalEvidenceV2, localization: ChannelLocalizationV2
) -> RetrievalQueryPlanV2:
    """仅拼接已存在事实和模型检索焦点，绝不补造病例症状。"""
    current_positive = [
        item
        for item in evidence.findings
        if item.polarity == "positive" and item.time_scope in {"current", "unspecified"}
    ]
    current_negative = [
        item
        for item in evidence.findings
        if item.polarity == "negative" and item.time_scope in {"current", "unspecified"}
    ]
    positive = [item.concept for item in current_positive]
    negative = [item.concept for item in current_negative]
    after_treatment = [
        item
        for item in evidence.findings
        if item.time_scope == "after_treatment" and item.polarity == "positive"
    ]
    excluded_temporal = {
        item.fact_id: "after_treatment_current_continuity_unconfirmed" for item in after_treatment
    }
    focus = [
        phrase for candidate in localization.candidates for phrase in candidate.retrieval_focus
    ]
    channels: list[str] = list(
        dict.fromkeys(
            channel for candidate in localization.candidates for channel in candidate.six_channels
        )
    )
    terms = channels + positive + focus
    primary = " ".join(dict.fromkeys(term for term in terms if term.strip())) or "伤寒论 方证"
    differential = " ".join(dict.fromkeys(channels + positive + negative + ["鉴别 方证 经典条文"]))
    rescue = " ".join(dict.fromkeys(positive + negative + focus + ["方证 经典条文"]))
    return RetrievalQueryPlanV2(
        channel_hypotheses=channels,
        positive_facts=positive,
        explicit_negative_facts=negative,
        raw_synonym_phrases=focus,
        primary_query=primary,
        differential_query=differential,
        rescue_query=rescue,
        filter_mode="soft_channel",
        positive_fact_ids=[item.fact_id for item in current_positive],
        negative_fact_ids=[item.fact_id for item in current_negative],
        treatment_after_fact_ids=[item.fact_id for item in after_treatment],
        excluded_temporal_fact_ids=excluded_temporal,
        primary_channel_indices=[0] if channels else [],
        secondary_channel_indices=list(range(1, len(channels))),
        primary_focus_indices=[0] if focus else [],
        secondary_focus_indices=list(range(1, len(focus))),
    )


def _evidence_ref(item: Evidence) -> EvidenceRefV2:
    """把来源对象投影为 S3 可公开的引用。"""
    metadata = item.metadata
    return EvidenceRefV2(
        evidence_id=item.evidence_id,
        knowledge_type=item.knowledge_type,
        knowledge_version=item.knowledge_version,
        clause_no=(
            str(metadata.get("clause_no")) if metadata.get("clause_no") is not None else None
        ),
        formula_name=(str(metadata.get("formula_name")) if metadata.get("formula_name") else None),
        excerpt=item.content[:1200],
        review_status=str(metadata.get("review_status", "unknown")),
        retrieval_score=item.retrieval_score,
        rerank_score=item.rerank_score,
    )


async def retrieve_formula_candidates(
    knowledge: KnowledgeService,
    plan: RetrievalQueryPlanV2,
    *,
    max_candidates: int = 5,
    recall_count: int = 20,
) -> RetrievalBundleV2:
    """分别检索主查询、对比查询与受控恢复查询，按真实方名去重。"""
    if not 1 <= max_candidates <= 5:
        raise ValueError("方证候选数必须在 1 至 5 之间")
    trace: dict[str, object] = {
        "query_variant": plan.query_variant,
        "channel_filter_applied": False,
        "rescue_status": "not_executed",
    }

    # soft_channel is an explicit open candidate mode: S2 channels remain query guidance,
    # but cannot irreversibly remove cross-channel evidence before ranking.
    async def search_formula_with_trace(
        query: str, channels: list[str] | None = None
    ) -> list[Evidence]:
        """向新版知识服务传递追踪；兼容旧的只读测试端口。"""
        try:
            return await knowledge.search_formula(
                query, channels=channels, top_k=recall_count, trace=trace
            )
        except TypeError as exc:
            if "trace" not in str(exc):
                raise
            return await knowledge.search_formula(query, channels=channels, top_k=recall_count)

    formula_hits: list[Evidence] = await search_formula_with_trace(
        plan.primary_query,
        channels=None if plan.filter_mode == "soft_channel" else plan.channel_hypotheses,
    )
    if len(formula_hits) < 10 and plan.rescue_query:
        trace["rescue_status"] = "executed"
        rescue_plan = plan.model_copy(update={"filter_mode": "open_recovery"})
        assert rescue_plan.rescue_query is not None
        formula_hits += await search_formula_with_trace(rescue_plan.rescue_query)
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in formula_hits:
        name = str(item.metadata.get("formula_name", "")).strip()
        if not name:
            continue
        if all(existing.evidence_id != item.evidence_id for existing in grouped[name]):
            grouped[name].append(item)
    candidates: list[FormulaEvidenceCandidateV2] = []
    # 保留检索/精排返回顺序，不能按方名字典序截断相关候选。
    for name, items in list(grouped.items())[:max_candidates]:
        sources = items[:3]
        provenance: Literal["classic", "other_text", "historical_case"] = (
            "historical_case"
            if any(
                item.metadata.get("association_scope") == "case_derived_historical"
                for item in sources
            )
            else "classic"
        )
        candidates.append(
            FormulaEvidenceCandidateV2(
                candidate_id="formula-" + hashlib.sha256(name.encode()).hexdigest()[:16],
                formula_name=name,
                six_channel_tags=list(
                    dict.fromkeys(
                        channel
                        for item in sources
                        for channel in cast(
                            list[object],
                            item.metadata.get(
                                (
                                    "canonical_six_channels"
                                    if item.metadata.get("association_scope")
                                    == "classic_clause_based"
                                    else "case_derived_six_channels"
                                ),
                                item.metadata.get("canonical_six_channels", []),
                            ),
                        )
                        if isinstance(channel, str)
                    )
                ),
                source_evidence=[_evidence_ref(item) for item in sources],
                provenance=provenance,
                coverage_notes=["同一方证的来源按 Evidence ID 去重；检索分仅供诊断"],
            )
        )
    evidence_ids = list(dict.fromkeys(item.evidence_id for item in formula_hits))
    status: Literal["ok", "partial", "no_relevant_evidence"] = (
        "ok" if candidates else "no_relevant_evidence"
    )
    if formula_hits and not candidates:
        status = "partial"
    return RetrievalBundleV2(
        query_plan=plan,
        formula_candidates=candidates,
        retrieved_evidence_ids=evidence_ids,
        retrieval_trace={
            "formula_hits": len(formula_hits),
            "distinct_formula_count": len(grouped),
            **trace,
        },
        retrieval_status=status,
    )
