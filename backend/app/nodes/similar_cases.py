"""后置医案核验：依赖 KnowledgeService 精确来源与当前事实，不参与 canonical 归属或处方。

只从初诊来源核对 grounded 症状；随访疗效、历史现代病名和剂量不作为当前诊断支持。
保留召回不相关和冲突的审计，不把向量/Reranker 分数当成医学支持。
"""

from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.nodes.source_conditions import assess_source_condition
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import ReasoningPath, SimilarCaseAssessment


def assess_similar_case(
    source: Evidence, clinical: ClinicalEvidence, paths: list[ReasoningPath]
) -> SimilarCaseAssessment:
    """初诊症状需连续对齐并用初诊局部极性纠正元数据，避免把无汗标签解释成汗出。"""
    raw = source.metadata.get("symptoms_source_grounded", [])
    initial = source.metadata.get("initial_visit", "")
    channels = source.metadata.get("six_channels", [])
    path_ids = [
        path.path_id
        for path in paths
        if isinstance(channels, list) and set(path.six_channels).intersection(channels)
    ]
    result = SimilarCaseAssessment(
        evidence_id=source.evidence_id,
        path_ids=path_ids,
        title=str(source.metadata.get("case_title", source.knowledge_id)),
        status="insufficient",
        reason="仅作 L5 历史参考，不改变经典归属、排名或处方；需医生核对",
    )
    if source.knowledge_type != "medical_case" or source.source_level != "L5" or not path_ids:
        result.status = "irrelevant"
        return result
    terms = (
        list(dict.fromkeys(term for term in raw if isinstance(term, str)))
        if isinstance(raw, list)
        else []
    )
    initial_facts = parse_clinical_text(initial).findings if isinstance(initial, str) else []
    for term in terms:
        if not isinstance(initial, str) or term not in initial:
            result.missing_conditions.append(term)
            continue
        normalized = assess_source_condition(term, ClinicalEvidence())
        keys = {key for branch in normalized.alternatives for key in branch}
        # 只有明确一个事实的完整来源词项允许绑定初诊极性，复杂说明保留未知。
        bound = [
            fact
            for fact in initial_facts
            if fact.key in keys and fact.source_text and term in fact.source_text
        ]
        requirement = term
        if len(keys) == 1 and bound and len({fact.polarity for fact in bound}) == 1:
            phrase = bound[0].source_text
            if phrase:
                proposed = assess_source_condition(phrase, ClinicalEvidence())
                if (
                    proposed.alternatives
                    and {key for branch in proposed.alternatives for key in branch} == keys
                ):
                    requirement = phrase
        condition = assess_source_condition(requirement, clinical)
        result.condition_assessments.append(condition)
        target = (
            result.matching_conditions
            if condition.status == "supports"
            else (
                result.conflicting_conditions
                if condition.status == "contradicts"
                else result.missing_conditions
            )
        )
        if requirement not in target:
            target.append(requirement)
    total = len(
        set(result.matching_conditions + result.conflicting_conditions + result.missing_conditions)
    )
    result.similarity_score = len(result.matching_conditions) / max(1, total)
    result.status = (
        "conflicting"
        if result.conflicting_conditions
        else "reference" if result.matching_conditions else "insufficient"
    )
    return result


async def retrieve_similar_cases(
    clinical: ClinicalEvidence, paths: list[ReasoningPath], knowledge: KnowledgeService
) -> tuple[list[SimilarCaseAssessment], list[Evidence]]:
    """路径生成后一次有限检索；历史结果只附着到来源目录，不回流候选和评分。"""
    channels = list(dict.fromkeys(channel for path in paths for channel in path.six_channels))
    if not channels:
        return [], []
    query = " ".join(
        fact.source_text or fact.key
        for fact in clinical.current_findings()
        if fact.polarity != "unknown"
    )
    if not query:
        return [], []
    recalled = await knowledge.search_case(query, channels=channels, top_k=5)
    exact = {
        item.evidence_id: item
        for item in await knowledge.get_evidence_by_ids([item.evidence_id for item in recalled])
    }
    sources = [
        item.model_copy(
            update={
                "content": exact[item.evidence_id].content,
                "metadata": exact[item.evidence_id].metadata,
            }
        )
        for item in recalled
    ]
    audits = [assess_similar_case(item, clinical, paths) for item in sources]
    for path in paths:
        path.case_evidence = [
            item.evidence_id
            for item in audits
            if path.path_id in item.path_ids and item.status == "reference"
        ]
    return audits, sources
