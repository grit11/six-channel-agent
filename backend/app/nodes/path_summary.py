"""可审核路径摘要：来源条件覆盖、次选差异和候选资料治法。

依赖已核验 Evidence 与 KnowledgeService，不直接访问知识文件或模型。
完整度不改变 RP-13 排序；治法只摘录唯一、简短的来源字段，歧义不猜测，不输出剂量。
"""

import json
import re

from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import (
    EvidenceGap,
    PathCompleteness,
    PathConditionCoverage,
    ReasoningPath,
    TreatmentPrincipleSummary,
)


def summarize_completeness(
    path: ReasoningPath,
    sources: list[Evidence],
    gaps: list[EvidenceGap],
    missing_timeline: list[str],
) -> PathCompleteness:
    """按来源完整条件去重计算已知比例；反证是已知，Unknown 留在分母。"""
    by_id = {source.evidence_id: source for source in sources}
    units: dict[str, PathConditionCoverage] = {}
    for assessment in path.evidence_assessments:
        source = by_id.get(assessment.evidence_id)
        if (
            assessment.grade == "irrelevant"
            or source is None
            or source.source_level == "L5"
            or source.metadata.get("association_scope") == "case_derived_historical"
        ):
            continue
        for condition in assessment.condition_assessments:
            branches = sorted(
                {json.dumps(branch, sort_keys=True) for branch in condition.alternatives}
            )
            identity = json.dumps(
                [condition.kind, branches or condition.source_text], ensure_ascii=False
            )
            keys = {key for branch in condition.alternatives for key in branch}
            unavailable = condition.status == "unknown" and any(
                gap.status == "unavailable"
                and gap.finding_keys
                and set(gap.finding_keys).issubset(keys)
                for gap in gaps
            )
            old = units.get(identity)
            if old:
                old.evidence_ids = sorted(set(old.evidence_ids + [assessment.evidence_id]))
                # 同一规范化条件出现互相矛盾的核验快照时保守退回 unknown。
                if old.status != condition.status:
                    old.status = "unknown"
                old.unavailable = old.unavailable or unavailable
            else:
                units[identity] = PathConditionCoverage(
                    source_text=condition.source_text,
                    kind=condition.kind,
                    status=condition.status,
                    evidence_ids=[assessment.evidence_id],
                    unavailable=unavailable,
                )
    conditions = sorted(units.values(), key=lambda item: (item.kind, item.source_text))
    return PathCompleteness(
        supported_count=sum(item.status == "supports" for item in conditions),
        contradicted_count=sum(item.status == "contradicts" for item in conditions),
        unknown_count=sum(item.status == "unknown" for item in conditions),
        total_count=len(conditions),
        conditions=conditions,
        missing_timeline=list(dict.fromkeys(missing_timeline)),
    )


def summarize_treatment(path: ReasoningPath, sources: list[Evidence]) -> TreatmentPrincipleSummary:
    """仅摘录可归属候选的唯一简短治法；多条聚合内容不择一、不按排名补全。"""
    if path.association_scope == "case_derived_historical":
        return TreatmentPrincipleSummary(
            status="historical", reason="历史派生资料不升级为当前经典治法摘要。"
        )
    if path.candidate_status == "excluded":
        return TreatmentPrincipleSummary(
            status="insufficient_support", reason="该路径已被来源条件排除，治法摘要暂不展示。"
        )
    related = [
        source
        for source in sources
        if source.evidence_id in path.formula_evidence
        and source.metadata.get("formula_name") == path.formula_pattern
        and source.metadata.get("association_scope") == "classic_clause_based"
    ]
    methods: dict[str, set[str]] = {}
    unsafe = False
    for source in related:
        values = source.metadata.get("treatment_method")
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip()
            # 剂量、频率、方名、治疗分支和长说明不进入摘要，原始资料仍可展开审核。
            if (
                len(text) > 40
                or re.search(
                    r"[0-9０-９]|[一二三四五六七八九十半]+(?:两|克|钱|升|合|剂|次|日)|服|方用|宜|汤|丸|散剂|若|先|后|【|：|;|；",
                    text,
                )
                or text not in source.content
            ):
                unsafe = True
                continue
            methods.setdefault(text, set()).add(source.evidence_id)
    if unsafe or len(methods) > 1:
        return TreatmentPrincipleSummary(
            status="ambiguous",
            evidence_ids=[source.evidence_id for source in related],
            reason="来源治法包含多分支、聚合歧义或不可安全摘要内容，等待来源升级及医生核对。",
        )
    if not methods:
        return TreatmentPrincipleSummary(
            status="no_source", reason="没有可明确归属的简短来源治法。"
        )
    supported = any(
        item.evidence_id in path.formula_evidence
        and item.grade in {"direct", "indirect"}
        and item.supporting_requirements
        and not item.contradicting_requirements
        for item in path.evidence_assessments
    )
    if not supported:
        return TreatmentPrincipleSummary(
            status="insufficient_support",
            evidence_ids=[source.evidence_id for source in related],
            reason="方证来源尚无当前支持或存在反证，治法摘要暂不展示。",
        )
    text = next(iter(methods))
    return TreatmentPrincipleSummary(
        status="available",
        text=text,
        evidence_ids=sorted(methods[text]),
        reason=(
            "摘录候选资料 treatment_method 的唯一短句；来源尚待专家复核，"
            "须医生审核，不构成治疗或处方指令。"
        ),
    )
