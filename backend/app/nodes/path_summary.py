"""可审核路径摘要：来源条件覆盖、次选差异和候选资料治法。

依赖已核验 Evidence 与 KnowledgeService，不直接访问知识文件或模型。
完整度不改变 RP-13 排序；治法只摘录唯一、简短的来源字段，歧义不猜测，不输出剂量。
"""

import json
import re
from typing import Literal

from backend.app.knowledge.differential import candidate_source_values, source_candidate_label
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.source_conditions import assess_source_condition
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import (
    DifferentialResult,
    EvidenceGap,
    PathComparison,
    PathCompleteness,
    PathConditionCoverage,
    PathKeyDifference,
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


def _conclusion(
    first: str, second: str
) -> Literal["favors_current", "favors_next", "shared", "unresolved", "both_contradicted"]:
    """按三值状态给出公开比较结论，单边 Unknown 不等同反证。"""
    if first == "supports" and second == "contradicts":
        return "favors_current"
    if first == "contradicts" and second == "supports":
        return "favors_next"
    if first == second == "supports":
        return "shared"
    if first == second == "contradicts":
        return "both_contradicted"
    return "unresolved"


async def compare_next_path(
    path: ReasoningPath,
    following: ReasoningPath,
    clinical: ClinicalEvidence,
    differential: list[DifferentialResult],
    knowledge: KnowledgeService,
) -> tuple[PathComparison, list[Evidence]]:
    """比较完整排序中的次选，重新读来源，不信任上游组级支持摘要或观察文本。"""
    comparison = PathComparison(
        next_path_id=following.path_id,
        next_syndrome_name=following.syndrome_name,
        next_formula_pattern=following.formula_pattern,
        status="no_source_rule",
        summary="当前候选对没有完整可归属的来源鉴别；分差仅为工程排序差异。",
    )
    if not path.formula_pattern or not following.formula_pattern:
        comparison.status, comparison.summary = (
            "no_formula",
            "至少一条路径没有方证，当前未建立来源支持的候选对差异。",
        )
        return comparison, []
    if path.formula_pattern == following.formula_pattern:
        comparison.status, comparison.summary = (
            "same_formula",
            "两条路径引用同一方证；六经或关系差异须另行核对，不能套用方证鉴别。",
        )
        return comparison, []
    rule_ids = {
        dimension.source_rule_id
        for result in differential
        if {path.formula_pattern, following.formula_pattern}.issubset(result.candidate_ids)
        for dimension in result.dimensions
    }
    sources: list[Evidence] = []
    for rule_id in sorted(rule_ids):
        rule = await knowledge.get_differential_by_id(rule_id)
        if rule is None:
            continue
        first, second = (
            source_candidate_label(name, rule.candidates)
            for name in (path.formula_pattern, following.formula_pattern)
        )
        if first is None or second is None or first == second:
            continue
        for dimension, description in rule.key_dimensions.items():
            values = candidate_source_values(description, rule.candidates, dimension=dimension)
            if first not in values or second not in values:
                continue
            left = assess_source_condition(values[first], clinical, dimension=dimension)
            right = assess_source_condition(values[second], clinical, dimension=dimension)
            # 治疗和病机说明不转成关键症状差异；保留在全量 DifferentialPanel。
            if not all(item.kind == "symptom" and item.alternatives for item in (left, right)):
                continue
            comparison.differences.append(
                PathKeyDifference(
                    dimension=dimension,
                    evidence_id=rule.evidence_id,
                    current_requirement=values[first],
                    next_requirement=values[second],
                    current_status=left.status,
                    next_status=right.status,
                    finding_keys=sorted(
                        {
                            key
                            for item in (left, right)
                            for branch in item.alternatives
                            for key in branch
                        }
                    ),
                    conclusion=_conclusion(left.status, right.status),
                )
            )
        if any(item.evidence_id == rule.evidence_id for item in comparison.differences):
            sources.append(rule)
    if comparison.differences:
        comparison.status = "compared"
        favored = [
            item.dimension for item in comparison.differences if item.conclusion == "favors_current"
        ]
        comparison.summary = (
            "当前已知条件更支持本路径的维度：" + "、".join(favored)
            if favored
            else "当前来源维度尚未形成明确支持本路径的完整差异；详见双方三值核对。"
        )
    return comparison, sources
