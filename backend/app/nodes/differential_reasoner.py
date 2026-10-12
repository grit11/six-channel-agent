"""候选特异鉴别节点：通过 KnowledgeService 读取规则并按原文归属拆分双方要求。

边界：只执行完整可识别的事实要求，未解析说明保留缺口；不把相关召回或共同表现
当作某一候选的独有支持，不创作医学规则。别名经唯一知识边界归一；长说明保留审计。
"""

from itertools import combinations
from typing import Literal

from backend.app.knowledge.differential import candidate_source_values, source_candidate_label
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import (
    source_phrase_observed_requirements,
)
from backend.app.nodes.source_conditions import assess_source_condition
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import (
    DifferentialDimension,
    DifferentialResult,
    FormulaCandidate,
    PathComparison,
    PathKeyDifference,
    ReasoningPath,
)


def _observe(
    values: dict[str, str], description: str, evidence: ClinicalEvidence
) -> tuple[str | None, str]:
    """列出来源要求涉及的当前明确事实；复合要求尚不完整时仍保留已知原文。"""
    phrases = [*values.values(), description]
    keys = {key for phrase in phrases for key in source_phrase_observed_requirements(phrase)}
    observed = [
        item
        for item in evidence.current_findings()
        if item.key in keys and item.polarity != "unknown"
    ]
    texts = list(dict.fromkeys(item.source_text or item.key for item in observed))
    polarities = {item.polarity for item in observed}
    polarity = (
        next(iter(polarities)) if len(polarities) == 1 else "mixed" if polarities else "unknown"
    )
    return "；".join(texts) or None, polarity


async def compare_formula_candidates(
    evidence: ClinicalEvidence, formulas: list[FormulaCandidate], knowledge: KnowledgeService
) -> list[DifferentialResult]:
    """只应用包含当前双方候选的规则，分别返回支持、反证和无法判定的维度。"""
    results: list[DifferentialResult] = []
    # S3 控制候选池上限；S4 仅去重，不能再次截断来源充分的后排候选。
    unique = list({item.formula_name: item for item in formulas}.values())
    rules = await knowledge.differential_rules_for_pool([item.formula_name for item in unique])
    for first, second in combinations(unique, 2):
        for rule in rules:
            # 按每条规则局部绑定，跨规则的同名/括注歧义不影响当前明确来源。
            bound = [
                source_candidate_label(item.formula_name, rule.candidates)
                for item in (first, second)
            ]
            if any(label is None for label in bound) or len(set(bound)) < 2:
                continue
            rule_labels = [label for label in bound if label is not None]
            dimensions: list[DifferentialDimension] = []
            decisive_support: list[str] = []
            decisive_contradictions: list[str] = []
            unresolved: list[str] = []
            shared: list[str] = []
            for dimension, description in rule.key_dimensions.items():
                all_values = candidate_source_values(
                    description, rule.candidates, dimension=dimension
                )
                values = {label: all_values[label] for label in rule_labels if label in all_values}
                observed, polarity = _observe(values, description, evidence)
                # 必须有双方来源值才能形成候选特异比较，单方说明不补齐另一方。
                # 复杂条件即使命中部分词项，也只能保留 observed，不升级为支持或反证。
                assessments = {
                    label: assess_source_condition(value, evidence, dimension=dimension)
                    for label, value in values.items()
                }
                statuses = {label: item.status for label, item in assessments.items()}
                supported = [label for label, status in statuses.items() if status == "supports"]
                contradicted = [
                    label for label, status in statuses.items() if status == "contradicts"
                ]
                status: Literal["supports", "contradicts", "unknown", "neutral"] = "unknown"
                if set(values) == set(rule_labels):
                    if len(supported) == 2:
                        status = "neutral"
                        shared.append(description)
                    elif supported:
                        status = "supports"
                        decisive_support.extend(f"{dimension}支持{label}" for label in supported)
                    elif contradicted:
                        status = "contradicts"
                    decisive_contradictions.extend(
                        f"{dimension}反证{label}" for label in contradicted
                    )
                if status == "unknown" or "unknown" in statuses.values():
                    unresolved.append(dimension)
                if dimension in {"共同表现", "共同点", "总纲"}:
                    shared.append(description)
                dimensions.append(
                    DifferentialDimension(
                        dimension=dimension,
                        candidate_values=values,
                        observed_value=observed,
                        observed_polarity=polarity,
                        discriminator_weight=(
                            0.8 if dimension in {"汗", "脉", "小便", "口渴"} else 0.4
                        ),
                        source_rule_id=rule.rule_id,
                        status=status,
                        condition_assessments=assessments,
                    )
                )
            results.append(
                DifferentialResult(
                    candidate_ids=[first.formula_name, second.formula_name],
                    shared_features=list(dict.fromkeys(shared)),
                    dimensions=dimensions,
                    decisive_support=decisive_support,
                    decisive_contradictions=decisive_contradictions,
                    unresolved_dimensions=list(dict.fromkeys(unresolved)),
                )
            )
    return results


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
    *,
    verified_only: bool = False,
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
    if verified_only:
        # S4 的条件核验已通过 KnowledgeService；S5 只消费，不再取同一条规则。
        for result in differential:
            if {path.formula_pattern, following.formula_pattern} != set(result.candidate_ids):
                continue
            for verified_dimension in result.dimensions:
                labels = list(verified_dimension.candidate_values)
                first = source_candidate_label(path.formula_pattern, labels)
                second = source_candidate_label(following.formula_pattern, labels)
                left = verified_dimension.condition_assessments.get(first or "")
                right = verified_dimension.condition_assessments.get(second or "")
                if (
                    not first
                    or not second
                    or first == second
                    or left is None
                    or right is None
                    or not all(
                        item.kind == "symptom" and item.alternatives for item in (left, right)
                    )
                ):
                    continue
                comparison.differences.append(
                    PathKeyDifference(
                        dimension=verified_dimension.dimension,
                        evidence_id=knowledge.version + ":" + verified_dimension.source_rule_id,
                        current_requirement=verified_dimension.candidate_values[first],
                        next_requirement=verified_dimension.candidate_values[second],
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
        if comparison.differences:
            comparison.status = "compared"
            comparison.summary = "来源鉴别维度已在 S4 核验；Unknown 保留待医生确认。"
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
