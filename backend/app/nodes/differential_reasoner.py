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
from backend.app.schemas.reasoning import (
    DifferentialDimension,
    DifferentialResult,
    FormulaCandidate,
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
    unique = list({item.formula_name: item for item in formulas[:8]}.values())
    resolved = await knowledge.resolve_differential_candidates(
        [item.formula_name for item in unique]
    )
    for first, second in combinations(unique, 2):
        labels = [resolved[first.formula_name], resolved[second.formula_name]]
        if labels[0] == labels[1]:
            continue
        rules = await knowledge.search_differential(labels, query=" vs ".join(labels), top_k=5)
        for rule in rules:
            bound = [source_candidate_label(label, rule.candidates) for label in labels]
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
