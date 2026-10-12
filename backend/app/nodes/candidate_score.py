"""RP-13 路径评分与来源约束排除门控。

职责：对已精确核验的来源及当前事实计算公开因子和反证惩罚，保留可复算分项。
边界：不检索、不调用模型，不把召回数量或模型排除标签作为硬排除依据；
未标注核心/排除条件的知识只能参与普通反证核对，不自行创造医学排除规则。
"""

from backend.app.knowledge.differential import candidate_source_values, source_candidate_label
from backend.app.nodes.clinical_parser import match_source_phrase
from backend.app.nodes.source_conditions import assess_source_condition
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence, EvidenceAssessment
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    DifferentialResult,
    ReasoningPath,
    ScoringPolicy,
    SyndromeRelationship,
)

_FACTORS = ("core", "differential", "classic", "formula", "temporal", "case", "context")


def _support_ratio(assessments: list[EvidenceAssessment]) -> float:
    """按唯一条件计算已知正证比例；重复来源不加分，未知仅进入完整度。"""
    support = {term for item in assessments for term in item.supporting_requirements}
    contradict = {term for item in assessments for term in item.contradicting_requirements}
    return len(support) / max(1, len(support | contradict))


def _exclusions(
    path: ReasoningPath, clinical: ClinicalEvidence, sources: list[Evidence], policy: ScoringPolicy
) -> list[str]:
    """仅执行来源明确标注且原文连续对齐的核心或高权重排除条件。"""
    reasons: list[str] = []
    for source in sources:
        if source.metadata.get("exclusion_review_status") != "expert_approved":
            continue
        if source.evidence_id not in {
            a.evidence_id for a in path.evidence_assessments if a.grade != "irrelevant"
        }:
            continue
        # 字段是可审核知识扩展契约；当前 V1.2 无标注时保持空，不能从普通阴性字段推导。
        if source.knowledge_type == "classic_clause":
            related = source.metadata.get("related_formulas", [])
            if not isinstance(related, list) or path.formula_pattern not in related:
                continue
            conditions = source.metadata.get("core_conditions_source_grounded", [])
            if isinstance(conditions, list):
                for condition in conditions:
                    if (
                        isinstance(condition, str)
                        and condition in source.content
                        and match_source_phrase(condition, clinical) == "contradicts"
                    ):
                        reasons.append(f"{source.evidence_id}：核心条件冲突：{condition}")
        elif source.knowledge_type == "differential":
            conditions = source.metadata.get("exclusion_conditions_source_grounded", [])
            if not isinstance(conditions, list):
                continue
            for entry in conditions:
                if not isinstance(entry, dict):
                    continue
                condition, weight = entry.get("condition"), entry.get("weight")
                label = (path.formula_pattern or "").removesuffix("证") + "证"
                if (
                    entry.get("candidate") == label
                    and isinstance(condition, str)
                    and condition in source.content
                    and isinstance(weight, (float, int))
                    and not isinstance(weight, bool)
                    and policy.high_threshold <= weight <= 1
                    and match_source_phrase(condition, clinical) == "contradicts"
                ):
                    reasons.append(f"{source.evidence_id}：高权重条件冲突：{condition}")
    return list(dict.fromkeys(reasons))


def score_path(
    path: ReasoningPath,
    clinical: ClinicalEvidence,
    group: list[ChannelCandidate],
    relation: SyndromeRelationship | None,
    differential: list[DifferentialResult],
    sources: list[Evidence],
    policy: ScoringPolicy,
) -> None:
    """核验后计算七因子与独立惩罚；每项缺失为零，Unknown 不增加任何反证。"""
    by_id = {item.evidence_id: item for item in sources}
    usable = [item for item in path.evidence_assessments if item.grade != "irrelevant"]
    by_kind = {
        kind: [item for item in usable if by_id[item.evidence_id].knowledge_type == kind]
        for kind in ("classic_clause", "formula_pattern", "differential", "medical_case")
    }
    core = sum(
        item.pass_a_score if item.pass_a_score is not None else item.independent_score
        for item in group
    ) / len(group)
    high_terms: set[str] = set()
    diff_support: dict[tuple[str, str], float] = {}
    for result in differential:
        if path.formula_pattern not in result.candidate_ids:
            continue
        for dimension in result.dimensions:
            rule = next(
                (source for source in sources if source.knowledge_id == dimension.source_rule_id),
                None,
            )
            if rule is None or rule.evidence_id not in path.differential_evidence:
                continue
            label = source_candidate_label(
                path.formula_pattern or "", list(dimension.candidate_values)
            )
            condition = dimension.candidate_values.get(label or "")
            if not condition:
                continue
            # 来源真实存在仍不等于分支真实归属；候选值须与规则原维度拆分结果一致。
            raw_dimensions = rule.metadata.get("key_dimensions")
            labels = rule.metadata.get("candidates")
            if not isinstance(raw_dimensions, dict) or not isinstance(labels, list):
                continue
            description = raw_dimensions.get(dimension.dimension)
            if not isinstance(description, str):
                continue
            # 鉴别得分须完整绑定当前双方；单方说明和双方共同支持只能作背景，
            # 不能凭一方症状命中制造区分性分数。
            bound = [source_candidate_label(name, labels) for name in result.candidate_ids]
            original_values = candidate_source_values(
                description, labels, dimension=dimension.dimension
            )
            if (
                len(bound) != 2
                or None in bound
                or len(set(bound)) != 2
                or any(candidate not in original_values for candidate in bound)
                or any(
                    dimension.candidate_values.get(candidate or "")
                    != original_values.get(candidate or "")
                    for candidate in bound
                )
            ):
                continue
            statuses = {
                candidate: assess_source_condition(
                    original_values[candidate], clinical, dimension=dimension.dimension
                ).status
                for candidate in bound
                if candidate is not None
            }
            status = assess_source_condition(
                condition, clinical, dimension=dimension.dimension
            ).status
            if status == "supports" and list(statuses.values()).count("supports") == 1:
                diff_support[(dimension.source_rule_id, condition)] = dimension.discriminator_weight
            elif (
                status == "contradicts" and dimension.discriminator_weight >= policy.high_threshold
            ):
                high_terms.add(condition)
    ordinary = {term for item in usable for term in item.contradicting_requirements} - high_terms
    # 来源条件去重后再罚分，不采用模型自报反证或未校验上游文本。
    factors = {
        "core": core,
        "differential": max(diff_support.values(), default=0.0),
        "classic": _support_ratio(by_kind["classic_clause"]),
        "formula": _support_ratio(by_kind["formula_pattern"]),
        "temporal": float(
            bool(
                relation
                and relation.relationship_type != "uncertain"
                and relation.temporal_evidence
                and relation.supporting_evidence
                and not relation.missing_timeline_evidence
                and not relation.contradicting_evidence
            )
        ),
        "case": _support_ratio(by_kind["medical_case"]),
        "context": 0.0,  # 没有独立上下文核验器时明确保持零，避免重复八纲事实加分。
    }
    components = {f"{name}_factor": factors[name] for name in _FACTORS}
    components.update({f"{name}_weight": getattr(policy, name) for name in _FACTORS})
    positive = sum(factors[name] * getattr(policy, name) for name in _FACTORS)
    high = min(1.0, len(high_terms) * policy.high_penalty)
    medium = min(1.0, len(ordinary) * policy.medium_penalty)
    score = max(0.0, min(1.0, positive - high - medium))
    components.update(
        positive_score=positive,
        high_contradiction_count=float(len(high_terms)),
        medium_contradiction_count=float(len(ordinary)),
        high_penalty_weight=policy.high_penalty,
        medium_penalty_weight=policy.medium_penalty,
        high_penalty=high,
        medium_penalty=medium,
        final_score=score,
    )
    path.ranking_score, path.score_components = score, components
    path.exclusion_reasons = _exclusions(path, clinical, sources, policy)
    path.candidate_status = (
        "excluded"
        if path.exclusion_reasons
        else "weak" if any(item.status == "weak" for item in group) else "active"
    )
    if path.exclusion_reasons:
        path.warnings.append("来源明确排除条件已触发；保留审计，不进入可采用 Top-K")
