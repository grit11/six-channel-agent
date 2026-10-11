"""来源约束的补证排序和停止判断。

依赖 KnowledgeService 精确核验规则，复用三值解析；只提出可观察的来源问题。
参数是可配置工程启发式，不是信息熵或临床概率；不可执行说明保留审计，不触发提问。
"""

import hashlib
import json
from typing import Literal

from backend.app.knowledge.differential import candidate_labels_match, candidate_source_values
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.source_conditions import assess_source_condition
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import DifferentialRule
from backend.app.schemas.reasoning import (
    DifferentialDimension,
    DifferentialResult,
    EvidenceGap,
    EvidenceGapContext,
    EvidenceGapDecision,
    EvidenceGapPolicy,
    ReasoningPath,
)


def _signature(alternatives: list[dict[str, str]]) -> str:
    """规范化分支身份，避免顺序或重复分支伪造区分度。"""
    return "|".join(sorted({json.dumps(item, sort_keys=True) for item in alternatives}))


def _build_gap(
    dimension: DifferentialDimension,
    rule: DifferentialRule | None,
    evidence: ClinicalEvidence,
    context: EvidenceGapContext,
    policy: EvidenceGapPolicy,
    total_candidates: int,
    candidate_ids: list[str],
) -> EvidenceGap | None:
    """用精确来源重新核对条件；部分 OR 未决仍可提问，共同条件不提问。"""
    description = rule.key_dimensions.get(dimension.dimension) if rule else None
    values = (
        candidate_source_values(description, rule.candidates, dimension=dimension.dimension)
        if rule and description
        else {}
    )
    bound = dimension.candidate_values
    verified = (
        len(bound) >= 2
        and all(values.get(label) == value for label, value in bound.items())
        and all(
            any(candidate_labels_match(candidate, label) for label in bound)
            for candidate in candidate_ids
        )
    )
    conditions = [
        assess_source_condition(value, evidence, dimension=dimension.dimension)
        for value in bound.values()
    ]
    if verified and conditions and all(item.status != "unknown" for item in conditions):
        return None
    executable = verified and all(
        item.kind == "symptom" and item.alternatives for item in conditions
    )
    signatures = {_signature(item.alternatives) for item in conditions}
    distinctive = executable and len(signatures) > 1
    known = {item.key for item in evidence.current_findings() if item.polarity != "unknown"}
    current = {
        item.key: item.polarity
        for item in evidence.current_findings()
        if item.polarity != "unknown"
    }
    keys = sorted(
        {
            key
            for item in conditions
            if item.status == "unknown"
            for branch in item.alternatives
            if not any(key in current and current[key] != value for key, value in branch.items())
            for key in branch
        }
        - known
    )
    identity = (
        "|".join(
            sorted({key for item in conditions for branch in item.alternatives for key in branch})
        )
        if distinctive and keys
        else f"{dimension.source_rule_id}|{dimension.dimension}"
    )
    question_id = "gap-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    reviewed = bool(
        rule and str(rule.metadata.get("review_status", "")).startswith("source_reviewed")
    )
    reliability = (
        policy.reviewed_source_reliability if reviewed else policy.unreviewed_source_reliability
    )
    factors = {
        "discriminative_power": dimension.discriminator_weight if distinctive else 0.0,
        "clinical_answerability": policy.answerability if distinctive and keys else 0.0,
        "source_reliability": reliability if verified else 0.0,
        "candidate_coverage": len(bound) / max(total_candidates, len(bound), 1),
        "repetition_penalty": min(
            1.0, context.asked_counts.get(question_id, 0) * policy.repetition_penalty
        ),
        "question_cost": policy.question_cost,
    }
    priority = max(
        0.0,
        factors["discriminative_power"]
        * factors["clinical_answerability"]
        * factors["source_reliability"]
        * factors["candidate_coverage"]
        - factors["repetition_penalty"]
        - factors["question_cost"],
    )
    status: Literal["pending", "unavailable", "deferred"] = "pending"
    reason = None
    if not distinctive or not keys:
        status, reason = (
            "deferred",
            "来源未完整归属、不可执行或属于共同条件；保留未知，待来源升级。",
        )
    elif question_id in context.unavailable_question_ids:
        status, reason = "unavailable", "医生明确当前无法获取；仍为未知，未计为反证。"
    elif priority < policy.minimum_priority:
        status, reason = "deferred", "当前提问优先级低于阈值；保留未解决条件。"
    requirements = "；".join(f"{label}：{value}" for label, value in bound.items())
    return EvidenceGap(
        question=(
            f"请核对{dimension.dimension}（{requirements}）。"
            if distinctive
            else f"待核对来源维度：{dimension.dimension}。"
        ),
        dimension=dimension.dimension,
        priority=priority,
        source_rule_ids=[dimension.source_rule_id],
        candidate_coverage=max(1, len(bound)),
        question_id=question_id,
        finding_keys=keys,
        status=status,
        unresolved_reason=reason,
        priority_factors=factors,
    )


async def build_evidence_gaps(
    results: list[DifferentialResult],
    evidence: ClinicalEvidence,
    knowledge: KnowledgeService,
    *,
    context: EvidenceGapContext | None = None,
    policy: EvidenceGapPolicy | None = None,
) -> list[EvidenceGap]:
    """精确核验后排序全部缺口；合并同事实问题，不丢弃低价值或不可获得审计。"""
    active_context, active_policy = context or EvidenceGapContext(), policy or EvidenceGapPolicy()
    total = len({candidate for result in results for candidate in result.candidate_ids})
    rules = {
        rule_id: await knowledge.get_differential_by_id(rule_id)
        for rule_id in sorted(
            {item.source_rule_id for result in results for item in result.dimensions}
        )
    }
    grouped: dict[str, EvidenceGap] = {}
    covered: dict[str, set[str]] = {}
    for result in results:
        for dimension in result.dimensions:
            gap = _build_gap(
                dimension,
                rules[dimension.source_rule_id],
                evidence,
                active_context,
                active_policy,
                total,
                result.candidate_ids,
            )
            if gap is None:
                continue
            covered.setdefault(gap.question_id, set()).update(result.candidate_ids)
            previous = grouped.get(gap.question_id)
            if previous is not None:
                rule_ids = sorted(set(previous.source_rule_ids + gap.source_rule_ids))
                gap = max((previous, gap), key=lambda item: item.priority).model_copy(
                    update={"source_rule_ids": rule_ids}
                )
            grouped[gap.question_id] = gap
    for identity, gap in grouped.items():
        count = len(covered[identity])
        factors = dict(gap.priority_factors)
        factors["candidate_coverage"] = count / max(total, count, 1)
        priority = max(
            0.0,
            factors["discriminative_power"]
            * factors["clinical_answerability"]
            * factors["source_reliability"]
            * factors["candidate_coverage"]
            - factors["repetition_penalty"]
            - factors["question_cost"],
        )
        status, reason = gap.status, gap.unresolved_reason
        if (
            factors["discriminative_power"]
            and factors["clinical_answerability"]
            and status != "unavailable"
        ):
            status = "pending" if priority >= active_policy.minimum_priority else "deferred"
            reason = None if status == "pending" else "当前提问优先级低于阈值；保留未解决条件。"
        grouped[identity] = gap.model_copy(
            update={
                "candidate_coverage": count,
                "priority_factors": factors,
                "priority": priority,
                "status": status,
                "unresolved_reason": reason,
            }
        )
    return sorted(
        grouped.values(),
        key=lambda item: (item.status != "pending", -item.priority, item.question_id),
    )


def decide_evidence_gap_stop(
    gaps: list[EvidenceGap],
    results: list[DifferentialResult],
    paths: list[ReasoningPath],
    *,
    supplementation_round: int,
    max_rounds: int,
    policy: EvidenceGapPolicy | None = None,
    doctor_current_information: bool = False,
) -> EvidenceGapDecision:
    """停止不等于证据齐全；高权重未决或来源未核验不能仅凭分差停止。"""
    active_policy = policy or EvidenceGapPolicy()
    high = [
        dimension
        for result in results
        for dimension in result.dimensions
        if dimension.discriminator_weight >= active_policy.high_weight_threshold
    ]
    # 至少存在经核验的高权重维度，且 Top2 引用的全部未决条件均已检查，才允许分差停止。
    checked = bool(high) and all(
        item.condition_assessments
        and all(condition.status != "unknown" for condition in item.condition_assessments.values())
        for item in high
    )
    checked = checked and not any(
        gap.source_rule_ids
        and gap.dimension == item.dimension
        and item.source_rule_id in gap.source_rule_ids
        for gap in gaps
        for item in high
    )
    checked = (
        checked
        and len(paths) >= 2
        and all(
            path.evidence_assessments
            and all(not item.unresolved_requirements for item in path.evidence_assessments)
            for path in paths[:2]
        )
    )
    separation = (
        max(0.0, paths[0].ranking_score - paths[1].ranking_score) if len(paths) >= 2 else None
    )
    actionable = sum(item.status == "pending" for item in gaps)
    reason: Literal[
        "continue_supplement",
        "score_separation",
        "no_actionable_gaps",
        "all_unavailable",
        "max_rounds",
        "doctor_current_information",
    ] = "continue_supplement"
    if doctor_current_information:
        reason = "doctor_current_information"
    elif supplementation_round >= max_rounds:
        reason = "max_rounds"
    elif checked and separation is not None and separation >= active_policy.separation_threshold:
        reason = "score_separation"
    elif not actionable:
        reason = (
            "all_unavailable"
            if any(item.status == "unavailable" for item in gaps)
            else "no_actionable_gaps"
        )
    return EvidenceGapDecision(
        stop=reason != "continue_supplement",
        reason=reason,
        actionable_count=actionable,
        high_weight_counterevidence_checked=checked,
        separation=separation,
        policy=active_policy,
    )
