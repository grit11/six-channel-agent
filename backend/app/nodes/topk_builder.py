"""Top-K 路径节点：组装上游可审计结果，通过 KnowledgeService 核验来源。

边界：不新增医学事实、治法、剂量或医案；只摘要可归属的治法来源，不把召回相关性当支持。
对选定路径逐条核验引用用途，反证保留供审核，无关来源退出有效引用并留审计 ID。
当前排序使用 RP-13 可配置工程权重，供工程验证与医生核对，不替代最终诊断。
"""

import hashlib
import json

from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.knowledge.differential import source_candidate_label
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.candidate_score import score_path
from backend.app.nodes.path_summary import (
    compare_next_path,
    summarize_completeness,
    summarize_treatment,
)
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    DifferentialResult,
    EightPrinciplesState,
    EvidenceGap,
    FormulaCandidate,
    ReasoningPath,
    ScoringPolicy,
    SyndromeRelationship,
)


def _unique(values: list[str]) -> list[str]:
    """保留首次顺序并去重，避免重复引用或事实影响展示。"""
    return list(dict.fromkeys(values))


def _groups(
    candidates: list[ChannelCandidate], relationships: list[SyndromeRelationship]
) -> list[tuple[list[ChannelCandidate], SyndromeRelationship | None]]:
    """已核验关系各保留独立路径；历史经不是当前 ID，未定多候选仍保持独立假设。"""
    eligible = {item.candidate_id: item for item in candidates if item.status != "excluded"}
    groups: list[tuple[list[ChannelCandidate], SyndromeRelationship | None]] = []
    for relation in relationships:
        if relation.relationship_type == "uncertain":
            continue
        ids = _unique([relation.primary_candidate_id, *relation.secondary_candidate_ids])
        if ids and all(candidate_id in eligible for candidate_id in ids):
            groups.append(([eligible[candidate_id] for candidate_id in ids], relation))
    for candidate in eligible.values():
        uncertain = next(
            (
                item
                for item in relationships
                if item.relationship_type == "uncertain"
                and candidate.candidate_id
                in [item.primary_candidate_id, *item.secondary_candidate_ids]
            ),
            None,
        )
        groups.append(([candidate], uncertain))
    return groups


def _compatible(formula: FormulaCandidate, channels: list[str]) -> bool:
    """按各自来源归属关联方证；多经路径必须覆盖全部经，不能任意给组合关系配方。"""
    associated = (
        formula.canonical_six_channels
        if formula.association_scope == "classic_clause_based"
        else formula.case_derived_six_channels
    )
    return set(channels).issubset(associated)


async def _differential_refs(
    formula: FormulaCandidate | None,
    differential: list[DifferentialResult],
    knowledge: KnowledgeService,
) -> tuple[list[str], list[str]]:
    """只纳入完整包含当前比较对的规则；多候选规则按双方归属核验，不借无关规则。"""
    evidence_ids: list[str] = []
    missing: list[str] = []
    if formula is None:
        return evidence_ids, missing
    for result in differential:
        if formula.formula_name not in result.candidate_ids:
            continue
        for rule_id in _unique([dimension.source_rule_id for dimension in result.dimensions]):
            rule = await knowledge.get_differential_by_id(rule_id)
            labels = (
                [source_candidate_label(name, rule.candidates) for name in result.candidate_ids]
                if rule is not None
                else []
            )
            if (
                rule is not None
                and None not in labels
                and len(set(labels)) == len(result.candidate_ids)
            ):
                evidence_ids.append(rule.evidence_id)
                missing.extend(result.unresolved_dimensions)
    return _unique(evidence_ids), _unique(missing)


async def _make_path(
    group: list[ChannelCandidate],
    relation: SyndromeRelationship | None,
    formula: FormulaCandidate | None,
    clinical: ClinicalEvidence,
    principles: EightPrinciplesState,
    differential: list[DifferentialResult],
    knowledge: KnowledgeService,
    channel_weight: float,
) -> ReasoningPath:
    """合并上游事实与缺失信息，不根据条文数量加分，也不自动补全证型或治法。"""
    channels = _unique([channel for item in group for channel in item.six_channels])
    candidate_ids = [item.candidate_id for item in group]
    diff_ids, diff_missing = await _differential_refs(formula, differential, knowledge)
    relation_sources = (
        await knowledge.get_evidence_by_ids(relation.supporting_evidence) if relation else []
    )
    channel_score = sum(item.independent_score for item in group) / len(group)
    formula_score = formula.rank_score if formula else 0.0
    # 构造时只保留原始值；精确来源核验后统一按 RP-13 排序，旧权重仅兼容调用入口。
    score = (
        channel_weight * channel_score + (1 - channel_weight) * formula_score
        if formula
        else channel_score
    )
    identity = json.dumps(
        [
            sorted(candidate_ids),
            sorted(set(formula.evidence_ids)) if formula else [],
            relation.relationship_type if relation else None,
        ],
        ensure_ascii=False,
    )
    missing = [value for item in group for value in item.missing_key_findings]
    # 全量未知是临床事实库存，不等于每条路径都需要补问；路径只汇总其候选、
    # 方证、关系和鉴别的缺口。原始 unknown 留在 ClinicalEvidence 中供医生编辑。
    missing.extend(f"事实冲突待核对：{item.key}" for item in clinical.contradictions)
    if relation is not None and relation.relationship_type != "uncertain":
        missing.extend(relation.missing_timeline_evidence)
    missing.extend(diff_missing)
    warnings = ["工程规则基线，八纲、方证匹配和鉴别尚待专家校准", "来源召回不等于已验证的患者支持"]
    if formula and formula.association_scope == "case_derived_historical":
        warnings.append("方证六经关联来自历史医案，不属于经典归属")
    if relation and relation.relationship_type != "uncertain":
        warnings.append("关系由医生病史条件与来源定义核验，仍为待审核候选")
    if not formula:
        warnings.append("暂无可用的关联方证，保留六经候选供核对")
    return ReasoningPath(
        rank=1,
        path_id="path_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
        channel_candidate_ids=candidate_ids,
        six_channels=channels,
        syndrome_name=(
            f"{formula.formula_name}方证候选（待核验）"
            if formula
            else "、".join(item.provisional_syndrome or item.six_channels[0] for item in group)
        ),
        relationship=relation.relationship_type if relation else None,
        eight_principles_summary=principles.model_copy(deep=True),
        supporting_findings=_unique(
            [value for item in group for value in item.supporting_findings]
            + (formula.supporting_findings if formula else [])
        ),
        contradicting_findings=_unique(
            [value for item in group for value in item.contradicting_findings]
            + (formula.contradicting_findings if formula else [])
        ),
        # 方证的完整来源条件在后续适用性核验中分类；病机/治法和不可解析原文
        # 保留在完整度审计，不能整体当作可补问的患者症状。
        missing_key_findings=_unique(missing),
        formula_pattern=formula.formula_name if formula else None,
        association_scope=formula.association_scope if formula else None,
        framework_evidence=_unique(
            [value for item in group for value in item.framework_evidence_ids]
            + principles.framework_evidence_ids
            + [
                item.evidence_id
                for item in relation_sources
                if item.knowledge_type == "six_channel_framework"
            ]
        ),
        classic_evidence=_unique(
            [value for item in group for value in item.classic_evidence_ids]
            + [
                item.evidence_id
                for item in relation_sources
                if item.knowledge_type == "classic_clause"
            ]
        ),
        formula_evidence=formula.evidence_ids if formula else [],
        differential_evidence=diff_ids,
        ranking_score=score,
        score_components={
            "channel_score": channel_score,
            "formula_score": formula_score,
            "channel_weight": channel_weight if formula else 1.0,
            "formula_weight": 1 - channel_weight if formula else 0.0,
        },
        evidence_completeness=min(item.evidence_completeness for item in group),
        risk_level=formula.risk_level if formula else None,
        warnings=warnings,
    )


async def build_topk_paths(
    clinical: ClinicalEvidence,
    principles: EightPrinciplesState,
    candidates: list[ChannelCandidate],
    relationships: list[SyndromeRelationship],
    formulas: list[FormulaCandidate],
    differential: list[DifferentialResult],
    knowledge: KnowledgeService,
    *,
    top_k: int = 3,
    channel_weight: float = 0.6,
    scoring_policy: ScoringPolicy | None = None,
    excluded_paths: list[ReasoningPath] | None = None,
    evidence_gaps: list[EvidenceGap] | None = None,
) -> tuple[list[ReasoningPath], list[Evidence]]:
    """生成最多 K 条可溯源路径与精确来源目录；证据不足不凑满候选，不创建虚构 ID。"""
    if not 1 <= top_k <= 5 or not 0 <= channel_weight <= 1:
        raise ValueError("Top-K 或聚合权重越界")
    paths: list[ReasoningPath] = []
    for group, relation in _groups(candidates, relationships):
        channels = _unique([channel for item in group for channel in item.six_channels])
        compatible = [formula for formula in formulas if _compatible(formula, channels)]
        # 缺少患者支持的弱候选不可仅因检索有结果被强行配方。
        if not all(item.supporting_findings and item.status == "active" for item in group):
            compatible = []
        choices: list[FormulaCandidate | None] = list(compatible) if compatible else [None]
        for formula in choices:
            paths.append(
                await _make_path(
                    group,
                    relation,
                    formula,
                    clinical,
                    principles,
                    differential,
                    knowledge,
                    channel_weight,
                )
            )
    paths = list({item.path_id: item for item in paths}.values())
    categories = {
        "framework_evidence": "six_channel_framework",
        "classic_evidence": "classic_clause",
        "formula_evidence": "formula_pattern",
        "differential_evidence": "differential",
        "case_evidence": "medical_case",
    }
    # 必须先核验所有路径再评分排序，先截断会使原先低分但来源充分的路径永远无法晋级。
    references = [ref for path in paths for field in categories for ref in getattr(path, field)]
    catalog = await knowledge.get_evidence_by_ids(references)
    by_id = {item.evidence_id: item for item in catalog}
    for path in paths:
        for field, expected in categories.items():
            if any(by_id[ref].knowledge_type != expected for ref in getattr(path, field)):
                raise KnowledgeRetrievalError("路径引用的来源类别不一致")
        for ref in path.formula_evidence:
            source = by_id[ref].metadata
            associated = source.get(
                "canonical_six_channels"
                if path.association_scope == "classic_clause_based"
                else "case_derived_six_channels"
            )
            if (
                source.get("formula_name") != path.formula_pattern
                or source.get("association_scope") != path.association_scope
                or not isinstance(associated, list)
                or not set(path.six_channels).issubset(associated)
            ):
                raise KnowledgeRetrievalError("路径方证关联与原始来源不一致")
        path.evidence_assessments = await knowledge.assess_candidate_evidence(
            _unique([ref for field in categories for ref in getattr(path, field)]),
            clinical,
            channels=path.six_channels,
            formula_name=path.formula_pattern,
        )
        irrelevant = {
            item.evidence_id for item in path.evidence_assessments if item.grade == "irrelevant"
        }
        for field in categories:
            setattr(path, field, [ref for ref in getattr(path, field) if ref not in irrelevant])
        group = [item for item in candidates if item.candidate_id in path.channel_candidate_ids]
        relation = next(
            (
                item
                for item in relationships
                if item.relationship_type == path.relationship
                and {item.primary_candidate_id, *item.secondary_candidate_ids}
                == set(path.channel_candidate_ids)
            ),
            None,
        )
        score_path(
            path,
            clinical,
            group,
            relation,
            differential,
            list(by_id.values()),
            scoring_policy or ScoringPolicy(),
        )
        path.completeness_detail = summarize_completeness(
            path,
            list(by_id.values()),
            evidence_gaps or [],
            relation.missing_timeline_evidence if relation else [],
        )
        detail = path.completeness_detail
        path.evidence_completeness = (
            (detail.supported_count + detail.contradicted_count) / detail.total_count
            if detail.total_count
            else 0.0
        )
        # 完整度只表达已知覆盖；不把反证计为支持，不改变评分和排除判定。
        path.missing_key_findings = _unique(
            path.missing_key_findings
            + [
                item.source_text
                for item in detail.conditions
                if item.status == "unknown" and item.kind == "symptom"
            ]
        )
        path.treatment_summary = summarize_treatment(path, list(by_id.values()))
        path.treatment_principle = path.treatment_summary.text
    excluded = [path for path in paths if path.candidate_status == "excluded"]
    if excluded_paths is not None:
        excluded_paths.extend(excluded)
    paths = [path for path in paths if path.candidate_status != "excluded"]
    paths.sort(key=lambda item: (-item.ranking_score, item.path_id))
    selected = paths[:top_k]
    comparison_sources: list[Evidence] = []
    for index, path in enumerate(selected):
        path.rank = index + 1
        path.separation_from_next = (
            path.ranking_score - paths[index + 1].ranking_score if index + 1 < len(paths) else None
        )
        if index + 1 < len(paths):
            path.comparison_with_next, sources = await compare_next_path(
                path,
                paths[index + 1],
                clinical,
                differential,
                knowledge,
            )
            comparison_sources.extend(sources)
    # 同一来源在两条路径中可分别支持或反驳，因此用途只存路径，不能改共享 catalog。
    retained = {
        ref
        for path in [*selected, *excluded]
        for field in categories
        for ref in getattr(path, field)
    }
    catalog = [item for item in catalog if item.evidence_id in retained]
    catalog = list({item.evidence_id: item for item in [*catalog, *comparison_sources]}.values())
    return selected, catalog
