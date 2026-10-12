"""方证候选节点：从 KnowledgeService 召回来源方证并保留 canonical/历史派生边界。"""

from typing import Literal, cast

from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.query_builder import build_formula_query
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.knowledge import FormulaBranchUnit
from backend.app.schemas.reasoning import ChannelCandidate, FormulaCandidate


def _metadata_terms(value: object) -> list[str]:
    """读取知识库中已审核的字符串列表，异常字段不被隐式转换。"""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _source_conditions(metadata: dict[str, object]) -> list[str]:
    """合并来源已标注的正向与阴性条件，保留无法完整规范化的原文。"""
    values = [
        *_metadata_terms(metadata.get("key_symptoms_source_grounded", [])),
        *_metadata_terms(metadata.get("negative_findings_source_grounded", [])),
    ]
    return list(dict.fromkeys(term.strip() for term in values if term.strip()))


async def generate_formula_candidates(
    evidence: ClinicalEvidence,
    channel_candidates: list[ChannelCandidate],
    knowledge: KnowledgeService,
    *,
    top_k: int = 5,
) -> list[FormulaCandidate]:
    """按六经候选检索方证，分层返回 canonical 与历史医案派生结果。"""
    if not 1 <= top_k <= 50:
        raise ValueError("方证候选 top_k 必须在 1 至 50 之间")
    query = build_formula_query(evidence, [])
    pooled, branches = await knowledge.search_formula_branches(query, top_k=top_k)
    gathered = {item.evidence_id: [item] for item in pooled}

    candidates: list[FormulaCandidate] = []
    for evidence_id, evidences in gathered.items():
        item = evidences[0]
        metadata = item.metadata
        from backend.app.rag.grader import grade_candidate_evidence

        selected_branches = [
            branch for branch in branches if branch.formula_evidence_id == evidence_id
        ]
        evaluable = [
            branch for branch in selected_branches if branch.logical_scope == "single_clause"
        ]
        if evaluable:
            # 与 Grader 同一排序准则选最佳单条分支；其余分支身份留在知识索引，不能累加。
            def branch_rank(
                branch: FormulaBranchUnit,
                parent: Evidence = item,
                source_metadata: dict[str, object] = metadata,
            ) -> tuple[int, int, str]:
                """按真实条件核验排序，重复分支不加分。"""
                checked = grade_candidate_evidence(
                    parent,
                    evidence,
                    channels=_metadata_terms(source_metadata.get("canonical_six_channels")),
                    formula_name=str(source_metadata.get("formula_name", "")),
                    branches=[branch],
                )
                return (
                    len(checked.supporting_requirements) - len(checked.contradicting_requirements),
                    -len(checked.unresolved_requirements),
                    checked.rationale,
                )

            selected_branches = [max(evaluable, key=branch_rank)]
        assessment = grade_candidate_evidence(
            item,
            evidence,
            channels=_metadata_terms(metadata.get("canonical_six_channels"))
            or _metadata_terms(metadata.get("case_derived_six_channels")),
            formula_name=str(metadata.get("formula_name", "")),
            branches=(
                selected_branches
                if metadata.get("association_scope") == "classic_clause_based"
                else None
            ),
        )
        supporting = assessment.supporting_requirements
        contradicting = assessment.contradicting_requirements
        missing = assessment.unresolved_requirements
        if not selected_branches:
            missing = [*missing, "缺少已核验单条方证分支；聚合来源仅供审核"]
        # 阴性来源字段不自动等同禁忌：完整可识别时参与当前事实核对，长说明保留为缺口。
        match_bonus = min(1.0, 0.12 * len(supporting))
        contradiction_penalty = min(1.0, 0.18 * len(contradicting))
        rank_score = max(0.0, match_bonus - contradiction_penalty)
        candidates.append(
            FormulaCandidate(
                formula_name=str(metadata.get("formula_name", item.knowledge_id)),
                association_scope=cast(
                    Literal["classic_clause_based", "case_derived_historical"],
                    metadata["association_scope"],
                ),
                canonical_six_channels=_metadata_terms(metadata.get("canonical_six_channels", [])),
                case_derived_six_channels=_metadata_terms(
                    metadata.get("case_derived_six_channels", [])
                ),
                supporting_findings=supporting,
                contradicting_findings=contradicting,
                missing_key_findings=missing,
                source_clause_nos=_metadata_terms(metadata.get("source_clause_nos", [])),
                evidence_ids=[evidence_id],
                branch_ids=[branch.branch_id for branch in selected_branches],
                branch_clause_evidence_ids=list(
                    dict.fromkeys(
                        ref for branch in selected_branches for ref in branch.clause_evidence_ids
                    )
                ),
                branch_status=(
                    "source_verified_pending_expert" if selected_branches else "unavailable"
                ),
                risk_level=(str(metadata["risk_level"]) if metadata.get("risk_level") else None),
                rank_score=rank_score,
            )
        )
    candidates.sort(key=lambda item: (-item.rank_score, item.formula_name, item.association_scope))
    return candidates[:top_k]
