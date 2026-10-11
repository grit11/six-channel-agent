"""方证候选节点：从 KnowledgeService 召回来源方证并保留 canonical/历史派生边界。"""

from collections import defaultdict
from typing import Literal, cast

from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import match_source_phrase
from backend.app.rag.query_builder import build_formula_query
from backend.app.schemas.clinical import ClinicalEvidence
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
    if not 1 <= top_k <= 5:
        raise ValueError("方证候选 top_k 必须在 1 至 5 之间")
    gathered: dict[str, list] = defaultdict(list)
    for channel in channel_candidates:
        query = build_formula_query(
            evidence, [channel.provisional_syndrome or "", *channel.six_channels]
        )
        results = await knowledge.search_formula(
            query,
            channels=channel.six_channels,
            association_scope=["classic_clause_based", "case_derived_historical"],
            top_k=max(top_k, 5),
        )
        for result in results:
            gathered[result.evidence_id].append(result)

    candidates: list[FormulaCandidate] = []
    for evidence_id, evidences in gathered.items():
        item = evidences[0]
        metadata = item.metadata
        source_terms = _source_conditions(metadata)
        # grounded 元数据仍必须与真实来源正文对齐，未对齐内容只保留待核对。
        matched = {
            term: match_source_phrase(term, evidence) if term in item.content else "unknown"
            for term in source_terms
        }
        supporting = [term for term, status in matched.items() if status == "supports"]
        contradicting = [term for term, status in matched.items() if status == "contradicts"]
        missing = [term for term, status in matched.items() if status == "unknown"]
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
                risk_level=(str(metadata["risk_level"]) if metadata.get("risk_level") else None),
                rank_score=rank_score,
            )
        )
    candidates.sort(key=lambda item: (-item.rank_score, item.formula_name, item.association_scope))
    return candidates[:top_k]
