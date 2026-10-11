"""六经候选节点：先独立宽召回，再用真实 KnowledgeService 证据校正排序。"""

from dataclasses import dataclass
from typing import Literal

from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.reasoning_llm import generate_channel_with_llm
from backend.app.rag.query_builder import build_classic_query, build_formula_query
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.reasoning import ChannelCandidate, EightPrinciplesState


@dataclass(frozen=True)
class _ChannelRule:
    """候选锚点规则，仅用于宽召回，不等于最终诊断规则。"""

    channel: str
    positive_keys: frozenset[str]
    negative_support_keys: frozenset[str]
    discriminators: frozenset[str]
    syndrome: str


_RULES = (
    _ChannelRule(
        "太阳",
        frozenset({"fever", "chills", "aversion_to_wind", "pulse_floating", "pulse_tight"}),
        frozenset({"sweating"}),
        frozenset({"chills", "pulse_floating", "pulse_tight", "sweating"}),
        "太阳相关表证候选（待经典核验）",
    ),
    _ChannelRule(
        "阳明",
        frozenset({"fever", "thirst", "abdominal_fullness", "sweating"}),
        frozenset({"chills"}),
        frozenset({"fever", "thirst", "abdominal_fullness", "chills"}),
        "阳明相关里证候选（待经典核验）",
    ),
    _ChannelRule(
        "少阳",
        frozenset({"bitter_taste", "thirst", "fever"}),
        frozenset(),
        frozenset({"bitter_taste", "thirst"}),
        "少阳相关枢机证候选（待经典核验）",
    ),
    _ChannelRule(
        "太阴",
        frozenset({"abdominal_fullness", "diarrhea", "appetite_loss"}),
        frozenset({"fever", "thirst"}),
        frozenset({"abdominal_fullness", "diarrhea", "appetite_loss"}),
        "太阴相关脾虚寒候选（待经典核验）",
    ),
    _ChannelRule(
        "少阴",
        frozenset({"pulse_fine", "limb_cold", "fatigue", "diarrhea"}),
        frozenset({"fever"}),
        frozenset({"pulse_fine", "limb_cold", "fatigue"}),
        "少阴相关候选（待经典核验）",
    ),
    _ChannelRule(
        "厥阴",
        frozenset({"bitter_taste", "vomiting", "diarrhea", "limb_cold"}),
        frozenset(),
        frozenset({"bitter_taste", "vomiting", "limb_cold"}),
        "厥阴相关候选（待经典核验）",
    ),
)


def _score_rule(
    rule: _ChannelRule, evidence: ClinicalEvidence
) -> tuple[float, list[str], list[str], list[str]]:
    """独立计算候选匹配分；Unknown 不扣分，明确阴性按规则用途处理。"""
    supporting: list[str] = []
    contradicting: list[str] = []
    known_keys: set[str] = set()
    findings = {(item.key, item.polarity): item for item in evidence.current_findings()}
    for finding in findings.values():
        if finding.polarity == "unknown":
            continue
        known_keys.add(finding.key)
        if finding.polarity == "positive" and finding.key in rule.positive_keys:
            supporting.append(finding.source_text or finding.key)
        elif finding.polarity == "negative" and finding.key in rule.negative_support_keys:
            supporting.append(f"明确阴性：{finding.source_text or finding.key}")
        elif finding.polarity == "negative" and finding.key in rule.positive_keys:
            contradicting.append(f"明确阴性反证：{finding.source_text or finding.key}")
    raw = min(1.0, len(supporting) / 4) - min(0.35, len(contradicting) * 0.18)
    return max(0.0, raw), supporting, contradicting, sorted(rule.discriminators - known_keys)


async def generate_channel_candidates(
    evidence: ClinicalEvidence,
    principles: EightPrinciplesState,
    knowledge: KnowledgeService,
    *,
    top_k: int = 3,
    llm: StructuredLLM | None = None,
) -> list[ChannelCandidate]:
    """生成多经独立候选，并附加框架、经典、方证真实来源引用。"""
    if not 1 <= top_k <= 5:
        raise ValueError("六经候选 top_k 必须在 1 至 5 之间")
    if llm is not None:
        return await _generate_llm_candidates(evidence, principles, knowledge, llm, top_k=top_k)
    classic_query = build_classic_query(evidence)
    candidates: list[ChannelCandidate] = []
    for rule in _RULES:
        score, supporting, contradicting, missing = _score_rule(rule, evidence)
        framework = await knowledge.search_core_framework(f"{rule.channel} 核心", top_k=2)
        classic = await knowledge.search_classic(
            classic_query or rule.channel, semantic_channels=[rule.channel], top_k=3
        )
        formula = await knowledge.search_formula(
            build_formula_query(evidence, [rule.syndrome]),
            channels=[rule.channel],
            association_scope=["classic_clause_based", "case_derived_historical"],
            top_k=3,
        )
        # 文档数量、召回相关性与八纲已计算过的事实不应重复制造支持分。
        # RAG 多因子校正由后续有适用性核验的评分协议接入。
        final_score = score
        completeness = len(rule.discriminators - set(missing)) / max(1, len(rule.discriminators))
        status: Literal["active", "weak"] = "active" if final_score >= 0.2 else "weak"
        candidates.append(
            ChannelCandidate(
                candidate_id=f"channel_{rule.channel}",
                six_channels=[rule.channel],
                provisional_syndrome=rule.syndrome,
                independent_score=final_score,
                pass_a_score=min(1.0, len(supporting) / 4),
                pass_b_score=final_score,
                supporting_findings=supporting,
                contradicting_findings=contradicting,
                missing_key_findings=missing,
                classic_evidence_ids=[item.evidence_id for item in classic],
                formula_evidence_ids=[item.evidence_id for item in formula],
                framework_evidence_ids=[item.evidence_id for item in framework],
                evidence_completeness=completeness,
                status=status,
            )
        )
    candidates.sort(key=lambda item: (-item.independent_score, item.candidate_id))
    return candidates[:top_k]


async def _generate_llm_candidates(
    evidence: ClinicalEvidence,
    principles: EightPrinciplesState,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
    *,
    top_k: int,
) -> list[ChannelCandidate]:
    """执行 Pass A 模型宽召回，再用真实来源适用性做透明 Pass B 校正。"""
    candidates = await generate_channel_with_llm(
        evidence, principles, knowledge, llm, top_k=min(6, max(top_k, 3))
    )
    query = build_classic_query(evidence)
    enriched: list[ChannelCandidate] = []
    for candidate in candidates:
        classic = await knowledge.search_classic(
            query or candidate.six_channels[0],
            semantic_channels=candidate.six_channels,
            top_k=3,
        )
        formulas = await knowledge.search_formula(
            build_formula_query(
                evidence, [candidate.provisional_syndrome or "", *candidate.six_channels]
            ),
            channels=candidate.six_channels,
            association_scope=["classic_clause_based", "case_derived_historical"],
            top_k=3,
        )
        assessments = await knowledge.assess_candidate_evidence(
            [item.evidence_id for item in classic],
            evidence,
            channels=candidate.six_channels,
        )
        # 方证核验必须绑定它自己的方名，否则 Grader 会因缺少归属判为不适用。
        for source in formulas:
            formula_name = source.metadata.get("formula_name")
            if not isinstance(formula_name, str):
                continue
            assessments.extend(
                await knowledge.assess_candidate_evidence(
                    [source.evidence_id],
                    evidence,
                    channels=candidate.six_channels,
                    formula_name=formula_name,
                )
            )
        relevant = [item for item in assessments if item.grade != "irrelevant"]
        supporting = {term for item in relevant for term in item.supporting_requirements}
        contradicting = {term for item in relevant for term in item.contradicting_requirements}
        unresolved = {term for item in relevant for term in item.unresolved_requirements}
        known_count = len(supporting | contradicting)
        # 正向/反向校正来自唯一已知条件比例；未知不罚分，重复来源不改变分数。
        adjustment = 0.25 * len(supporting) / max(1, known_count) - 0.35 * len(contradicting) / max(
            1, known_count
        )
        pass_a = (
            candidate.pass_a_score
            if candidate.pass_a_score is not None
            else candidate.independent_score
        )
        pass_b = max(0.0, min(1.0, pass_a + adjustment))
        candidate.pass_a_score = pass_a
        candidate.evidence_adjustment = adjustment
        candidate.evidence_assessments = relevant
        candidate.pass_b_score = pass_b
        candidate.independent_score = pass_b
        candidate.classic_evidence_ids = [item.evidence_id for item in classic]
        candidate.formula_evidence_ids = [item.evidence_id for item in formulas]
        candidate.evidence_completeness = max(
            candidate.evidence_completeness,
            known_count / max(1, len(supporting | contradicting | unresolved)),
        )
        if contradicting and not candidate.supporting_findings:
            candidate.status = "weak"
        enriched.append(candidate)
    enriched.sort(key=lambda item: (-(item.pass_b_score or 0.0), item.candidate_id))
    return enriched[:top_k]
