"""按知识类型构造可审计检索查询，不决定六经、证型或方剂。"""

from backend.app.schemas.clinical import ClinicalEvidence


def _known_finding_text(evidence: ClinicalEvidence) -> list[str]:
    """只拼接医生或记录中明确的 positive/negative 事实，跳过 unknown。"""
    return [
        finding.source_text or f"{finding.key}:{finding.polarity}:{finding.value}"
        for finding in evidence.current_findings()
        if finding.polarity in {"positive", "negative"}
        and finding.value is not None
        and finding.source != "inference"
    ]


def build_classic_query(
    evidence: ClinicalEvidence, channels: list[str] | None = None, terms: list[str] | None = None
) -> str:
    """构造经典查询：关键脉证、候选六经和原文术语分层拼接。"""
    parts = _known_finding_text(evidence) + (channels or []) + (terms or [])
    return "；".join(dict.fromkeys(part for part in parts if part))


def build_formula_query(
    evidence: ClinicalEvidence,
    candidates: list[str] | None = None,
    key_negatives: list[str] | None = None,
) -> str:
    """构造方证查询：主症、已明确阴性事实和候选方证分开保留。"""
    parts = _known_finding_text(evidence) + (key_negatives or []) + (candidates or [])
    return "；".join(dict.fromkeys(part for part in parts if part))


def build_differential_query(
    candidates: list[str], unresolved_dimensions: list[str] | None = None
) -> str:
    """构造鉴别查询：保留候选 A/B 和尚未解决的维度。"""
    return " vs ".join(candidates) + (
        "；未解决：" + "、".join(unresolved_dimensions) if unresolved_dimensions else ""
    )


def build_case_query(evidence: ClinicalEvidence, channels: list[str] | None = None) -> str:
    """构造医案查询：主诉、明确脉证和宽六经候选用于后段参考。"""
    parts = [evidence.chief_complaint] if evidence.chief_complaint else []
    parts += _known_finding_text(evidence) + (channels or [])
    return "；".join(dict.fromkeys(part for part in parts if part))
