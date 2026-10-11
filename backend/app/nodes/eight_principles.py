"""八纲分析节点：把明确事实映射为四轴倾向，不直接判断六经或方剂。"""

from collections.abc import Iterable
from typing import cast

from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.reasoning_llm import analyze_eight_principles_with_llm
from backend.app.schemas.clinical import ClinicalEvidence, Finding
from backend.app.schemas.reasoning import AxisLean, EightPrinciplesState, PrincipleAxis

_AXIS_RULES: dict[str, tuple[set[str], set[str], str, str]] = {
    "yin_yang": (
        {"pulse_fine", "limb_cold", "diarrhea", "abdominal_fullness"},
        {"fever", "thirst", "bitter_taste", "pulse_floating", "pulse_tight"},
        "阴",
        "阳",
    ),
    "exterior_interior": (
        {"chills", "aversion_to_wind", "pulse_floating", "pulse_tight"},
        {"abdominal_fullness", "thirst", "bitter_taste", "diarrhea"},
        "表",
        "里",
    ),
    "cold_heat": (
        {"chills", "aversion_to_wind", "pulse_tight", "diarrhea", "abdominal_fullness"},
        {"fever", "thirst", "bitter_taste", "sweating"},
        "寒",
        "热",
    ),
    "deficiency_excess": (
        {"pulse_fine", "diarrhea", "limb_cold", "fatigue"},
        {"fever", "pulse_tight", "abdominal_fullness", "bitter_taste"},
        "虚",
        "实",
    ),
}


def _axis(
    findings: list[Finding],
    left_keys: set[str],
    right_keys: set[str],
    left_name: str,
    right_name: str,
) -> PrincipleAxis:
    """计算单轴倾向；unknown 只进入缺失列表，不参与任一侧分数。"""
    left_hits = [
        finding
        for finding in findings
        if finding.key in left_keys and finding.polarity == "positive"
    ]
    right_hits = [
        finding
        for finding in findings
        if finding.key in right_keys and finding.polarity == "positive"
    ]
    negative_left = [
        finding
        for finding in findings
        if finding.key in left_keys and finding.polarity == "negative"
    ]
    negative_right = [
        finding
        for finding in findings
        if finding.key in right_keys and finding.polarity == "negative"
    ]
    unknown = [finding.key for finding in findings if finding.polarity == "unknown"]
    left_score = min(1.0, sum(finding.certainty for finding in left_hits) / 2)
    right_score = min(1.0, sum(finding.certainty for finding in right_hits) / 2)
    # 多个重复原文不算多项独立依据；单一症状不决定八纲轴向。
    if len({finding.key for finding in left_hits + right_hits}) < 2:
        lean = "uncertain"
    elif left_score > 0.4 and right_score > 0.4 and abs(left_score - right_score) <= 0.2:
        lean = "mixed"
    elif left_score >= right_score:
        lean = "left"
    else:
        lean = "right"
    support = [f"{left_name}:{finding.key}" for finding in left_hits]
    support.extend(f"{right_name}:{finding.key}" for finding in right_hits)
    contradict = [f"{left_name}:{finding.key}" for finding in negative_left]
    contradict.extend(f"{right_name}:{finding.key}" for finding in negative_right)
    return PrincipleAxis(
        lean=cast(AxisLean, lean),
        score_left=left_score,
        score_right=right_score,
        supporting_findings=support,
        contradicting_findings=contradict,
        unknown_key_findings=sorted(set(unknown).intersection(left_keys | right_keys)),
    )


def _summary(axes: Iterable[tuple[str, PrincipleAxis]]) -> str:
    """生成面向医生的短摘要，不生成诊断或处方结论。"""
    return "；".join(
        f"{name}:"
        + (
            f"{name[0]}倾向"
            if axis.lean == "left"
            else (
                f"{name[1]}倾向"
                if axis.lean == "right"
                else "夹杂" if axis.lean == "mixed" else "未定"
            )
        )
        for name, axis in axes
    )


def analyze_eight_principles(evidence: ClinicalEvidence) -> EightPrinciplesState:
    """根据明确 Finding 计算四轴倾向，Unknown 不扣分也不补全。"""
    axes: dict[str, PrincipleAxis] = {}
    # 同键同极性去重，历史和冲突事实不用于当前轴评分。
    findings = list(
        {(item.key, item.polarity): item for item in evidence.current_findings()}.values()
    )
    for name, (left, right, left_name, right_name) in _AXIS_RULES.items():
        axes[name] = _axis(findings, left, right, left_name, right_name)
    return EightPrinciplesState(
        yin_yang=axes["yin_yang"],
        exterior_interior=axes["exterior_interior"],
        cold_heat=axes["cold_heat"],
        deficiency_excess=axes["deficiency_excess"],
        summary=_summary(
            (
                ("阴阳", axes["yin_yang"]),
                ("表里", axes["exterior_interior"]),
                ("寒热", axes["cold_heat"]),
                ("虚实", axes["deficiency_excess"]),
            )
        ),
    )


async def analyze_eight_principles_node(
    evidence: ClinicalEvidence,
    knowledge: KnowledgeService,
    llm: StructuredLLM | None,
) -> EightPrinciplesState:
    """按显式模式选择规则基线或结构化八纲模型；模型失败不静默回退。"""
    if llm is None:
        return analyze_eight_principles(evidence)
    return await analyze_eight_principles_with_llm(evidence, knowledge, llm)
