"""来源条件规范化与三值核验。

职责：依赖共享临床词项解析，将完整症状条件转为 AND/OR 分支并保留原文审计。
边界：不检索、不调用模型、不创作医学条件；治疗、病机、未解析分支保持 unknown。
自然语言中混合且/或的作用域不清时不猜测优先级，明确括号才执行复合逻辑。
"""

import re
from itertools import product

from backend.app.nodes.clinical_parser import source_phrase_requirements
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import SourceConditionAssessment

_OR = re.compile(r"或者|或是|或")
_AND = re.compile(r"并且|同时|以及|且|而|并|伴|和|与|[、，,；;]")
_TEMPORAL = re.compile(r"治疗前|治疗后|服药前|服药后|汗后|误下后|误火后|病后|先.+后|若|之后|之前")
_TREATMENT = re.compile(r"治法|用药|治疗|得汤|攻下|温降|宜|不宜|泻热|和解|救逆")
_MECHANISM = re.compile(r"病机|气化|阳虚|阴虚|寒邪|邪热|水热|血瘀|瘀热|气分|营卫|枢机")


def _split_top_level(text: str, pattern: re.Pattern[str]) -> list[str] | None:
    """只在括号外划分逻辑连接词；不平衡括号和空分支直接拒绝。"""
    depth, start = 0, 0
    parts: list[str] = []
    position = 0
    while position < len(text):
        char = text[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        match = pattern.match(text, position) if depth == 0 else None
        if match:
            parts.append(text[start:position].strip())
            position = start = match.end()
        else:
            position += 1
    parts.append(text[start:].strip())
    return parts if depth == 0 and all(parts) else None


def _branches(text: str, depth: int = 0) -> list[dict[str, str]] | None:
    """解析有限深度的明确括号表达式，拒绝歧义、冲突和分支爆炸。"""
    if depth > 8:
        return None
    text = text.strip()
    # 只有括号包住整个表达式才可剥离，避免 (A)且(B) 被误当成一组。
    if text.startswith("(") and text.endswith(")"):
        level = 0
        for index, char in enumerate(text):
            level += int(char == "(") - int(char == ")")
            if level == 0:
                if index == len(text) - 1:
                    return _branches(text[1:-1], depth + 1)
                break
    ors, ands = _split_top_level(text, _OR), _split_top_level(text, _AND)
    if ors is None or ands is None or (len(ors) > 1 and len(ands) > 1):
        return None
    parts = ors if len(ors) > 1 else ands
    if len(parts) == 1:
        required = source_phrase_requirements(text)
        return [required] if required else None
    children = [_branches(part, depth + 1) for part in parts]
    if any(child is None for child in children):
        return None
    parsed = [child for child in children if child is not None]
    if len(ors) > 1:
        branches = [branch for child in parsed for branch in child]
        return branches if len(branches) <= 32 else None
    if _product_size(parsed) > 32:
        return None
    combined: list[dict[str, str]] = []
    for selection in product(*parsed):
        merged: dict[str, str] = {}
        for branch in selection:
            if any(key in merged and merged[key] != value for key, value in branch.items()):
                return None
            merged.update(branch)
        combined.append(merged)
    return combined


def _product_size(groups: list[list[dict[str, str]]]) -> int:
    """执行展开前限制分支组合数量，避免来源文本消耗无界计算。"""
    size = 1
    for group in groups:
        size *= len(group)
    return size


def assess_source_condition(
    phrase: str, evidence: ClinicalEvidence, *, dimension: str = ""
) -> SourceConditionAssessment:
    """来源完整解析才执行；OR 任一支持即支持，所有分支反证才反证。"""
    condition = SourceConditionAssessment(source_text=phrase, kind="unparsed")
    if any(term in dimension for term in ("病机", "病性", "病位", "总纲", "共同结构")):
        condition.kind = "mechanism"
    elif any(term in dimension for term in ("治法", "用药", "治疗")):
        condition.kind = "treatment"
    elif _TEMPORAL.search(phrase) or any(term in dimension for term in ("病程", "次序")):
        condition.kind = "temporal"
    elif _TREATMENT.search(phrase):
        condition.kind = "treatment"
    elif "追问" in dimension or "问题" in dimension or "是否" in phrase or "有无" in phrase:
        condition.kind = "question"
    elif _MECHANISM.search(phrase):
        condition.kind = "mechanism"
    if condition.kind != "unparsed":
        condition.unresolved_reason = "需单独核验病机、治疗或时间分支，不能作为当前症状条件执行"
        return condition
    normalized = (
        phrase.strip().rstrip("。.").removesuffix("者").replace("（", "(").replace("）", ")")
    )
    # 来源中的频率限定不改变症状极性；只归一整段开头的修饰语。
    # 不拆长说明、不移除“非必见”等非必要性声明，残余词项仍保持 Unknown。
    normalized = re.sub(r"^(?:多|常)(?=\S)", "", normalized)
    branches = _branches(normalized)
    if branches is None:
        condition.unresolved_reason = "词项、修饰语或逻辑作用域未完整规范化，保留原文供医生核验"
        return condition
    condition.kind, condition.alternatives = "symptom", branches
    known = {
        item.key: item.polarity
        for item in evidence.current_findings()
        if item.polarity != "unknown"
    }
    statuses = [
        (
            "contradicts"
            if any(key in known and known[key] != expected for key, expected in branch.items())
            else (
                "supports"
                if all(known.get(key) == expected for key, expected in branch.items())
                else "unknown"
            )
        )
        for branch in branches
    ]
    condition.status = (
        "supports"
        if "supports" in statuses
        else "contradicts" if all(item == "contradicts" for item in statuses) else "unknown"
    )
    if condition.status == "unknown":
        condition.unresolved_reason = "当前明确事实不足以确认或反证整体条件"
    return condition
