"""确定性临床词项解析与来源短语核对。

职责：保留每次明确出现的事实、局部否定、复合脉象及原文时间；供节点共享词项语义。
边界：依赖 ClinicalEvidence 校验冲突与注入模型，不推断六经、不访问知识或模型厂商。
无法完整规范化的来源短语保持 unknown，不能用子串命中补齐患者事实。
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.ports import StructuredLLM
from backend.app.schemas.clinical import (
    ClinicalEvidence,
    Finding,
    FindingSource,
    Polarity,
    fact_time_group,
)

# 仅为同一明确表现的词项归一，不在此建立辨证或排除规则。
_TERMS = {
    "发热": "fever",
    "高热": "high_fever",
    "恶寒": "chills",
    "恶风": "aversion_to_wind",
    "汗出": "sweating",
    "出汗": "sweating",
    "自汗": "sweating",
    "无汗": "sweating",
    "口渴": "thirst",
    "口苦": "bitter_taste",
    "腹满": "abdominal_fullness",
    "下利": "diarrhea",
    "腹泻": "diarrhea",
    "呕吐": "vomiting",
    "干呕": "dry_heaving",
    "纳差": "appetite_loss",
    "食欲不振": "appetite_loss",
    "乏力": "fatigue",
    "四肢厥冷": "limb_cold",
    "手足厥冷": "limb_cold",
    "肢冷": "limb_cold",
    "头痛": "headache",
    "头项强痛": "head_neck_pain",
    "身疼痛": "body_pain",
    "身痛": "body_pain",
    "骨节疼痛": "joint_pain",
    "腰痛": "back_pain",
    "胸满": "chest_fullness",
    "喘": "dyspnea",
    "咳嗽": "cough",
    "烦躁": "agitation",
    "身热": "fever",
    "身疼": "body_pain",
    "骨节痛": "joint_pain",
    "身体痛": "body_pain",
    "不渴": "thirst",
    "小便不利": "urination_difficult",
    "小便自利": "urination_free",
    "小便短赤": "urination_short_red",
    "小便清长": "urination_clear_long",
    "少腹急结": "lower_abdominal_tension",
    "少腹硬满": "lower_abdominal_hard_fullness",
    "少腹苦里急": "lower_abdominal_urgency",
    "如狂": "mania_like",
    "发狂": "mania",
    "消渴": "consumptive_thirst",
    "便秘": "constipation",
    "便结": "constipation",
    "大便硬": "hard_stool",
    "潮热": "tidal_fever",
    "谵语": "delirium",
    "胸胁苦满": "hypochondriac_fullness",
    "往来寒热": "alternating_chills_fever",
    "心下痞": "epigastric_stuffiness",
    "心下满": "epigastric_fullness",
    "心下悸": "epigastric_palpitation",
    "心烦": "irritability",
    "不得眠": "insomnia",
    "不得卧": "unable_to_lie_sleep",
    "但欲寐": "somnolence",
    "腹痛": "abdominal_pain",
    "腹中痛": "abdominal_pain",
    "腹中急痛": "abdominal_cramping",
    "肠鸣": "borborygmus",
    "咽干": "dry_throat",
    "口燥": "dry_mouth",
    "目眩": "dizziness",
    "头眩": "dizziness",
    "不欲饮食": "appetite_loss",
    "喜呕": "nausea",
    "欲呕吐": "nausea",
    "欲呕": "nausea",
    "身黄": "jaundice",
    "里急后重": "tenesmus",
    "便脓血": "bloody_purulent_stool",
    "肛门灼热": "anal_burning",
    "手足厥寒": "limb_cold",
    "手足逆冷": "limb_cold",
    "手足寒": "limb_cold",
    "四肢不温": "limb_cold",
    "舌红": "tongue_red",
    "舌淡": "tongue_pale",
    "苔黄": "coating_yellow",
    "苔白": "coating_white",
}
_PULSE_KEYS = {
    "浮": "pulse_floating",
    "紧": "pulse_tight",
    "缓": "pulse_relaxed",
    "弱": "pulse_weak",
    "细": "pulse_fine",
    "沉": "pulse_deep",
    "数": "pulse_rapid",
    "迟": "pulse_slow",
    "微": "pulse_faint",
    "弦": "pulse_wiry",
    "滑": "pulse_slippery",
    "涩": "pulse_choppy",
}
_TOKEN_PATTERN = re.compile(
    "脉["
    + "".join(_PULSE_KEYS)
    + "]+|"
    + "|".join(re.escape(term) for term in sorted(_TERMS, key=len, reverse=True))
)
_NEGATION = re.compile(r"(?:否认|未见|没有|不是|未|无|不)\s*$")
_UNCERTAIN = re.compile(r"(?:是否|有无|不详|不确定|可能|疑似)\s*$")
_UNCERTAIN_SUFFIX = re.compile(r"\s*(?:不详|不确定|未知|未问|待查)")
_TIME_PATTERN = re.compile(
    r"昨日|昨天|今日|今天|当前|现在|目前|既往|曾经|治疗前|治疗后|服药前|服药后|"
    r"[一二三四五六七八九十\d]+天前"
)
_UNKNOWN_KEYS = tuple(dict.fromkeys((*_TERMS.values(), *_PULSE_KEYS.values(), "tongue", "pulse")))
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "clinical_parser.md"
PhraseStatus = Literal["supports", "contradicts", "unknown"]
RULE_PARSER_VERSION = "rules-v0.3"


def _extract(text: str, source: FindingSource) -> list[Finding]:
    """逐段逐次抽取，否定仅绑定紧邻词项；明确时间可跨同一句逗号延续。"""
    findings: list[Finding] = []
    scope: str | None = None
    for clause in re.split(r"([，,。；;\n])", text):
        if clause in {"。", "；", ";", "\n"}:
            scope = None
            continue
        if not clause or clause in {"，", ","}:
            continue
        cursor = 0
        for match in _TOKEN_PATTERN.finditer(clause):
            prefix = clause[cursor : match.start()]
            times = list(_TIME_PATTERN.finditer(prefix))
            if times:
                scope = times[-1].group()
            negative = _NEGATION.search(prefix)
            uncertain = _UNCERTAIN.search(prefix)
            suffix = _UNCERTAIN_SUFFIX.match(clause, match.end())
            term = match.group()
            polarity: Polarity = (
                "unknown"
                if uncertain or suffix or "或" in clause
                else "negative" if negative or term in {"无汗", "不渴"} else "positive"
            )
            modifier = uncertain or negative
            start = cursor + modifier.start() if modifier else match.start()
            # 后置“不详”同样属于该事实的原文，不能因前面出现症状词就认定阳性。
            end = suffix.end() if suffix else match.end()
            original = clause[start:end]
            keys = (
                [_PULSE_KEYS[char] for char in term[1:]]
                if term.startswith("脉")
                else [_TERMS[term]]
            )
            # 否认一个复合脉象不等于分别否认每个组成，留给模型/医生规范化。
            if polarity == "negative" and len(keys) > 1:
                keys = ["pulse_composite:" + term]
            for key in dict.fromkeys(keys):
                findings.append(
                    Finding(
                        key=key,
                        value=None if polarity == "unknown" else polarity == "positive",
                        polarity=polarity,
                        certainty=0.0 if polarity == "unknown" else 1.0,
                        source=source,
                        source_text=original,
                        time_scope=scope,
                    )
                )
            cursor = end
    return findings


def parse_clinical_text(text: str, *, source: FindingSource = "doctor_input") -> ClinicalEvidence:
    """保留逐次出现的原文事实；未提及项生成 Unknown，历史变化不覆盖当前事实。"""
    if not text.strip():
        raise ValueError("临床输入不能为空")
    findings = _extract(text, source)
    present_keys = {finding.key for finding in findings}
    findings.extend(
        Finding(key=key, value=None, polarity="unknown", certainty=0.0, source=source)
        for key in _UNKNOWN_KEYS
        if key not in present_keys
    )
    return ClinicalEvidence(
        chief_complaint=re.split(r"[，,。；;\n]", text, maxsplit=1)[0].strip() or None,
        findings=findings,
    )


def source_phrase_requirements(phrase: str) -> dict[str, str] | None:
    """仅将完整可识别的来源短语转为要求；长说明和残余实体不被猜测补全。"""
    # 来源时间要求不能经词项解析丢失时间后参与当前事实评分。
    if _TIME_PATTERN.search(phrase):
        return None
    remainder = _TOKEN_PATTERN.sub("", phrase)
    remainder = re.sub(r"否认|未见|没有|不是|无|未|不|者|而|并|伴|\s|[，,。；;、]", "", remainder)
    if remainder:
        return None
    facts = _extract(phrase, "record")
    requirements: dict[str, str] = {}
    for finding in facts:
        if finding.polarity == "unknown" or finding.key.startswith("pulse_composite:"):
            return None
        if finding.key in requirements and requirements[finding.key] != finding.polarity:
            return None
        requirements[finding.key] = finding.polarity
    return requirements or None


def source_phrase_observed_requirements(phrase: str) -> dict[str, str]:
    """提取长说明中可观察的完整事实；不代表整段条件已完成核验。"""
    facts = _extract(phrase, "record")
    requirements: dict[str, str] = {}
    for finding in facts:
        if finding.polarity == "unknown" or finding.key.startswith("pulse_composite:"):
            continue
        if finding.key in requirements and requirements[finding.key] != finding.polarity:
            continue
        requirements[finding.key] = finding.polarity
    return requirements


def match_source_phrase(phrase: str, evidence: ClinicalEvidence) -> PhraseStatus:
    """逐项核对来源要求与当前明确事实；缺失不能成为支持，阴性要求不是禁忌。"""
    # 延迟导入避免词项解析器与来源逻辑核验器互相初始化；公开旧接口保留兼容。
    from backend.app.nodes.source_conditions import assess_source_condition

    return assess_source_condition(phrase, evidence).status


def _validate_model_facts(
    evidence: ClinicalEvidence, text: str, source: FindingSource
) -> ClinicalEvidence:
    """校验事实原文和词项极性；拒绝无来源、诊断键、改写否定和虚构时间线。"""
    keys = {item.key for item in evidence.findings}
    original_facts = _extract(text, source)
    for finding in evidence.findings:
        if finding.source != source or not re.fullmatch(r"[a-z][a-z0-9_:.]*", finding.key):
            raise ValueError("模型事实来源或规范键无效")
        if any(term in finding.key for term in ("diagnosis", "formula", "six_channel", "syndrome")):
            raise ValueError("模型不得在解析阶段输出诊断")
        if finding.polarity == "unknown":
            if finding.value is not None:
                raise ValueError("Unknown 事实值必须为空")
            continue
        if not finding.source_text or finding.source_text not in text or finding.value is None:
            raise ValueError("模型事实缺少原文支持")
        if finding.time_scope is not None and finding.time_scope not in text:
            raise ValueError("模型时间范围缺少原文支持")
        if isinstance(finding.value, bool) and finding.value != (finding.polarity == "positive"):
            raise ValueError("布尔事实值与极性不一致")
        extracted = _extract(finding.source_text, source)
        matching = [item for item in extracted if item.key == finding.key]
        if matching and finding.polarity not in {item.polarity for item in matching}:
            raise ValueError("模型改写了明确的局部否定")
        # 已识别的规范键须在来源片段中出现，不能拿脉浮原文补脉紧。
        if finding.key in _UNKNOWN_KEYS and not matching:
            raise ValueError("模型补齐了原文未出现的事实")
        if finding.key in _UNKNOWN_KEYS and not any(
            item.key == finding.key
            and item.polarity == finding.polarity
            and fact_time_group(item.time_scope) == fact_time_group(finding.time_scope)
            for item in original_facts
        ):
            raise ValueError("模型改变了事实所属的时间范围")
        if (
            isinstance(finding.value, (str, float))
            and str(finding.value) not in finding.source_text
        ):
            raise ValueError("模型数值或定性值缺少原文支持")
    for original in original_facts:
        if original.polarity == "unknown":
            continue
        if not any(
            item.key == original.key
            and item.polarity == original.polarity
            and fact_time_group(item.time_scope) == fact_time_group(original.time_scope)
            for item in evidence.findings
        ):
            raise ValueError("模型丢失了原文中明确存在的词项或局部否定")
    if (
        evidence.timeline
        or evidence.treatment_events
        or evidence.relationship_history != ClinicalEvidence().relationship_history
    ):
        raise ValueError("模型解析不能生成医生确认的六经时间线")
    for field in ("chief_complaint", "onset", "course"):
        value = getattr(evidence, field)
        if value and value not in text:
            raise ValueError("模型输入摘要缺少连续原文支持")
    if any(item not in text for item in evidence.triggers):
        raise ValueError("模型诱因缺少原文支持")
    # 区域只允许通过规范键关联已验证事实，防止在未校验区域隐藏新增事实。
    for field in (
        "fever_chills",
        "sweating",
        "thirst_mouth",
        "appetite_digestive",
        "stool",
        "urination",
        "chest_hypochondrium",
        "cough_dyspnea_phlegm",
        "spirit_sleep",
        "limbs_body",
        "tongue",
        "pulse",
    ):
        for key, value in getattr(evidence, field).items():
            if key not in keys or not any(
                item.key == key and item.value == value for item in evidence.findings
            ):
                raise ValueError("模型四诊区域缺少对应的已验证事实")
    if evidence.pain or evidence.prior_treatment:
        raise ValueError("疼痛与治疗变化应先作为有时间和原文的 Finding 返回")
    return ClinicalEvidence.model_validate(evidence.model_dump())


async def parse_clinical_with_llm(
    text: str, llm: StructuredLLM, *, source: FindingSource = "doctor_input"
) -> ClinicalEvidence:
    """调用专用 Prompt 和结构化模型，来源验证失败时明确失败，不偷偷退回规则解析。"""
    if not text.strip():
        raise ValueError("临床输入不能为空")
    prompt = await asyncio.to_thread(_PROMPT_PATH.read_text, encoding="utf-8")
    evidence = await llm.ainvoke_structured(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"source": source, "clinical_text": text}, ensure_ascii=False
                ),
            },
        ],
        ClinicalEvidence,
    )
    try:
        return _validate_model_facts(evidence, text, source)
    except ValueError as exc:
        raise ProviderUnavailableError("Clinical Parser 模型输出未通过事实来源校验") from exc
