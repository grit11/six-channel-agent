"""来源逻辑回归：覆盖 OR、明确括号、说明隔离及三值边界，不作为临床标注集。"""

import pytest

from backend.app.knowledge.differential import candidate_source_values
from backend.app.nodes.clinical_parser import match_source_phrase, parse_clinical_text
from backend.app.nodes.source_conditions import assess_source_condition


@pytest.mark.parametrize(
    ("phrase", "text", "status"),
    [
        ("汗出或无汗", "无汗", "supports"),
        ("如狂或发狂", "如狂", "supports"),
        ("如狂或发狂", "无如狂", "unknown"),
        ("如狂或发狂", "无如狂，无发狂", "contradicts"),
        ("(无汗且脉浮紧)或(汗出且脉浮缓)", "无汗，脉浮紧", "supports"),
        ("(无汗且脉浮紧)或(汗出且脉浮缓)", "无汗，脉浮", "unknown"),
        ("无汗且(喘或咳嗽)", "无汗，咳嗽", "supports"),
        ("无汗且(喘或咳嗽)", "无汗，无喘", "unknown"),
        ("无汗且(喘或咳嗽)", "无汗，无喘，无咳嗽", "contradicts"),
        ("无汗且喘或咳嗽", "无汗，喘", "unknown"),
        ("小便不利", "小便不利", "supports"),
        ("多小便不利", "小便不利", "supports"),
        ("多小便不利", "无小便不利", "contradicts"),
        ("常口渴或消渴", "口渴", "supports"),
        ("可如狂或发狂", "如狂", "unknown"),
        ("非必见", "口渴", "unknown"),
        ("多小便不利", "昨日小便不利", "unknown"),
        ("不渴", "不渴", "supports"),
        ("不渴", "口渴", "contradicts"),
        ("脉浮紧", "不是脉浮紧", "unknown"),
        ("身疼", "身痛", "supports"),
        ("如狂或发狂", "昨日如狂", "unknown"),
        ("如狂或发狂", "如狂，无如狂", "unknown"),
        ("如狂或发狂", "如狂或发狂", "unknown"),
        ("昨日汗出", "汗出", "unknown"),
        ("治疗前无汗", "无汗", "unknown"),
        ("无汗或者(汗出", "无汗", "unknown"),
        ("无汗或", "无汗", "unknown"),
    ],
)
def test_boolean_source_conditions(phrase: str, text: str, status: str) -> None:
    """部分 OR 分支反证不否认未知分支；混合连接词须有明确括号作用域。"""
    assessment = assess_source_condition(phrase, parse_clinical_text(text))
    assert assessment.status == status
    assert assessment.source_text == phrase
    assert match_source_phrase(phrase, parse_clinical_text(text)) == status


@pytest.mark.parametrize(
    ("phrase", "dimension", "kind"),
    [
        ("汗后身痛", "治疗变化", "treatment"),
        ("治疗前无汗，治疗后汗出", "病程", "temporal"),
        ("先发热后下利", "次序", "temporal"),
        ("无汗而喘", "病机", "mechanism"),
        ("寒邪束表，无汗", "主症", "mechanism"),
        ("得汤反剧", "验证线索", "treatment"),
        ("是否汗出", "追问重点", "question"),
    ],
)
def test_explanations_do_not_supply_current_counterevidence(
    phrase: str, dimension: str, kind: str
) -> None:
    """患者当前症状不能用于确认治疗前后、病机说明或追问本身。"""
    assessment = assess_source_condition(
        phrase, parse_clinical_text("汗出，身痛，下利，喘"), dimension=dimension
    )
    assert assessment.kind == kind
    assert assessment.status == "unknown"
    assert not assessment.alternatives
    assert assessment.unresolved_reason


def test_alias_and_source_subject_attribution() -> None:
    """短方名和后两者仅按当前来源显式候选归属；长说明仍保留完整原文。"""
    candidates = ["葛根黄芩黄连汤证", "麻黄杏仁甘草石膏汤证", "桂枝加厚朴杏子汤证"]
    assert candidate_source_values("葛根芩连汤证以下利为主；后两者以喘为主", candidates) == {
        candidates[0]: "以下利为主",
        candidates[1]: "以喘为主",
        candidates[2]: "以喘为主",
    }
    assert candidate_source_values("汗出而喘", candidates, dimension="麻杏石甘汤") == {
        candidates[1]: "汗出而喘"
    }
    assert candidate_source_values("前者汗出；后者无汗", candidates) == {}
    assert candidate_source_values("汗出；无汗", candidates) == {}
    assert candidate_source_values("脉浮", ["通脉四逆汤证", "白通汤证"], dimension="脉") == {}
    assert (
        candidate_source_values("汗出", ["桂枝汤证", "桂枝汤证（某路径）"], dimension="桂枝汤")
        == {}
    )
