"""S1 原文来源验证测试：拒绝模型编造片段和冲突身份。"""

import pytest

from backend.app.clinical_v2.schemas import S1ExtractionDraft
from backend.app.clinical_v2.verification import ExtractionInvalidError, validate_extraction


def test_rejects_source_text_not_in_original() -> None:
    """模型不能用未出现在病例中的规范术语替换原文。"""
    draft = S1ExtractionDraft(
        findings=[
            {
                "local_id": "x",
                "concept": "无汗",
                "polarity": "negative",
                "time_scope": "current",
                "source_text": "无汗",
            }
        ]
    )
    with pytest.raises(ExtractionInvalidError, match="对齐"):
        validate_extraction("汗始终出不来", draft)


def test_repeated_source_requires_occurrence() -> None:
    """重复原文必须由 occurrence 区分，避免两个事实引用同一位置。"""
    draft = S1ExtractionDraft(
        findings=[
            {
                "local_id": "first",
                "concept": "疼痛",
                "polarity": "positive",
                "time_scope": "current",
                "source_text": "疼痛",
                "occurrence": 0,
            },
            {
                "local_id": "second",
                "concept": "疼痛",
                "polarity": "positive",
                "time_scope": "current",
                "source_text": "疼痛",
                "occurrence": 1,
            },
        ]
    )
    result = validate_extraction("头疼痛，腹疼痛", draft)
    assert result.findings[0].source_span.start != result.findings[1].source_span.start


def test_compound_pulse_span_can_support_distinct_concepts() -> None:
    """同一复合脉象片段可拆成不同概念，但不能重复声明同一概念。"""
    draft = S1ExtractionDraft(
        findings=[
            {
                "local_id": "floating",
                "concept": "脉浮",
                "polarity": "positive",
                "time_scope": "current",
                "source_text": "脉浮而紧",
            },
            {
                "local_id": "tight",
                "concept": "脉紧",
                "polarity": "positive",
                "time_scope": "current",
                "source_text": "脉浮而紧",
            },
        ]
    )
    result = validate_extraction("脉浮而紧", draft)
    assert [item.concept for item in result.findings] == ["脉浮", "脉紧"]
